import os
import copy
import json
import re
import datetime
import itertools
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np

from comfy.utils import ProgressBar

# Minimal inlined definitions for node interface types.
class MultiInput(str):
    def __new__(cls, string, allowed_types="*"):
        res = super().__new__(cls, string)
        res.allowed_types = allowed_types
        return res
    def __ne__(self, other):
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types

imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])
floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])

# Helper: Convert an input (numpy array or torch.Tensor) to a tensor image in HxWxC float32 [0,1].
def to_tensor_image(im):
    if isinstance(im, torch.Tensor):
        im = im.detach().cpu()
        # If a batch is provided, take the first item.
        if im.ndim == 4:
            im = im[0]
        # If channels are first, move them to last.
        if im.ndim == 3 and im.shape[0] <= 4:
            im = im.permute(1, 2, 0)
        im = im.to(torch.float32)
        # If values are > 1 assume they are in [0,255]
        if im.max() > 1.0:
            im = im / 255.0
        return im
    elif isinstance(im, np.ndarray):
        im = torch.from_numpy(im)
        if im.ndim == 3 and im.shape[0] <= 4:
            im = im.permute(1, 2, 0)
        im = im.to(torch.float32)
        if im.max() > 1.0:
            im = im / 255.0
        return im
    else:
        raise Exception("Unsupported image format: " + str(type(im)))

# Helper: Resize a tensor image (HxWxC) to a new size using bilinear interpolation.
def resize_tensor_image(img, new_size: int):
    # Convert from (H, W, C) to (1, C, H, W)
    img = img.permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(img, size=(new_size, new_size), mode='bilinear', align_corners=False)
    # Back to (H, W, C)
    img = img.squeeze(0).permute(1, 2, 0)
    return img

# Helper: Build a full 3x3 grid tensor image from the provided dictionary of eight cell images.
# The center cell is always white.
def build_full_grid_image_tensor(frame_data: dict, cell_size: int) -> torch.Tensor:
    # Create a white background tensor in HxWx3 (values 1.0 represent white).
    full_h = 3 * cell_size
    full_w = 3 * cell_size
    grid = torch.ones((full_h, full_w, 3), dtype=torch.float32)
    
    # Mapping positions to top-left coordinates.
    # Note: In tensor indexing the first coordinate is y (row) and second is x (column).
    pos_coords = {
         "top_left": (0, 0),
         "top_middle": (0, cell_size),
         "top_right": (0, 2 * cell_size),
         "middle_left": (cell_size, 0),
         "middle_right": (cell_size, 2 * cell_size),
         "bottom_left": (2 * cell_size, 0),
         "bottom_middle": (2 * cell_size, cell_size),
         "bottom_right": (2 * cell_size, 2 * cell_size),
    }
    
    for pos, (y, x) in pos_coords.items():
        if pos in frame_data and frame_data[pos] is not None:
            # Convert input to a tensor image and resize.
            cell_img = to_tensor_image(frame_data[pos])
            cell_img = resize_tensor_image(cell_img, cell_size)
            # Paste cell image into grid.
            grid[y:y+cell_size, x:x+cell_size, :] = cell_img
    return grid

# Helper: Build a full mask grid for eight cells based on original provided flags.
# For each cell that was provided, the corresponding cell is set to 0 (black) else 1 (white).
# The center cell is always white.
def build_full_grid_mask_tensor(orig: dict, cell_size: int) -> torch.Tensor:
    full_h = 3 * cell_size
    full_w = 3 * cell_size
    # Start with a full white mask.
    grid = torch.ones((full_h, full_w), dtype=torch.float32)
    
    # Order of cells (same as used in the original): top_left, top_middle, top_right, middle_left, middle_right, bottom_left, bottom_middle, bottom_right.
    # We'll build each cell as a constant tensor.
    cells = []
    positions = ["top_left", "top_middle", "top_right",
                 "middle_left", "middle_right",
                 "bottom_left", "bottom_middle", "bottom_right"]
    for pos in positions:
        if orig.get(pos, False):
            # Provided cell: mask black (0)
            cell_mask = torch.zeros((cell_size, cell_size), dtype=torch.float32)
        else:
            # Not provided: white (1)
            cell_mask = torch.ones((cell_size, cell_size), dtype=torch.float32)
        cells.append(cell_mask)
    
    # The center cell is forced white.
    center = torch.ones((cell_size, cell_size), dtype=torch.float32)
    # Paste the cells into the grid. (Coordinates are given as (y, x))
    grid[0:cell_size, 0:cell_size] = cells[0]           # top_left
    grid[0:cell_size, cell_size:2*cell_size] = cells[1]    # top_middle
    grid[0:cell_size, 2*cell_size:3*cell_size] = cells[2]  # top_right
    grid[cell_size:2*cell_size, 0:cell_size] = cells[3]    # middle_left
    grid[cell_size:2*cell_size, cell_size:2*cell_size] = center  # center forced white
    grid[cell_size:2*cell_size, 2*cell_size:3*cell_size] = cells[4]  # middle_right
    grid[2*cell_size:3*cell_size, 0:cell_size] = cells[5]  # bottom_left
    grid[2*cell_size:3*cell_size, cell_size:2*cell_size] = cells[6]  # bottom_middle
    grid[2*cell_size:3*cell_size, 2*cell_size:3*cell_size] = cells[7]  # bottom_right
    return grid

# Helper: Centrally crop a tensor image (HxWxC or HxW) if its dimensions exceed crop_max_size.
def central_crop_tensor(img: torch.Tensor, crop_max_size: float) -> torch.Tensor:
    if crop_max_size <= 0:
        return img
    if img.ndim == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    new_h = min(h, int(crop_max_size))
    new_w = min(w, int(crop_max_size))
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    if img.ndim == 3:
        return img[top:top+new_h, left:left+new_w, :]
    else:
        return img[top:top+new_h, left:left+new_w]

# Helper: Safely get the length of a sequence (list or torch.Tensor).
def seq_length(seq):
    if seq is None:
        return 0
    if isinstance(seq, torch.Tensor):
        return seq.size(0)
    try:
        return len(seq)
    except Exception:
        return 0

# The main node.
class VideoGridCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cell_size": ("INT", {"default": 128, "min": 16, "max": 4096, "step": 16}),
                "crop_max_size": ("FLOAT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "tiling": (["enable", "x_only", "y_only", "disable"],),
            },
            "optional": {
                "top_left": (imageOrLatent, {"default": []}),
                "top_middle": (imageOrLatent, {"default": []}),
                "top_right": (imageOrLatent, {"default": []}),
                "middle_left": (imageOrLatent, {"default": []}),
                "middle_right": (imageOrLatent, {"default": []}),
                "bottom_left": (imageOrLatent, {"default": []}),
                "bottom_middle": (imageOrLatent, {"default": []}),
                "bottom_right": (imageOrLatent, {"default": []}),
            }
        }

    # Outputs: combined image sequence (type IMAGE), mask sequence (type MASK), and tiling (STRING).
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("combined_sequence", "mask_sequence", "tiling")
    CATEGORY = "custom"
    FUNCTION = "combine_grid"

    def combine_grid(
        self,
        cell_size: int,
        crop_max_size: float,
        tiling,
        top_left=None,
        top_middle=None,
        top_right=None,
        middle_left=None,
        middle_right=None,
        bottom_left=None,
        bottom_middle=None,
        bottom_right=None,
    ):
        # Build original flags from raw inputs.
        orig = {
            "top_left": True if (top_left is not None and seq_length(top_left) > 0) else False,
            "top_middle": True if (top_middle is not None and seq_length(top_middle) > 0) else False,
            "top_right": True if (top_right is not None and seq_length(top_right) > 0) else False,
            "middle_left": True if (middle_left is not None and seq_length(middle_left) > 0) else False,
            "middle_right": True if (middle_right is not None and seq_length(middle_right) > 0) else False,
            "bottom_left": True if (bottom_left is not None and seq_length(bottom_left) > 0) else False,
            "bottom_middle": True if (bottom_middle is not None and seq_length(bottom_middle) > 0) else False,
            "bottom_right": True if (bottom_right is not None and seq_length(bottom_right) > 0) else False,
        }
        # Build dictionary for optional inputs.
        seqs = {
            "top_left": top_left if top_left is not None else [],
            "top_middle": top_middle if top_middle is not None else [],
            "top_right": top_right if top_right is not None else [],
            "middle_left": middle_left if middle_left is not None else [],
            "middle_right": middle_right if middle_right is not None else [],
            "bottom_left": bottom_left if bottom_left is not None else [],
            "bottom_middle": bottom_middle if bottom_middle is not None else [],
            "bottom_right": bottom_right if bottom_right is not None else [],
        }
        provided_counts = [seq_length(seq) for seq in seqs.values() if seq_length(seq) > 0]
        if provided_counts:
            min_frames = min(provided_counts)
        else:
            # All sequences are null: Return a white image sequence of length 16 and a white mask sequence of length 16.
            min_frames = 16
            full_h = 3 * cell_size
            full_w = 3 * cell_size
            white_full = torch.ones((full_h, full_w, 3), dtype=torch.float32)
            white_mask_full = torch.ones((full_h, full_w), dtype=torch.float32)
            if crop_max_size > 0:
                white_full = central_crop_tensor(white_full, crop_max_size)
                white_mask_full = central_crop_tensor(white_mask_full, crop_max_size)
            combined_tensor = white_full.unsqueeze(0).repeat(16, 1, 1, 1)
            mask_tensor = white_mask_full.unsqueeze(0).repeat(16, 1, 1)
            return (combined_tensor, mask_tensor, tiling)
            
        combined_frames = []
        mask_frames = []
        pbar = ProgressBar(min_frames)
        for i in range(min_frames):
            # Build a dict mapping each cell to its i-th frame.
            frame_data = {}
            for pos in ["top_left", "top_middle", "top_right",
                        "middle_left", "middle_right",
                        "bottom_left", "bottom_middle", "bottom_right"]:
                frame_data[pos] = seqs[pos][i] if seq_length(seqs[pos]) > 0 else None
            # Build full grid image and full grid mask.
            full_img = build_full_grid_image_tensor(frame_data, cell_size)
            full_mask = build_full_grid_mask_tensor(orig, cell_size)
            # Determine active rows and columns.
            # For rows: index 0 (top) active if any of top_left, top_middle, top_right is provided.
            # Index 1 (middle) always active.
            # Index 2 (bottom) active if any of bottom_left, bottom_middle, bottom_right is provided.
            active_rows = []
            if orig["top_left"] or orig["top_middle"] or orig["top_right"]:
                active_rows.append(0)
            active_rows.append(1)
            if orig["bottom_left"] or orig["bottom_middle"] or orig["bottom_right"]:
                active_rows.append(2)
            # For columns: index 0 (left) active if any of top_left, middle_left, bottom_left is provided.
            # Index 1 (middle) always active.
            # Index 2 (right) active if any of top_right, middle_right, bottom_right is provided.
            active_cols = []
            if orig["top_left"] or orig["middle_left"] or orig["bottom_left"]:
                active_cols.append(0)
            active_cols.append(1)
            if orig["top_right"] or orig["middle_right"] or orig["bottom_right"]:
                active_cols.append(2)
            # Compute crop box in the full grid image (grid shape is 3*cell_size x 3*cell_size).
            top_idx = min(active_rows)
            left_idx = min(active_cols)
            bottom_idx = max(active_rows) + 1
            right_idx = max(active_cols) + 1
            # Slicing: note that the full image is in (H, W, C) so indices are in multiples of cell_size.
            grid_img = full_img[top_idx*cell_size : bottom_idx*cell_size,
                                left_idx*cell_size : right_idx*cell_size, :]
            grid_mask = full_mask[top_idx*cell_size : bottom_idx*cell_size,
                                  left_idx*cell_size : right_idx*cell_size]
            if crop_max_size > 0:
                grid_img = central_crop_tensor(grid_img, crop_max_size)
                grid_mask = central_crop_tensor(grid_mask, crop_max_size)
            combined_frames.append(full_img)
            mask_frames.append(grid_mask)
            pbar.update(1)
        # Stack the frames into tensors.
        combined_tensor = torch.stack(combined_frames, dim=0)
        mask_tensor = torch.stack(mask_frames, dim=0)
        
        # Tiling adjustment logic.
        # For horizontal viability: only inputs in top_middle and bottom_middle are allowed.
        h_viable = (orig.get("top_middle", False) or orig.get("bottom_middle", False)) and not (
            orig.get("top_left", False) or orig.get("top_right", False) or 
            orig.get("middle_left", False) or orig.get("middle_right", False) or 
            orig.get("bottom_left", False) or orig.get("bottom_right", False)
        )
        # For vertical viability: only inputs in middle_left and middle_right are allowed.
        v_viable = (orig.get("middle_left", False) or orig.get("middle_right", False)) and not (
            orig.get("top_left", False) or orig.get("bottom_left", False) or 
            orig.get("top_middle", False) or orig.get("top_right", False) or 
            orig.get("bottom_middle", False) or orig.get("bottom_right", False)
        )
        user_tiling = tiling
        if user_tiling != "disable":
            if user_tiling == "enable":
                if not h_viable and v_viable:
                    tiling = "y_only"
                elif not v_viable and h_viable:
                    tiling = "x_only"
                elif not h_viable and not v_viable:
                    tiling = "disable"
                else:
                    tiling = "enable"
            elif user_tiling == "x_only":
                tiling = "x_only" if h_viable else "disable"
            elif user_tiling == "y_only":
                tiling = "y_only" if v_viable else "disable"
        return (combined_tensor, mask_tensor, tiling)
