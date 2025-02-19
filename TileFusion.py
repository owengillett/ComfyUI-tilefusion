import os
import copy
import json
import re
import datetime
import itertools
from typing import List, Optional

import torch
import torch.nn.functional as F

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

# Helper: Resize a tensor image (assumed to be in H×W×C) to a square image of size new_size using bilinear interpolation.
def resize_tensor_image(img: torch.Tensor, new_size: int) -> torch.Tensor:
    # Convert from (H, W, C) to (1, C, H, W)
    img = img.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(img, size=(new_size, new_size), mode='bilinear', align_corners=False)
    # Back to (H, W, C)
    return resized.squeeze(0).permute(1, 2, 0)

# Helper: Build a full 3x3 grid tensor image from a dictionary of eight cell images.
# The center cell remains white.
def build_full_grid_image_tensor(orig: dict, frame_data: dict, cell_size: int) -> torch.Tensor:
    full_h = 3 * cell_size
    full_w = 3 * cell_size
    grid = torch.ones((full_h, full_w, 3), dtype=torch.float32)
    
    positions = ["top_left", "top_middle", "top_right",
                 "middle_left", "middle_right",
                 "bottom_left", "bottom_middle", "bottom_right"]
    
    cells = []
    for pos in positions:
        if orig.get(pos, False):
            cell_mask = resize_tensor_image(frame_data[pos], cell_size)
        else:
            cell_mask = torch.ones((cell_size, cell_size, 3), dtype=torch.float32)
        cells.append(cell_mask)
    
    center = torch.ones((cell_size, cell_size, 3), dtype=torch.float32)
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

# Helper: Build a full mask grid for eight cells based on which inputs were provided.
# For each provided cell the mask is 0 (black) and for not provided it is 1 (white).
# The center cell is always white.
def build_full_grid_mask_tensor(orig: dict, cell_size: int) -> torch.Tensor:
    full_h = 3 * cell_size
    full_w = 3 * cell_size
    grid = torch.ones((full_h, full_w), dtype=torch.float32)
    
    positions = ["top_left", "top_middle", "top_right",
                 "middle_left", "middle_right",
                 "bottom_left", "bottom_middle", "bottom_right"]
    
    cells = []
    for pos in positions:
        if orig.get(pos, False):
            cell_mask = torch.zeros((cell_size, cell_size), dtype=torch.float32)
        else:
            cell_mask = torch.ones((cell_size, cell_size), dtype=torch.float32)
        cells.append(cell_mask)
    
    center = torch.ones((cell_size, cell_size), dtype=torch.float32)
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

# Helper: Centrally crop a tensor image (either H×W×C or H×W) if its dimensions exceed crop_max_size.
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

# Helper: Get the length of a sequence (list or tensor).
def seq_length(seq):
    if seq is None:
        return 0
    if isinstance(seq, torch.Tensor):
        return seq.size(0)
    try:
        return len(seq)
    except Exception:
        return 0

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
                "top_left": (imageOrLatent, {"default": None}),
                "top_middle": (imageOrLatent, {"default": None}),
                "top_right": (imageOrLatent, {"default": None}),
                "middle_left": (imageOrLatent, {"default": None}),
                "middle_right": (imageOrLatent, {"default": None}),
                "bottom_left": (imageOrLatent, {"default": None}),
                "bottom_middle": (imageOrLatent, {"default": None}),
                "bottom_right": (imageOrLatent, {"default": None}),
            }
        }

    # Updated return types: now includes a BOOLEAN for use_inpaint.
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "BOOLEAN")
    RETURN_NAMES = ("combined_sequence", "mask_sequence", "tiling", "use_inpaint")
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
        # Determine which inputs are provided.
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
        
        # Determine if we have any actual image input.
        use_inpaint = any(orig.values())
        
        # Build a dictionary of sequences.
        seqs = {
            "top_left": top_left,
            "top_middle": top_middle,
            "top_right": top_right,
            "middle_left": middle_left,
            "middle_right": middle_right,
            "bottom_left": bottom_left,
            "bottom_middle": bottom_middle,
            "bottom_right": bottom_right,
        }
        
        provided_counts = [seq_length(seq) for seq in seqs.values() if seq is not None]
        if provided_counts:
            min_frames = min(provided_counts)
        else:
            # If no sequences provided, create a white grid image sequence.
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
            # When no image input is provided, use_inpaint is False.
            return (combined_tensor, mask_tensor, tiling, False)
        
        combined_frames = []
        mask_frames = []
        pbar = ProgressBar(min_frames)
        for i in range(min_frames):
            # For each frame, gather the i-th image from each sequence.
            frame_data = {}
            for pos in ["top_left", "top_middle", "top_right",
                        "middle_left", "middle_right",
                        "bottom_left", "bottom_middle", "bottom_right"]:
                frame_data[pos] = seqs[pos][i] if seq_length(seqs[pos]) > i else None
            full_img = build_full_grid_image_tensor(orig, frame_data, cell_size)
            full_mask = build_full_grid_mask_tensor(orig, cell_size)
            
            # Determine which rows and columns are active.
            active_rows = []
            if orig["top_left"] or orig["top_middle"] or orig["top_right"]:
                active_rows.append(0)
            active_rows.append(1)
            if orig["bottom_left"] or orig["bottom_middle"] or orig["bottom_right"]:
                active_rows.append(2)
            active_cols = []
            if orig["top_left"] or orig["middle_left"] or orig["bottom_left"]:
                active_cols.append(0)
            active_cols.append(1)
            if orig["top_right"] or orig["middle_right"] or orig["bottom_right"]:
                active_cols.append(2)
            
            top_idx = min(active_rows)
            left_idx = min(active_cols)
            bottom_idx = max(active_rows) + 1
            right_idx = max(active_cols) + 1
            
            grid_img = full_img[top_idx*cell_size : bottom_idx*cell_size,
                                left_idx*cell_size : right_idx*cell_size, :]
            grid_mask = full_mask[top_idx*cell_size : bottom_idx*cell_size,
                                  left_idx*cell_size : right_idx*cell_size]
            if crop_max_size > 0:
                grid_img = central_crop_tensor(grid_img, crop_max_size)
                grid_mask = central_crop_tensor(grid_mask, crop_max_size)
            
            combined_frames.append(grid_img)
            mask_frames.append(grid_mask)
            pbar.update(1)
        
        combined_tensor = torch.stack(combined_frames, dim=0)
        mask_tensor = torch.stack(mask_frames, dim=0)
        
        # Tiling adjustment logic.
        h_viable = (orig.get("top_middle", False) or orig.get("bottom_middle", False)) and not (
            orig.get("top_left", False) or orig.get("top_right", False) or 
            orig.get("middle_left", False) or orig.get("middle_right", False) or 
            orig.get("bottom_left", False) or orig.get("bottom_right", False)
        )
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
        
        return (combined_tensor, mask_tensor, tiling, use_inpaint)
