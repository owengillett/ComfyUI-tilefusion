import os
import copy
import json
import re
import datetime
import itertools
from typing import List, Optional

import torch
import numpy as np
from PIL import Image, ExifTags, PngImagePlugin

import folder_paths
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

# Helper: Convert an input (PIL image, numpy array, or torch.Tensor) to a PIL Image.
def to_pil(im):
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    elif isinstance(im, np.ndarray):
        if im.max() <= 1.0:
            im = (im * 255).astype(np.uint8)
        else:
            im = im.astype(np.uint8)
        return Image.fromarray(im).convert("RGB")
    elif isinstance(im, torch.Tensor):
        im = im.cpu().detach()
        if im.ndim == 4:
            im = im[0]
        if im.ndim == 3 and im.shape[0] <= 4:
            im = im.permute(1, 2, 0)
        im = im.numpy()
        if im.max() <= 1.0:
            im = (im * 255).astype(np.uint8)
        else:
            im = im.astype(np.uint8)
        return Image.fromarray(im).convert("RGB")
    else:
        raise Exception("Unsupported image format: " + str(type(im)))

# Helper: Build a full 3x3 grid image from a dictionary of eight cell images.
# The positions are fixed: top_left, top_middle, top_right, middle_left, middle_right, bottom_left, bottom_middle, bottom_right.
# The center cell is always white.
def build_full_grid_image(frame_data: dict, cell_size: int) -> Image.Image:
    # Create a white background image for the full grid (3 x 3 cells).
    grid = Image.new("RGB", (3 * cell_size, 3 * cell_size), (255, 255, 255))
    
    # Mapping cell positions to grid coordinates.
    pos_coords = {
         "top_left": (0, 0),
         "top_middle": (cell_size, 0),
         "top_right": (2 * cell_size, 0),
         "middle_left": (0, cell_size),
         "middle_right": (2 * cell_size, cell_size),
         "bottom_left": (0, 2 * cell_size),
         "bottom_middle": (cell_size, 2 * cell_size),
         "bottom_right": (2 * cell_size, 2 * cell_size),
    }
    
    # For each cell position, if an image is provided, convert and paste it.
    for pos, coord in pos_coords.items():
        if pos in frame_data and frame_data[pos] is not None:
            cell_img = to_pil(frame_data[pos]).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            grid.paste(cell_img, coord)
    
    return grid


# Helper: Build a full mask grid for eight cells based on original provided flags.
# For each cell, if originally provided then the mask cell is black (0); otherwise white (255).
# The center cell is always white.
def build_full_grid_mask(orig: dict, cell_size: int) -> Image.Image:
    positions = ["top_left", "top_middle", "top_right",
                 "middle_left", "middle_right",
                 "bottom_left", "bottom_middle", "bottom_right"]
    cells = []
    for pos in positions:
        if orig.get(pos, False):
            cell_mask = Image.new("L", (cell_size, cell_size), 0)
        else:
            cell_mask = Image.new("L", (cell_size, cell_size), 255)
        cells.append(cell_mask)
    center = Image.new("L", (cell_size, cell_size), 255)
    grid = Image.new("L", (3 * cell_size, 3 * cell_size), 255)
    grid.paste(cells[0], (0, 0))
    grid.paste(cells[1], (cell_size, 0))
    grid.paste(cells[2], (2 * cell_size, 0))
    grid.paste(cells[3], (0, cell_size))
    grid.paste(center, (cell_size, cell_size))
    grid.paste(cells[4], (2 * cell_size, cell_size))
    grid.paste(cells[5], (0, 2 * cell_size))
    grid.paste(cells[6], (cell_size, 2 * cell_size))
    grid.paste(cells[7], (2 * cell_size, 2 * cell_size))
    return grid

# Helper: Centrally crop an image if its dimensions exceed crop_max_size.
def central_crop(img: Image.Image, crop_max_size: float) -> Image.Image:
    if crop_max_size <= 0:
        return img
    w, h = img.size
    new_w = min(w, int(crop_max_size))
    new_h = min(h, int(crop_max_size))
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))

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
            return (torch.tensor([]), torch.tensor([]), tiling)
        # # For each cell, if empty, substitute with a white sequence.
        # for key, seq in seqs.items():
        #     if seq_length(seq) == 0:
        #         white = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
        #         seqs[key] = [np.array(white).astype(np.float32)/255.0 for _ in range(min_frames)]
        #     else:
        #         if not isinstance(seq, torch.Tensor):
        #             seqs[key] = seq[:min_frames]
        #         else:
        #             seqs[key] = seq[:min_frames]
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
            full_img = build_full_grid_image(frame_data, cell_size)
            full_mask = build_full_grid_mask(orig, cell_size)
            # Determine active rows and columns.
            # Rows: index 0 (top) active if any of top_left, top_middle, top_right is True.
            # Index 1 (middle) always active.
            # Index 2 (bottom) active if any of bottom_left, bottom_middle, bottom_right is True.
            active_rows = []
            if orig["top_left"] or orig["top_middle"] or orig["top_right"]:
                active_rows.append(0)
            active_rows.append(1)
            if orig["bottom_left"] or orig["bottom_middle"] or orig["bottom_right"]:
                active_rows.append(2)
            # Columns: index 0 (left) active if any of top_left, middle_left, bottom_left is True.
            # Index 1 (middle) always active.
            # Index 2 (right) active if any of top_right, middle_right, bottom_right is True.
            active_cols = []
            if orig["top_left"] or orig["middle_left"] or orig["bottom_left"]:
                active_cols.append(0)
            active_cols.append(1)
            if orig["top_right"] or orig["middle_right"] or orig["bottom_right"]:
                active_cols.append(2)
            # Compute crop box in the full grid image (which is 3*cell_size by 3*cell_size).
            left = min(active_cols) * cell_size
            upper = min(active_rows) * cell_size
            right = (max(active_cols)+1) * cell_size
            lower = (max(active_rows)+1) * cell_size
            grid_img = full_img.crop((left, upper, right, lower))
            grid_mask = full_mask.crop((left, upper, right, lower))
            if crop_max_size > 0:
                grid_img = central_crop(grid_img, crop_max_size)
                grid_mask = central_crop(grid_mask, crop_max_size)
            grid_np = np.array(grid_img).astype(np.float32) / 255.0
            mask_np = np.array(grid_mask).astype(np.float32) / 255.0
            combined_frames.append(grid_np)
            mask_frames.append(mask_np)
            pbar.update(1)
        combined_tensor = torch.from_numpy(np.stack(combined_frames))
        mask_tensor = torch.from_numpy(np.stack(mask_frames))
        
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