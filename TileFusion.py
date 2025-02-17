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

# Helper: Convert an input (PIL Image, numpy array, or torch.Tensor) to a PIL Image.
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

# Helper: Dynamically tile provided cells into a grid.
# The positions correspond to:
#   Top row: "top_left", "top_middle", "top_right"
#   Middle row: "middle_left", "center", "middle_right" (center is always blank)
#   Bottom row: "bottom_left", "bottom_middle", "bottom_right"
# Only rows/columns that contain at least one provided input (per original input) are included.
def build_grid_for_frame(frame_data: dict, cell_size: int) -> (Image.Image, Image.Image):
    # frame_data is a dict mapping positions to an image (or None).
    # Positions: top_left, top_middle, top_right, middle_left, middle_right, bottom_left, bottom_middle, bottom_right.
    # Determine original provided flags.
    orig = {}
    for pos in ["top_left", "top_middle", "top_right",
                "middle_left", "middle_right",
                "bottom_left", "bottom_middle", "bottom_right"]:
        # Consider it provided if the original sequence (before substitution) was non-empty.
        orig[pos] = (frame_data.get(pos) is not None)
    # Determine which rows to include:
    include_top = orig["top_left"] or orig["top_middle"] or orig["top_right"]
    include_bottom = orig["bottom_left"] or orig["bottom_middle"] or orig["bottom_right"]
    # Columns:
    include_left = orig["top_left"] or orig["middle_left"] or orig["bottom_left"]
    include_right = orig["top_right"] or orig["middle_right"] or orig["bottom_right"]
    # Always include central row and column.
    # Build rows as lists of cell keys.
    rows = []
    if include_top:
        rows.append(["top_left", "top_middle", "top_right"])
    # Middle row: note that the center cell is always "center" (blank)
    rows.append(["middle_left", "center", "middle_right"])
    if include_bottom:
        rows.append(["bottom_left", "bottom_middle", "bottom_right"])
    # Now, for each row, drop cells in columns that are not included.
    final_rows = []
    for row in rows:
        new_row = []
        for pos in row:
            if pos == "center":
                new_row.append(pos)
            else:
                if pos.endswith("left") and not include_left:
                    continue
                if pos.endswith("right") and not include_right:
                    continue
                new_row.append(pos)
        # Only add non-empty rows.
        if new_row:
            final_rows.append(new_row)
    # Determine output grid size.
    num_rows = len(final_rows)
    num_cols = max(len(r) for r in final_rows) if final_rows else 1
    # Create a blank canvas for grid image (RGB) and mask (grayscale "L").
    grid_img = Image.new("RGB", (num_cols * cell_size, num_rows * cell_size), (255, 255, 255))
    grid_mask = Image.new("L", (num_cols * cell_size, num_rows * cell_size), 255)
    # For each cell in final_rows, determine the image.
    for r_idx, row in enumerate(final_rows):
        for c_idx, pos in enumerate(row):
            if pos == "center":
                cell_img = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
                cell_mask = Image.new("L", (cell_size, cell_size), 255)
            else:
                # Use the provided image if available; otherwise white.
                if frame_data.get(pos) is not None:
                    cell_img = to_pil(frame_data[pos]).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                    cell_mask = Image.new("L", (cell_size, cell_size), 0)  # black where input provided.
                else:
                    cell_img = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
                    cell_mask = Image.new("L", (cell_size, cell_size), 255)
            grid_img.paste(cell_img, (c_idx * cell_size, r_idx * cell_size))
            grid_mask.paste(cell_mask, (c_idx * cell_size, r_idx * cell_size))
    return grid_img, grid_mask

# Helper: Centrally crop an image if its dimension exceeds crop_max_size.
def central_crop(img: Image.Image, crop_max_size: float) -> Image.Image:
    if crop_max_size <= 0:
        return img
    w, h = img.size
    new_w = min(w, int(crop_max_size))
    new_h = min(h, int(crop_max_size))
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))

# Helper: Safely get length of a sequence.
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

    # Outputs: combined image sequence, mask sequence, and tiling string.
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
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
        # Record original provided status before substitution.
        orig = {
            "top_left": bool(top_left and seq_length(top_left) > 0),
            "top_middle": bool(top_middle and seq_length(top_middle) > 0),
            "top_right": bool(top_right and seq_length(top_right) > 0),
            "middle_left": bool(middle_left and seq_length(middle_left) > 0),
            "middle_right": bool(middle_right and seq_length(middle_right) > 0),
            "bottom_left": bool(bottom_left and seq_length(bottom_left) > 0),
            "bottom_middle": bool(bottom_middle and seq_length(bottom_middle) > 0),
            "bottom_right": bool(bottom_right and seq_length(bottom_right) > 0),
        }
        # Create a dictionary for the eight optional inputs.
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
        # Determine minimum frame count among sequences that were originally provided.
        provided_counts = [seq_length(seq) for seq in seqs.values() if seq_length(seq) > 0]
        if provided_counts:
            min_frames = min(provided_counts)
        else:
            # If none provided, return empty outputs.
            return (torch.tensor([]), torch.tensor([]), tiling)
        # For each cell, if empty, substitute with a white sequence of length min_frames.
        for key, seq in seqs.items():
            if seq_length(seq) == 0:
                white = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
                seqs[key] = [np.array(white).astype(np.float32)/255.0 for _ in range(min_frames)]
            else:
                if not isinstance(seq, torch.Tensor):
                    seqs[key] = seq[:min_frames]
                else:
                    seqs[key] = seq[:min_frames]
        # Determine grid structure.
        include_top = orig["top_left"] or orig["top_middle"] or orig["top_right"]
        include_bottom = orig["bottom_left"] or orig["bottom_middle"] or orig["bottom_right"]
        include_left = orig["top_left"] or orig["middle_left"] or orig["bottom_left"]
        include_right = orig["top_right"] or orig["middle_right"] or orig["bottom_right"]
        # Build row order: always include middle row.
        rows = []
        if include_top:
            rows.append(["top_left", "top_middle", "top_right"])
        rows.append(["middle_left", "center", "middle_right"])
        if include_bottom:
            rows.append(["bottom_left", "bottom_middle", "bottom_right"])
        # For each row, drop cells in columns not included.
        final_rows = []
        for row in rows:
            new_row = []
            for pos in row:
                if pos == "center":
                    new_row.append(pos)
                else:
                    if pos.endswith("left") and not include_left:
                        continue
                    if pos.endswith("right") and not include_right:
                        continue
                    new_row.append(pos)
            if new_row:
                final_rows.append(new_row)
        # Determine grid output dimensions.
        num_rows = len(final_rows)
        num_cols = max(len(r) for r in final_rows) if final_rows else 1

        combined_frames = []
        mask_frames = []
        pbar = ProgressBar(min_frames)
        for i in range(min_frames):
            # For each frame, build a dict mapping cell positions to the i-th frame.
            frame_data = {}
            for pos in ["top_left", "top_middle", "top_right",
                        "middle_left", "middle_right",
                        "bottom_left", "bottom_middle", "bottom_right"]:
                frame_data[pos] = seqs[pos][i] if seq_length(seqs[pos]) > 0 else None
            # Build grid: for positions that are in final_rows.
            # For cells not present in the grid (because their column is omitted), they will not appear.
            # For "center", always use a white cell.
            grid_cells = {}
            for r in final_rows:
                for pos in r:
                    if pos == "center":
                        grid_cells[pos] = None  # will be handled as blank
                    else:
                        grid_cells[pos] = frame_data.get(pos)
            # Now, build the grid image and mask.
            # We'll reconstruct the grid row-by-row.
            row_imgs = []
            row_masks = []
            for row in final_rows:
                cell_imgs = []
                cell_masks = []
                for pos in row:
                    if pos == "center":
                        cell_imgs.append(Image.new("RGB", (cell_size, cell_size), (255,255,255)))
                        cell_masks.append(Image.new("L", (cell_size, cell_size), 255))
                    else:
                        if grid_cells[pos] is not None:
                            cell_img = to_pil(grid_cells[pos]).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                            cell_imgs.append(cell_img)
                            cell_masks.append(Image.new("L", (cell_size, cell_size), 0))
                        else:
                            cell_imgs.append(Image.new("RGB", (cell_size, cell_size), (255,255,255)))
                            cell_masks.append(Image.new("L", (cell_size, cell_size), 255))
                # Concatenate cells horizontally.
                row_img = Image.new("RGB", (len(cell_imgs)*cell_size, cell_size))
                row_mask = Image.new("L", (len(cell_masks)*cell_size, cell_size), 255)
                for idx, im in enumerate(cell_imgs):
                    row_img.paste(im, (idx*cell_size, 0))
                for idx, im in enumerate(cell_masks):
                    row_mask.paste(im, (idx*cell_size, 0))
                row_imgs.append(row_img)
                row_masks.append(row_mask)
            # Concatenate rows vertically.
            grid_img = Image.new("RGB", (num_cols*cell_size, num_rows*cell_size))
            grid_mask = Image.new("L", (num_cols*cell_size, num_rows*cell_size), 255)
            for idx, im in enumerate(row_imgs):
                grid_img.paste(im, (0, idx*cell_size))
            for idx, im in enumerate(row_masks):
                grid_mask.paste(im, (0, idx*cell_size))
            # If crop_max_size is specified and greater than 0, centrally crop the grid.
            if crop_max_size > 0:
                grid_img = central_crop(grid_img, crop_max_size)
                grid_mask = central_crop(grid_mask, crop_max_size)
            # Convert images to normalized numpy arrays.
            grid_np = np.array(grid_img).astype(np.float32) / 255.0
            mask_np = np.array(grid_mask).astype(np.float32) / 255.0
            combined_frames.append(grid_np)
            mask_frames.append(mask_np)
            pbar.update(1)
        # Stack the frames into tensors.
        combined_tensor = torch.from_numpy(np.stack(combined_frames))
        mask_tensor = torch.from_numpy(np.stack(mask_frames))
        return (combined_tensor, mask_tensor, tiling)

# Helper: Centrally crop an image.
def central_crop(img: Image.Image, crop_max_size: float) -> Image.Image:
    if crop_max_size <= 0:
        return img
    w, h = img.size
    new_w = min(w, int(crop_max_size))
    new_h = min(h, int(crop_max_size))
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))
