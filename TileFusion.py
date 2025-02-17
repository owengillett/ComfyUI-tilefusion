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

# Minimal inlined definitions to replace external VHS utils dependency
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

# Helper: Convert an input (PIL image, numpy array, or torch.Tensor) to a PIL image.
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

# Helper: Create a 3x3 grid from 8 images and one blank center.
def tile_images_grid8(images: List, cell_size: int) -> Image.Image:
    # images: [top_left, top_middle, top_right,
    #          middle_left, middle_right,
    #          bottom_left, bottom_middle, bottom_right]
    if len(images) != 8:
        raise Exception("Expected 8 images, got " + str(len(images)))
    # Resize each image to (cell_size, cell_size)
    resized = [to_pil(im).resize((cell_size, cell_size), Image.ANTIALIAS) for im in images]
    # Create a blank (white) cell for the center.
    blank = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
    # Create a new image for the 3x3 grid.
    grid = Image.new("RGB", (3 * cell_size, 3 * cell_size))
    # Top row.
    grid.paste(resized[0], (0, 0))
    grid.paste(resized[1], (cell_size, 0))
    grid.paste(resized[2], (2 * cell_size, 0))
    # Middle row: left, center (blank), right.
    grid.paste(resized[3], (0, cell_size))
    grid.paste(blank, (cell_size, cell_size))
    grid.paste(resized[4], (2 * cell_size, cell_size))
    # Bottom row.
    grid.paste(resized[5], (0, 2 * cell_size))
    grid.paste(resized[6], (cell_size, 2 * cell_size))
    grid.paste(resized[7], (2 * cell_size, 2 * cell_size))
    return grid

# Helper: Create a mask grid for 8 cells.
# For each cell, if the corresponding input is provided (non-empty) then mask = black (0), else white (255).
def tile_mask_grid8(provided: List[bool], cell_size: int) -> Image.Image:
    # provided: list of 8 booleans in order as in tile_images_grid8.
    # Black cell: 0, White cell: 255.
    black_cell = Image.new("L", (cell_size, cell_size), 0)
    white_cell = Image.new("L", (cell_size, cell_size), 255)
    cells = []
    for flag in provided:
        cells.append(black_cell if flag else white_cell)
    # Center cell is always white.
    center = white_cell
    grid = Image.new("L", (3 * cell_size, 3 * cell_size), 255)
    # Top row.
    grid.paste(cells[0], (0, 0))
    grid.paste(cells[1], (cell_size, 0))
    grid.paste(cells[2], (2 * cell_size, 0))
    # Middle row.
    grid.paste(cells[3], (0, cell_size))
    grid.paste(center, (cell_size, cell_size))
    grid.paste(cells[4], (2 * cell_size, cell_size))
    # Bottom row.
    grid.paste(cells[5], (0, 2 * cell_size))
    grid.paste(cells[6], (cell_size, 2 * cell_size))
    grid.paste(cells[7], (2 * cell_size, 2 * cell_size))
    return grid

class VideoGridCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cell_size": ("INT", {"default": 128, "min": 16, "max": 4096, "step": 16}),
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

    # We now return two outputs: the combined image sequence and the mask sequence.
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("combined_sequence", "mask_sequence", "tiling")
    CATEGORY = "custom"
    FUNCTION = "combine_grid"

    def combine_grid(
        self,
        cell_size: int,
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
        # For each optional sequence input, if it is None or empty, substitute a white sequence.
        # Determine the minimum frame count among provided sequences.
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
        # Filter out sequences that are provided (non-empty lists)
        provided_counts = [len(seq) for seq in seqs.values() if seq and len(seq) > 0]
        if provided_counts:
            min_frames = min(provided_counts)
        else:
            # If none are provided, return empty outputs.
            return (torch.tensor([]), torch.tensor([]), tiling)
        # For each sequence, if missing or empty, create a white image sequence of length min_frames.
        for key, seq in seqs.items():
            if not seq or len(seq) == 0:
                white_img = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
                seqs[key] = [np.array(white_img).astype(np.float32)/255.0 for _ in range(min_frames)]
            else:
                # Otherwise, keep only up to min_frames.
                seqs[key] = seq[:min_frames]
        # Build a list of booleans indicating whether each cell was supplied.
        # A cell is considered provided if its sequence was non-empty originally.
        provided_flags = []
        for key in ["top_left", "top_middle", "top_right",
                    "middle_left", "middle_right",
                    "bottom_left", "bottom_middle", "bottom_right"]:
            provided_flags.append(bool(seqs[key]) and len(seqs[key]) > 0)
        combined_frames = []
        mask_frames = []
        pbar = ProgressBar(min_frames)
        # For each frame index, tile the eight cells into a 3x3 grid (with center blank).
        for i in range(min_frames):
            # Get the i-th frame from each cell.
            frames = [
                seqs["top_left"][i],
                seqs["top_middle"][i],
                seqs["top_right"][i],
                seqs["middle_left"][i],
                seqs["middle_right"][i],
                seqs["bottom_left"][i],
                seqs["bottom_middle"][i],
                seqs["bottom_right"][i],
            ]
            try:
                # Create combined grid image (each input resized to cell_size).
                grid_img = tile_images_grid8(frames, cell_size)
            except Exception as e:
                raise Exception(f"Error tiling frame {i}: " + str(e))
            # Convert to normalized numpy array.
            grid_np = np.array(grid_img).astype(np.float32) / 255.0
            combined_frames.append(grid_np)
            # Create mask grid for this frame.
            mask_img = tile_mask_grid8(provided_flags, cell_size)
            mask_np = np.array(mask_img).astype(np.float32) / 255.0
            mask_frames.append(mask_np)
            pbar.update(1)
        # Stack the sequences into tensors.
        combined_tensor = torch.from_numpy(np.stack(combined_frames))
        mask_tensor = torch.from_numpy(np.stack(mask_frames))
        return (combined_tensor, mask_tensor, tiling)
