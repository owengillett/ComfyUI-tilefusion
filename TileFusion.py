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

# Helper: Tile 8 images into a 3x3 grid with the center blank.
def tile_images_grid8(images: List, cell_size: int) -> Image.Image:
    if len(images) != 8:
        raise Exception("Expected 8 images, got " + str(len(images)))
    pil_images = [to_pil(im).resize((cell_size, cell_size), Image.ANTIALIAS) for im in images]
    blank = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
    grid = Image.new("RGB", (3 * cell_size, 3 * cell_size))
    # Top row.
    grid.paste(pil_images[0], (0, 0))
    grid.paste(pil_images[1], (cell_size, 0))
    grid.paste(pil_images[2], (2 * cell_size, 0))
    # Middle row.
    grid.paste(pil_images[3], (0, cell_size))
    grid.paste(blank, (cell_size, cell_size))
    grid.paste(pil_images[4], (2 * cell_size, cell_size))
    # Bottom row.
    grid.paste(pil_images[5], (0, 2 * cell_size))
    grid.paste(pil_images[6], (cell_size, 2 * cell_size))
    grid.paste(pil_images[7], (2 * cell_size, 2 * cell_size))
    return grid

# Helper: Create a mask grid for 8 cells.
# For each cell, if the corresponding input is provided, mask = black (0); if not, white (255).
def tile_mask_grid8(provided: List[bool], cell_size: int) -> Image.Image:
    black_cell = Image.new("L", (cell_size, cell_size), 0)
    white_cell = Image.new("L", (cell_size, cell_size), 255)
    cells = [black_cell if flag else white_cell for flag in provided]
    center = white_cell
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

# Helper: Safely get the length of a sequence (list or torch.Tensor)
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

    # We now return two outputs (combined sequence and mask) plus tiling.
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
        # Dictionary of optional sequences.
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
        # Determine the minimum frame count among provided sequences.
        provided_counts = [seq_length(seq) for seq in seqs.values() if seq_length(seq) > 0]
        if provided_counts:
            min_frames = min(provided_counts)
        else:
            # If no inputs are provided, return empty outputs.
            return (torch.tensor([]), torch.tensor([]), tiling)
        # For each sequence, if missing/empty, substitute with a white image sequence of length min_frames.
        for key, seq in seqs.items():
            if seq_length(seq) == 0:
                white = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
                seqs[key] = [np.array(white).astype(np.float32)/255.0 for _ in range(min_frames)]
            else:
                # If sequence is provided as a list, truncate to min_frames.
                if not isinstance(seqs[key], torch.Tensor):
                    seqs[key] = seq[:min_frames]
                # If it's a tensor, assume the first dimension is frames.
                else:
                    seqs[key] = seq[:min_frames]
        # Create provided flags for mask generation.
        # For each key, a flag is True if originally provided non-empty.
        provided_flags = []
        for key in ["top_left", "top_middle", "top_right",
                    "middle_left", "middle_right",
                    "bottom_left", "bottom_middle", "bottom_right"]:
            provided_flags.append(seq_length(seqs[key]) > 0)
        combined_frames = []
        mask_frames = []
        pbar = ProgressBar(min_frames)
        for i in range(min_frames):
            try:
                # Retrieve the i-th frame for each cell.
                frames = [seqs[key][i] for key in ["top_left", "top_middle", "top_right",
                                                   "middle_left", "middle_right",
                                                   "bottom_left", "bottom_middle", "bottom_right"]]
            except Exception as e:
                raise Exception(f"Error retrieving frame {i}: " + str(e))
            try:
                grid_img = tile_images_grid8(frames, cell_size)
            except Exception as e:
                raise Exception(f"Error tiling frame {i}: " + str(e))
            grid_np = np.array(grid_img).astype(np.float32) / 255.0
            combined_frames.append(grid_np)
            # Create mask for this f
