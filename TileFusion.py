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
def tile_images_grid8(images: List) -> Image.Image:
    # images: [top_left, top_middle, top_right, middle_left, middle_right, bottom_left, bottom_middle, bottom_right]
    if len(images) != 8:
        raise Exception("Expected 8 images, got " + str(len(images)))
    pil_images = [to_pil(im) for im in images]
    # Use the size of the first image as reference
    w, h = pil_images[0].size
    # Create a blank (white) image for the center cell.
    blank = Image.new("RGB", (w, h), (255, 255, 255))
    # Create a new image for the 3x3 grid.
    grid = Image.new("RGB", (3 * w, 3 * h))
    # Paste the 8 images into their positions:
    # Top row: top_left, top_middle, top_right
    grid.paste(pil_images[0], (0, 0))
    grid.paste(pil_images[1], (w, 0))
    grid.paste(pil_images[2], (2 * w, 0))
    # Middle row: middle_left, blank, middle_right
    grid.paste(pil_images[3], (0, h))
    grid.paste(blank, (w, h))
    grid.paste(pil_images[4], (2 * w, h))
    # Bottom row: bottom_left, bottom_middle, bottom_right
    grid.paste(pil_images[5], (0, 2 * h))
    grid.paste(pil_images[6], (w, 2 * h))
    grid.paste(pil_images[7], (2 * w, 2 * h))
    return grid

class VideoGridCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "top_left": (imageOrLatent,),
                "top_middle": (imageOrLatent,),
                "top_right": (imageOrLatent,),
                "middle_left": (imageOrLatent,),
                "middle_right": (imageOrLatent,),
                "bottom_left": (imageOrLatent,),
                "bottom_middle": (imageOrLatent,),
                "bottom_right": (imageOrLatent,),
                "tiling": (["enable", "x_only", "y_only", "disable"],),
            }
        }

    # Return types: an image sequence and the tiling setting.
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("combined_sequence", "tiling")
    CATEGORY = "custom"
    FUNCTION = "combine_grid"

    def combine_grid(
        self,
        top_left,
        top_middle,
        top_right,
        middle_left,
        middle_right,
        bottom_left,
        bottom_middle,
        bottom_right,
        tiling,
    ):
        # Gather the eight image sequences.
        sequences = [
            top_left, top_middle, top_right,
            middle_left, middle_right,
            bottom_left, bottom_middle, bottom_right
        ]
        try:
            min_frames = min(len(seq) for seq in sequences)
        except Exception as e:
            raise Exception("One or more inputs are not valid image sequences: " + str(e))
        if min_frames == 0:
            return (torch.tensor([]), tiling)
        combined_frames = []
        pbar = ProgressBar(min_frames)
        # Process frame by frame.
        for i in range(min_frames):
            try:
                # Retrieve the i-th frame from each sequence.
                frames = [sequences[j][i] for j in range(8)]
            except Exception as e:
                raise Exception(f"Error retrieving frame {i}: " + str(e))
            try:
                grid_frame = tile_images_grid8(frames)
            except Exception as e:
                raise Exception(f"Error tiling frame {i}: " + str(e))
            # Convert the grid image to a normalized numpy array.
            grid_np = np.array(grid_frame).astype(np.float32) / 255.0
            combined_frames.append(grid_np)
            pbar.update(1)
        # Stack frames into a single numpy array and then convert to torch.Tensor.
        combined_tensor = torch.from_numpy(np.stack(combined_frames))
        return (combined_tensor, tiling)
