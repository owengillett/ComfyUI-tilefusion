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

# Minimal inlined definitions to replace external utils dependency
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

# Helper function to convert an image (numpy array or PIL image) to a PIL image
def to_pil(im):
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    elif isinstance(im, np.ndarray):
        # If pixel values are in [0,1], scale to [0,255]
        if im.max() <= 1.0:
            im = (im * 255).astype(np.uint8)
        else:
            im = im.astype(np.uint8)
        return Image.fromarray(im).convert("RGB")
    else:
        raise Exception("Unsupported image format: " + str(type(im)))

# Helper function to tile 9 images into a 3x3 grid.
def tile_images(images: List) -> Image.Image:
    if len(images) != 9:
        raise Exception("Expected 9 images to form a 3x3 grid, got " + str(len(images)))
    pil_images = [to_pil(im) for im in images]
    w, h = pil_images[0].size
    grid = Image.new("RGB", (3 * w, 3 * h))
    for idx, img in enumerate(pil_images):
        col = idx % 3
        row = idx // 3
        grid.paste(img, (col * w, row * h))
    return grid

class VideoGridCombine:
    @classmethod
    def INPUT_TYPES(cls):
        # Expect nine image sequences (lists of images) as input.
        return {
            "required": {
                "seq1": (imageOrLatent,),
                "seq2": (imageOrLatent,),
                "seq3": (imageOrLatent,),
                "seq4": (imageOrLatent,),
                "seq5": (imageOrLatent,),
                "seq6": (imageOrLatent,),
                "seq7": (imageOrLatent,),
                "seq8": (imageOrLatent,),
                "seq9": (imageOrLatent,),
            }
        }

    RETURN_TYPES = ("IMAGE_SEQUENCE",)
    RETURN_NAMES = ("combined_sequence",)
    CATEGORY = "custom"
    FUNCTION = "combine_grid"

    def combine_grid(self, seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8, seq9):
        # Gather the nine input sequences
        sequences = [seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8, seq9]
        try:
            min_frames = min(len(seq) for seq in sequences)
        except Exception as e:
            raise Exception("One or more inputs are not valid image sequences: " + str(e))
        if min_frames == 0:
            print("Warning: At least one input sequence is empty. Returning empty sequence.", file=sys.stderr)
            return ([],)
        combined_sequence = []
        pbar = ProgressBar(min_frames)
        # Process each frame index (up to the shortest sequence length)
        for i in range(min_frames):
            try:
                # Get the i-th frame from each sequence
                frames = [sequences[j][i] for j in range(9)]
            except Exception as e:
                raise Exception(f"Error retrieving frame {i} from input sequences: " + str(e))
            try:
                grid_frame = tile_images(frames)
            except Exception as e:
                raise Exception(f"Error tiling frame {i}: " + str(e))
            # Convert the composite grid image to a numpy array in float32 format normalized to [0,1]
            grid_np = np.array(grid_frame).astype(np.float32) / 255.0
            combined_sequence.append(grid_np)
            pbar.update(1)
        return (combined_sequence,)
