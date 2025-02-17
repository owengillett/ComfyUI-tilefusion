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
from .utils import floatOrInt, imageOrLatent  # imageOrLatent is used to type image sequences
from comfy.utils import ProgressBar

# Helper function to convert an input (numpy array or PIL image) to a PIL image.
def to_pil(im):
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    elif isinstance(im, np.ndarray):
        # If the image values are in [0,1], scale them to [0,255]
        if im.max() <= 1.0:
            im = (im * 255).astype(np.uint8)
        else:
            im = im.astype(np.uint8)
        return Image.fromarray(im).convert("RGB")
    else:
        raise Exception("Unsupported image format: " + str(type(im)))

# Tile a list of 9 images (assumed to be of identical dimensions) into a 3x3 grid.
def tile_images(images: List) -> Image.Image:
    if len(images) != 9:
        raise Exception("Expected 9 images to form a 3x3 grid")
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
        # Accept nine image sequences; each input is of type imageOrLatent,
        # which in our context represents a list (sequence) of images.
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
        # Collect the nine input sequences
        sequences = [seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8, seq9]
        # Ensure each sequence is a list and compute the minimum frame count
        try:
            min_frames = min(len(seq) for seq in sequences)
        except Exception as e:
            raise Exception("One or more inputs are not valid image sequences: " + str(e))
        if min_frames == 0:
            return ([],)
        combined_sequence = []
        pbar = ProgressBar(min_frames)
        # Process frame by frame (up to the shortest sequence length)
        for i in range(min_frames):
            # Get the i-th frame from each sequence
            frames = [sequences[j][i] for j in range(9)]
            try:
                grid_frame = tile_images(frames)
            except Exception as e:
                raise Exception(f"Error tiling frame {i}: " + str(e))
            # Convert grid_frame back to a numpy array in float32 format (normalized)
            grid_np = np.array(grid_frame).astype(np.float32) / 255.0
            combined_sequence.append(grid_np)
            pbar.update(1)
        return (combined_sequence,)
