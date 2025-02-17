import os
import copy
import json
import re
import datetime
from typing import Optional, List
import itertools

import PIL
from PIL import Image, ExifTags, PngImagePlugin
import torch
import numpy as np

# Assume these modules are available in your ComfyUI environment.
from comfy.utils import ProgressBar
import folder_paths

# Helper function to tile nine images (assumed to be the same dimensions) into a 3x3 grid.
def tile_images(image_list):
    # Convert any numpy arrays to PIL images if needed.
    imgs = [Image.fromarray(img) if not isinstance(img, Image.Image) else img for img in image_list]
    # Assume all images have the same size.
    w, h = imgs[0].size
    # Create a new blank image with width=3*w and height=3*h.
    grid_img = Image.new('RGB', (3 * w, 3 * h))
    # Paste each image into its corresponding position.
    for idx, im in enumerate(imgs):
        row = idx // 3
        col = idx % 3
        grid_img.paste(im, (col * w, row * h))
    return grid_img

class VideoGridCombine:
    @classmethod
    def INPUT_TYPES(cls):
        # Each required input is expected to be a sequence (list) of images.
        # In ComfyUI VHS the "images" input type is often defined as "imageOrLatent".
        # Here we explicitly use "IMAGE_SEQUENCE" to indicate a list of images.
        return {
            "required": {
                "seq1": ("IMAGE_SEQUENCE",),
                "seq2": ("IMAGE_SEQUENCE",),
                "seq3": ("IMAGE_SEQUENCE",),
                "seq4": ("IMAGE_SEQUENCE",),
                "seq5": ("IMAGE_SEQUENCE",),
                "seq6": ("IMAGE_SEQUENCE",),
                "seq7": ("IMAGE_SEQUENCE",),
                "seq8": ("IMAGE_SEQUENCE",),
                "seq9": ("IMAGE_SEQUENCE",),
            },
            "optional": {
                # You can add extra parameters if needed (for example, an option to choose PNG or JPEG).
            }
        }

    RETURN_TYPES = ("IMAGE_SEQUENCE",)
    RETURN_NAMES = ("combined_sequence",)
    CATEGORY = "custom"  # You can change this to fit your node categorization.
    FUNCTION = "combine_sequence"

    def combine_sequence(
        self,
        seq1,
        seq2,
        seq3,
        seq4,
        seq5,
        seq6,
        seq7,
        seq8,
        seq9,
        **kwargs
    ):
        # Collect the nine input sequences.
        sequences = [seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8, seq9]
        # Determine the minimum number of frames among the inputs.
        min_len = min(len(seq) for seq in sequences)
        if min_len == 0:
            return ([],)
        combined_sequence = []
        pbar = ProgressBar(min_len)
        # For each frame index, pick one image from each sequence and tile them.
        for i in range(min_len):
            # Gather the i-th frame from each sequence.
            frames = [sequences[j][i] for j in range(9)]
            grid_frame = tile_images(frames)
            # Convert the combined PIL image back to a numpy array (the typical format in ComfyUI).
            combined_sequence.append(np.array(grid_frame))
            pbar.update(1)
        return (combined_sequence,)
