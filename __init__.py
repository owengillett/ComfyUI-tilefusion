from .TileFusion import (VideoGridCombine)

NODE_CLASS_MAPPINGS = {
    "VideoGridCombine": VideoGridCombine
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoGridCombine": "Video Grid Combine"
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
