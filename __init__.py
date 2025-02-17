from .TileFusion import (VideoGridCombine)

NODE_CLASS_MAPPINGS = {
    # "RepeatVideo": RepeatVideo,
    "VideoGrid": VideoGridCombine
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # "RepeatVideo": "Repeat Video",
    "VideoGridCombine": "Video Grid Combine"
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
