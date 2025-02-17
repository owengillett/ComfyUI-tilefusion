from .TileFusion import (RepeatVideo, VideoGrid)

NODE_CLASS_MAPPINGS = {
    "RepeatVideo": RepeatVideo,
    "VideoGrid": VideoGrid
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RepeatVideo": "Repeat Video",
    "VideoGrid": "Video Grid"
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
