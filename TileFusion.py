import os
import subprocess
import copy
from typing import Optional

# In this example we use folder_paths to determine output directories.
# If you already have folder_paths in your ComfyUI environment, it will be used;
# otherwise, we provide a fallback implementation.
try:
    import folder_paths
except ImportError:
    class folder_paths:
        @staticmethod
        def get_output_directory():
            return os.getcwd()
        @staticmethod
        def get_temp_directory():
            return os.getcwd()

class VideoGridCombine:
    """
    Custom node that accepts nine video inputs (each as a VHS_VIDEOINFO dictionary)
    and arranges them in a 3x3 grid using ffmpeg’s filter_complex.

    IMPORTANT:
      - Each VHS_VIDEOINFO is expected to be a dict that includes either a "loaded_path"
        or "source_path" key with the video file's path.
      - All input videos should have the same resolution and framerate.
      - ffmpeg must be installed and available in the system PATH.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video1": ("VHS_VIDEOINFO",),
                "video2": ("VHS_VIDEOINFO",),
                "video3": ("VHS_VIDEOINFO",),
                "video4": ("VHS_VIDEOINFO",),
                "video5": ("VHS_VIDEOINFO",),
                "video6": ("VHS_VIDEOINFO",),
                "video7": ("VHS_VIDEOINFO",),
                "video8": ("VHS_VIDEOINFO",),
                "video9": ("VHS_VIDEOINFO",),
            }
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("combined_video",)
    CATEGORY = "custom_video"
    FUNCTION = "combine_grid"

    def combine_grid(self, video1, video2, video3, video4, video5, video6, video7, video8, video9):
        # Collect the nine video info dictionaries into a list.
        videos = [video1, video2, video3, video4, video5, video6, video7, video8, video9]
        input_paths = []
        for idx, vid in enumerate(videos):
            # Try to get a valid path; prefer "loaded_path" over "source_path".
            path = vid.get("loaded_path") or vid.get("source_path")
            if path is None:
                raise Exception(f"Input video {idx+1} does not contain a valid file path.")
            input_paths.append(path)

        # Determine output directory and construct an output filename.
        output_dir = folder_paths.get_output_directory()
        base_name = "grid_video"
        output_path = os.path.join(output_dir, f"{base_name}.mp4")

        # Build the ffmpeg command:
        #  - Add each input video.
        #  - Reset timestamps for each stream using setpts=PTS-STARTPTS.
        #  - Horizontally stack every three videos into a row.
        #  - Vertically stack the three rows into a 3x3 grid.
        cmd = ["ffmpeg", "-y"]
        for path in input_paths:
            cmd.extend(["-i", path])

        filter_complex = (
            "[0:v] setpts=PTS-STARTPTS [v0]; "
            "[1:v] setpts=PTS-STARTPTS [v1]; "
            "[2:v] setpts=PTS-STARTPTS [v2]; "
            "[3:v] setpts=PTS-STARTPTS [v3]; "
            "[4:v] setpts=PTS-STARTPTS [v4]; "
            "[5:v] setpts=PTS-STARTPTS [v5]; "
            "[6:v] setpts=PTS-STARTPTS [v6]; "
            "[7:v] setpts=PTS-STARTPTS [v7]; "
            "[8:v] setpts=PTS-STARTPTS [v8]; "
            "[v0][v1][v2] hstack=inputs=3 [row0]; "
            "[v3][v4][v5] hstack=inputs=3 [row1]; "
            "[v6][v7][v8] hstack=inputs=3 [row2]; "
            "[row0][row1][row2] vstack=inputs=3 [grid]"
        )

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[grid]",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "veryfast",
            output_path
        ])

        # Run the ffmpeg command.
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception("ffmpeg failed:\n" + e.stderr.decode() if e.stderr else str(e))

        # Return output in the expected format (a tuple with a VHS_FILENAMES-like structure).
        return (("grid_video", [output_path]),)
