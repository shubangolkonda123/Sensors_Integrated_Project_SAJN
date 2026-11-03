"""
camera.py - Intel RealSense RGB-D camera wrapper.

Gives the rest of the code exactly two things:
    Camera.Intrinsics   -> fx, fy, cx, cy, distortion
    Camera.get_frame()  -> (Colour_BGR, Depth_Metres) or None if no new frame yet

The depth image is ALIGNED to the colour image, so pixel (u, v) in the colour image
and pixel (u, v) in the depth image look at the same point. That is what lets us
take a blob found in colour and read its depth directly.

If the lab uses a different camera, write a new class with the same two members
and nothing else in the project needs to change.
"""
import os
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Camera_Intrinsics:
    Fx: float
    Fy: float
    Cx: float
    Cy: float
    Width: int
    Height: int
    Dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def K(self):
        """3x3 camera matrix used by OpenCV."""
        return np.array([[self.Fx, 0, self.Cx], [0, self.Fy, self.Cy], [0, 0, 1.0]])


def load_custom_intrinsics(Path, Width, Height):
    """Load intrinsics saved by calibrate_intrinsics.py."""
    Data = np.load(Path)
    K = Data["K"]
    return Camera_Intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], Width, Height, Data["Dist"].ravel())


class RealSense_Camera:
    def __init__(self, Width=640, Height=480, Fps=30, Custom_Intrinsics_File=None):
        import pyrealsense2 as rs   # imported here so mock mode works without the library
        self.rs = rs

        self.Pipeline = rs.pipeline()
        Cfg = rs.config()
        Cfg.enable_stream(rs.stream.color, Width, Height, rs.format.bgr8, Fps)
        Cfg.enable_stream(rs.stream.depth, Width, Height, rs.format.z16, Fps)
        Profile = self.Pipeline.start(Cfg)

        # Raw depth is in integer units; multiply by Depth_Scale to get metres
        self.Depth_Scale = Profile.get_device().first_depth_sensor().get_depth_scale()

        # Re-project depth into the colour camera's viewpoint
        self.Align = rs.align(rs.stream.color)

        # Noise handling (Part B): RealSense's own post-processing filters
        #   spatial  = edge-preserving smoothing within one frame
        #   temporal = averaging across frames (good: our target is static)
        self.Spatial = rs.spatial_filter()
        self.Temporal = rs.temporal_filter()

        # Factory intrinsics of the colour stream (very accurate on RealSense)
        Intr = Profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.Intrinsics = Camera_Intrinsics(Intr.fx, Intr.fy, Intr.ppx, Intr.ppy,
                                            Intr.width, Intr.height, np.array(Intr.coeffs))
        if Custom_Intrinsics_File and os.path.exists(Custom_Intrinsics_File):
            self.Intrinsics = load_custom_intrinsics(Custom_Intrinsics_File, Width, Height)
            print(f"[camera] using custom intrinsics from {Custom_Intrinsics_File}")

        # Let auto-exposure settle before we trust anything
        for _ in range(30):
            self.Pipeline.wait_for_frames()
        print(f"[camera] RealSense started, fx={self.Intrinsics.Fx:.1f}, depth scale={self.Depth_Scale}")

    def get_frame(self):
        """Non-blocking. Returns (Colour_BGR uint8 HxWx3, Depth float32 metres HxW) or None."""
        Frames = self.Pipeline.poll_for_frames()
        if not Frames:
            return None
        Aligned = self.Align.process(Frames)
        Colour_Frame = Aligned.get_color_frame()
        Depth_Frame = Aligned.get_depth_frame()
        if not Colour_Frame or not Depth_Frame:
            return None
        Depth_Frame = self.Spatial.process(Depth_Frame)
        Depth_Frame = self.Temporal.process(Depth_Frame)
        Colour = np.asanyarray(Colour_Frame.get_data()).copy()
        Depth = np.asanyarray(Depth_Frame.get_data()).astype(np.float32) * self.Depth_Scale
        return Colour, Depth          # depth of 0 means "no reading"

    def close(self):
        self.Pipeline.stop()
