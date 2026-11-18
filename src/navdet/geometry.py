"""Pinhole camera geometry used to turn a 2D box into a distance from the robot."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraIntrinsics:
    """Focal lengths and principal point in pixels for an image of width x height."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> CameraIntrinsics:
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    def scaled(self, width: int, height: int) -> CameraIntrinsics:
        sx = width / self.width
        sy = height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
        )

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates, corners (x1, y1) top-left and (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def distance_from_height(box_height_px: float, real_height_m: float, fy: float) -> float:
    """Depth along the optical axis from similar triangles: Z = fy * H / h."""
    if box_height_px <= 0:
        raise ValueError("box height must be positive")
    return fy * real_height_m / box_height_px


def distance_from_ground(
    y_bottom_px: float,
    camera_height_m: float,
    intrinsics: CameraIntrinsics,
    pitch_deg: float = 0.0,
) -> float | None:
    """Ground distance to the point where the box touches the ground plane.

    The ray through the bottom pixel is tilted atan((y - cy) / fy) below the optical
    axis. Adding the camera pitch (positive when the camera looks down) gives the ray's
    depression angle, and the ray meets the ground at camera_height / tan(angle).
    Returns None when the ray never reaches the ground (bottom at or above the horizon).
    """
    depression = math.atan((y_bottom_px - intrinsics.cy) / intrinsics.fy) + math.radians(pitch_deg)
    if depression <= 0:
        return None
    return camera_height_m / math.tan(depression)


def lateral_offset(u_px: float, z_m: float, intrinsics: CameraIntrinsics) -> float:
    """Sideways offset in metres at depth z, positive to the right of the optical axis."""
    return (u_px - intrinsics.cx) * z_m / intrinsics.fx


def bearing_deg(u_px: float, intrinsics: CameraIntrinsics) -> float:
    """Horizontal angle in degrees from the optical axis, positive to the right."""
    return math.degrees(math.atan((u_px - intrinsics.cx) / intrinsics.fx))
