"""Ground-plane homography and bird's-eye-view warping from the camera intrinsics, height and pitch."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from navdet.detect import COLORS, Detection


def rotation_pitch(pitch_deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ground_to_image_homography(
    k: np.ndarray, camera_height_m: float, pitch_deg: float
) -> np.ndarray:
    """Maps ground coordinates (X right, Z forward, metres) to pixels for a camera camera_height_m above the ground.

    A ground point is (X, h, Z) in a level camera frame with y down. Pitching the camera down by
    pitch_deg rotates it about x, so the pixel is K R (X, h, Z)^T = K [r1 | r3 | h r2] (X, Z, 1)^T.
    """
    r = rotation_pitch(pitch_deg)
    return k @ np.column_stack([r[:, 0], r[:, 2], camera_height_m * r[:, 1]])


def project_ground(h: np.ndarray, x_m: float, z_m: float) -> tuple[float, float]:
    p = h @ np.array([x_m, z_m, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def image_to_ground(h: np.ndarray, u: float, v: float) -> tuple[float, float]:
    p = np.linalg.inv(h) @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


@dataclass(frozen=True)
class BevGrid:
    """Metric window drawn in the bird's-eye view, forward (Z) up and right (X) to the right."""

    x_min: float = -10.0
    x_max: float = 10.0
    z_min: float = 0.0
    z_max: float = 40.0
    px_per_m: float = 20.0

    @property
    def width(self) -> int:
        return round((self.x_max - self.x_min) * self.px_per_m)

    @property
    def height(self) -> int:
        return round((self.z_max - self.z_min) * self.px_per_m)

    def x_to_col(self, x_m: float) -> float:
        return (x_m - self.x_min) * self.px_per_m

    def z_to_row(self, z_m: float) -> float:
        return (self.z_max - z_m) * self.px_per_m

    def bev_to_ground(self) -> np.ndarray:
        return np.array(
            [
                [1.0 / self.px_per_m, 0.0, self.x_min],
                [0.0, -1.0 / self.px_per_m, self.z_max],
                [0.0, 0.0, 1.0],
            ]
        )


def bev_warp(
    image: np.ndarray,
    k: np.ndarray,
    camera_height_m: float,
    pitch_deg: float,
    grid: BevGrid = BevGrid(),
) -> np.ndarray:
    h = ground_to_image_homography(k, camera_height_m, pitch_deg)
    image_from_bev = h @ grid.bev_to_ground()
    return cv2.warpPerspective(
        image,
        image_from_bev,
        (grid.width, grid.height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )


def bev_from_points(
    image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Four known ground points in the image and where they should land give the warp directly."""
    m = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    return cv2.warpPerspective(image, m, size)


def draw_detections_on_bev(
    bev: np.ndarray, grid: BevGrid, detections: list[Detection]
) -> np.ndarray:
    out = bev.copy()
    for z in range(int(grid.z_min), int(grid.z_max) + 1, 5):
        row = round(grid.z_to_row(z))
        cv2.line(out, (0, row), (out.shape[1], row), (70, 70, 70), 1)
        cv2.putText(
            out,
            f"{z}m",
            (2, max(10, row - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    for det in detections:
        if det.distance_m is None or det.lateral_m is None:
            continue
        col = round(grid.x_to_col(det.lateral_m))
        row = round(grid.z_to_row(det.distance_m))
        cv2.circle(out, (col, row), 5, COLORS.get(det.label, (255, 255, 255)), -1)
        cv2.putText(
            out,
            det.caption(),
            (col + 7, row + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out
