"""Two-view geometry: the disparity to depth relation, the fundamental matrix and a block-matching disparity map."""

from __future__ import annotations

import cv2
import numpy as np


def depth_from_disparity(disparity_px: float, focal_px: float, baseline_m: float) -> float | None:
    """Z = f * B / d for a rectified stereo pair; None when the disparity carries no depth."""
    if disparity_px <= 0:
        return None
    return focal_px * baseline_m / disparity_px


def disparity_from_depth(depth_m: float, focal_px: float, baseline_m: float) -> float:
    return focal_px * baseline_m / depth_m


def skew(t: np.ndarray) -> np.ndarray:
    x, y, z = (float(v) for v in t)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental_matrix(
    k_left: np.ndarray, k_right: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """F = K_r^-T [t]x R K_l^-1, so that x_r^T F x_l = 0 for every correspondence."""
    essential = skew(translation) @ rotation
    f = np.linalg.inv(k_right).T @ essential @ np.linalg.inv(k_left)
    return f / np.linalg.norm(f)


def epipolar_line(fundamental: np.ndarray, point_left: np.ndarray) -> np.ndarray:
    """Line a*u + b*v + c = 0 in the right image on which the match of point_left must lie."""
    u, v = (float(x) for x in point_left)
    line = fundamental @ np.array([u, v, 1.0])
    return line / np.hypot(line[0], line[1])


def point_line_distance(line: np.ndarray, point: np.ndarray) -> float:
    u, v = (float(x) for x in point)
    return abs(line[0] * u + line[1] * v + line[2]) / np.hypot(line[0], line[1])


def disparity_map(
    left_gray: np.ndarray, right_gray: np.ndarray, num_disparities: int = 64, block_size: int = 5
) -> np.ndarray:
    """Semi-global block matching; returns disparity in pixels as float32, <= 0 where unknown."""
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0


def depth_map(disparity: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = focal_px * baseline_m / disparity[valid]
    return depth


def colorize_depth(depth: np.ndarray, max_depth_m: float = 40.0) -> np.ndarray:
    scaled = np.nan_to_num(depth, nan=max_depth_m)
    scaled = np.clip(scaled / max_depth_m, 0.0, 1.0)
    return cv2.applyColorMap((255 * (1.0 - scaled)).astype(np.uint8), cv2.COLORMAP_TURBO)
