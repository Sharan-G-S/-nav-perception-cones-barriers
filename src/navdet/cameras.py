"""Camera presets. Intrinsics scale with the frame, so a preset applies to any resize of its reference frame.

The robot cameras (ZED 2i, RealSense D455, RealSense D435i) are entered from their datasheet RGB
fields of view; the exact calibrated values should be read from the camera SDK at runtime and
passed with --fx --fy --cx --cy, which override any preset.
"""

from dataclasses import dataclass

from navdet.geometry import CameraIntrinsics


@dataclass(frozen=True)
class CameraPreset:
    intrinsics: CameraIntrinsics
    height_m: float
    baseline_m: float | None
    note: str


PRESETS = {
    "nuimages": CameraPreset(
        CameraIntrinsics(fx=1266.417, fy=1266.417, cx=816.267, cy=491.507, width=1600, height=900),
        height_m=1.5,
        baseline_m=None,
        note="nuScenes/nuImages CAM_FRONT published calibration; used for the training images",
    ),
    "bdd100k": CameraPreset(
        CameraIntrinsics.from_hfov(1280, 720, 65.0),
        height_m=1.4,
        baseline_m=None,
        note="dashcam without published calibration, 65 degree horizontal field of view assumed",
    ),
    "zed2i": CameraPreset(
        CameraIntrinsics.from_hfov(1280, 720, 110.0),
        height_m=0.6,
        baseline_m=0.120,
        note="Stereolabs ZED 2i, 2.1 mm lens: 110 x 70 degree field of view, 120 mm baseline, HD720 frame",
    ),
    "d455": CameraPreset(
        CameraIntrinsics.from_hfov(1280, 720, 90.0),
        height_m=0.6,
        baseline_m=0.095,
        note="Intel RealSense D455 RGB: 90 x 65 degree field of view, 95 mm stereo baseline, 1280 x 720 frame",
    ),
    "d435i": CameraPreset(
        CameraIntrinsics.from_hfov(1280, 720, 69.0),
        height_m=0.6,
        baseline_m=0.050,
        note="Intel RealSense D435i RGB: 69 x 42 degree field of view, 50 mm stereo baseline, 1280 x 720 frame",
    ),
}


def intrinsics_for(
    width: int,
    height: int,
    camera: str | None = None,
    hfov_deg: float | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> CameraIntrinsics:
    """Exact values win over a field of view, which wins over the preset scaled to the frame."""
    if fx is not None:
        return CameraIntrinsics(
            fx=fx,
            fy=fy if fy is not None else fx,
            cx=cx if cx is not None else width / 2.0,
            cy=cy if cy is not None else height / 2.0,
            width=width,
            height=height,
        )
    if hfov_deg is not None:
        return CameraIntrinsics.from_hfov(width, height, hfov_deg)
    return PRESETS[camera or "nuimages"].intrinsics.scaled(width, height)


def camera_height_for(camera: str | None, override: float | None = None) -> float:
    if override is not None:
        return override
    return PRESETS[camera or "nuimages"].height_m
