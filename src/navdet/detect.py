"""Runs a detector on an image and attaches a camera-geometry distance to every detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from navdet.classes import OBJECT_HEIGHTS_M, display_name
from navdet.geometry import (
    Box,
    CameraIntrinsics,
    bearing_deg,
    distance_from_ground,
    distance_from_height,
    lateral_offset,
)

COLORS = {"cone": (0, 140, 255), "barrier": (0, 200, 0), "stop_sign": (0, 0, 230)}


@dataclass
class Detection:
    label: str
    confidence: float
    box: Box
    distance_m: float | None = None
    lateral_m: float | None = None
    bearing_deg: float | None = None

    def caption(self) -> str:
        name = display_name(self.label)
        if self.distance_m is None:
            return name
        return f"{name}, {self.distance_m:.1f}m"


@dataclass(frozen=True)
class DistanceConfig:
    """How to turn a box into metres. 'height' uses the known object height, 'ground' the box bottom."""

    method: str = "height"
    heights_m: dict[str, float] = field(default_factory=lambda: dict(OBJECT_HEIGHTS_M))
    camera_height_m: float = 1.5
    pitch_deg: float = 0.0


class Detector:
    """Thin wrapper over an Ultralytics model (.pt, .onnx or any exported format)."""

    def __init__(
        self,
        weights: Path | str,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
        half: bool = False,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.half = half

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            quantize=16 if self.half else None,
            verbose=False,
        )[0]
        boxes = result.boxes
        return [
            Detection(
                label=result.names[int(cls)],
                confidence=float(conf),
                box=Box(*(float(v) for v in xyxy)),
            )
            for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
        ]


def estimate_distance(
    det: Detection, intrinsics: CameraIntrinsics, config: DistanceConfig
) -> float | None:
    if config.method == "ground":
        return distance_from_ground(
            det.box.y2, config.camera_height_m, intrinsics, config.pitch_deg
        )
    height = config.heights_m.get(det.label)
    if height is None or det.box.height <= 0:
        return None
    return distance_from_height(det.box.height, height, intrinsics.fy)


def with_distances(
    detections: list[Detection],
    intrinsics: CameraIntrinsics,
    config: DistanceConfig = DistanceConfig(),
) -> list[Detection]:
    out = []
    for det in detections:
        z = estimate_distance(det, intrinsics, config)
        u = det.box.center[0]
        out.append(
            Detection(
                label=det.label,
                confidence=det.confidence,
                box=det.box,
                distance_m=z,
                lateral_m=None if z is None else lateral_offset(u, z, intrinsics),
                bearing_deg=bearing_deg(u, intrinsics),
            )
        )
    return out


def annotate(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image.copy()
    scale = max(0.5, min(out.shape[1], out.shape[0]) / 900.0)
    thickness = max(1, round(2 * scale))
    for det in detections:
        color = COLORS.get(det.label, (120, 120, 120))
        x1, y1, x2, y2 = (round(v) for v in (det.box.x1, det.box.y1, det.box.x2, det.box.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = det.caption()
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, thickness)
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + th + baseline + 4), color, -1)
        cv2.putText(
            out,
            text,
            (x1 + 3, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out


def detections_to_json(detections: list[Detection]) -> list[dict]:
    return [
        {
            "label": d.label,
            "caption": d.caption(),
            "confidence": d.confidence,
            "box": [d.box.x1, d.box.y1, d.box.x2, d.box.y2],
            "distance_m": d.distance_m,
            "lateral_m": d.lateral_m,
            "bearing_deg": d.bearing_deg,
        }
        for d in detections
    ]
