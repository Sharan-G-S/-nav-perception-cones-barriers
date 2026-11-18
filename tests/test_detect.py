from pathlib import Path

import numpy as np
import pytest

from navdet.detect import (
    Detection,
    Detector,
    DistanceConfig,
    annotate,
    detections_to_json,
    with_distances,
)
from navdet.geometry import Box, CameraIntrinsics

CAM = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)


def test_caption_matches_the_assignment_format():
    det = Detection(label="stop_sign", confidence=0.9, box=Box(0, 0, 10, 10), distance_m=3.24)
    assert det.caption() == "Stop Sign, 3.2m"
    assert Detection(label="cone", confidence=0.9, box=Box(0, 0, 10, 10)).caption() == "Cone"


def test_with_distances_uses_the_height_prior():
    det = Detection(label="cone", confidence=0.8, box=Box(600.0, 400.0, 680.0, 500.0))
    out = with_distances([det], CAM, DistanceConfig(method="height"))
    assert out[0].distance_m == pytest.approx(1000.0 * 0.70 / 100.0)
    assert out[0].lateral_m == pytest.approx((640.0 - 640.0) * 7.0 / 1000.0)
    assert out[0].bearing_deg == pytest.approx(0.0)


def test_with_distances_ground_method_uses_the_box_bottom():
    det = Detection(label="barrier", confidence=0.8, box=Box(600.0, 400.0, 700.0, 560.0))
    out = with_distances([det], CAM, DistanceConfig(method="ground", camera_height_m=1.5))
    assert out[0].distance_m == pytest.approx(1000.0 * 1.5 / 200.0)


def test_with_distances_leaves_unknown_classes_without_distance():
    det = Detection(label="bus", confidence=0.8, box=Box(0.0, 0.0, 100.0, 100.0))
    assert with_distances([det], CAM, DistanceConfig())[0].distance_m is None


def test_annotate_draws_on_a_copy():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    det = Detection(
        label="cone", confidence=0.8, box=Box(100.0, 100.0, 300.0, 400.0), distance_m=2.5
    )
    out = annotate(image, [det])
    assert out.shape == image.shape
    assert image.sum() == 0
    assert out.sum() > 0


def test_detections_to_json_is_plain_data():
    det = Detection(
        label="cone",
        confidence=0.8,
        box=Box(1.0, 2.0, 3.0, 4.0),
        distance_m=2.5,
        lateral_m=0.1,
        bearing_deg=1.0,
    )
    assert detections_to_json([det]) == [
        {
            "label": "cone",
            "caption": "Cone, 2.5m",
            "confidence": 0.8,
            "box": [1.0, 2.0, 3.0, 4.0],
            "distance_m": 2.5,
            "lateral_m": 0.1,
            "bearing_deg": 1.0,
        }
    ]


def test_pretrained_detector_finds_the_bus(tmp_path):
    from ultralytics.utils import ASSETS
    from ultralytics.utils.downloads import attempt_download_asset

    weights = Path(attempt_download_asset("yolov8n.pt"))
    image_path = ASSETS / "bus.jpg"
    import cv2

    image = cv2.imread(str(image_path))
    detector = Detector(weights, device="cpu", imgsz=640, conf=0.4)
    dets = detector.detect(image)
    labels = {d.label for d in dets}
    assert "bus" in labels
    for d in dets:
        assert 0 <= d.box.x1 < d.box.x2 <= image.shape[1]
        assert 0 <= d.box.y1 < d.box.y2 <= image.shape[0]
        assert 0.4 <= d.confidence <= 1.0
