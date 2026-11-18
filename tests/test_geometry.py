import math

import pytest

from navdet.geometry import (
    Box,
    CameraIntrinsics,
    bearing_deg,
    distance_from_ground,
    distance_from_height,
    lateral_offset,
)


def test_intrinsics_from_hfov_gives_focal_length_in_pixels():
    cam = CameraIntrinsics.from_hfov(1280, 720, 90.0)
    assert cam.fx == pytest.approx(640.0)
    assert cam.fy == pytest.approx(640.0)
    assert (cam.cx, cam.cy) == (640.0, 360.0)


def test_scaled_intrinsics_follow_the_resize():
    cam = CameraIntrinsics(fx=1266.0, fy=1266.0, cx=816.0, cy=491.0, width=1600, height=900)
    small = cam.scaled(640, 360)
    assert small.fx == pytest.approx(506.4)
    assert small.fy == pytest.approx(506.4)
    assert small.cx == pytest.approx(326.4)
    assert small.cy == pytest.approx(196.4)
    assert (small.width, small.height) == (640, 360)


def test_distance_from_height_is_pinhole_similar_triangles():
    assert distance_from_height(box_height_px=100.0, real_height_m=0.7, fy=1000.0) == pytest.approx(
        7.0
    )
    assert distance_from_height(box_height_px=350.0, real_height_m=0.7, fy=1000.0) == pytest.approx(
        2.0
    )


def test_distance_from_height_rejects_empty_box():
    with pytest.raises(ValueError):
        distance_from_height(box_height_px=0.0, real_height_m=0.7, fy=1000.0)


def test_distance_from_ground_with_level_camera():
    cam = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
    z = distance_from_ground(y_bottom_px=560.0, camera_height_m=1.5, intrinsics=cam)
    assert z == pytest.approx(1000.0 * 1.5 / 200.0)


def test_distance_from_ground_with_pitched_camera():
    cam = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
    z = distance_from_ground(y_bottom_px=360.0, camera_height_m=1.5, intrinsics=cam, pitch_deg=10.0)
    assert z == pytest.approx(1.5 / math.tan(math.radians(10.0)))


def test_distance_from_ground_is_none_above_horizon():
    cam = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
    assert distance_from_ground(y_bottom_px=300.0, camera_height_m=1.5, intrinsics=cam) is None


def test_lateral_offset_and_bearing():
    cam = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
    assert lateral_offset(u_px=740.0, z_m=5.0, intrinsics=cam) == pytest.approx(0.5)
    assert lateral_offset(u_px=540.0, z_m=5.0, intrinsics=cam) == pytest.approx(-0.5)
    assert bearing_deg(u_px=1640.0, intrinsics=cam) == pytest.approx(45.0)


def test_box_helpers():
    box = Box(10.0, 20.0, 50.0, 100.0)
    assert box.width == 40.0
    assert box.height == 80.0
    assert box.bottom_center == (30.0, 100.0)
    assert box.area == 3200.0
