import cv2
import numpy as np
import pytest

from navdet.detect import Detection
from navdet.extras.bev import (
    BevGrid,
    bev_from_points,
    bev_warp,
    ground_to_image_homography,
    image_to_ground,
    project_ground,
)
from navdet.extras.flow import FlowTracker, flow_points
from navdet.extras.stereo import (
    depth_from_disparity,
    depth_map,
    disparity_from_depth,
    disparity_map,
    epipolar_line,
    fundamental_matrix,
    point_line_distance,
)
from navdet.geometry import Box

K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])


def textured(height, width, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (3, 3), 0)


def test_disparity_depth_relation_is_inverse():
    assert depth_from_disparity(disparity_px=8.0, focal_px=700.0, baseline_m=0.12) == pytest.approx(
        10.5
    )
    assert disparity_from_depth(depth_m=10.5, focal_px=700.0, baseline_m=0.12) == pytest.approx(8.0)
    assert depth_from_disparity(disparity_px=0.0, focal_px=700.0, baseline_m=0.12) is None


def test_fundamental_matrix_of_a_horizontal_stereo_rig_gives_horizontal_epipolar_lines():
    F = fundamental_matrix(K, K, np.eye(3), np.array([-0.12, 0.0, 0.0]))
    left = np.array([700.0, 400.0])
    line = epipolar_line(F, left)
    for disparity in (3.0, 20.0, 57.0):
        right = np.array([700.0 - disparity, 400.0])
        assert point_line_distance(line, right) == pytest.approx(0.0, abs=1e-6)
    assert point_line_distance(line, np.array([650.0, 430.0])) == pytest.approx(30.0, abs=1e-6)


def test_block_matching_recovers_a_known_shift():
    left = textured(240, 400)
    shift = 8
    right = np.zeros_like(left)
    right[:, :-shift] = left[:, shift:]
    disp = disparity_map(left, right, num_disparities=32, block_size=5)
    valid = disp[20:-20, 60:-40]
    assert np.median(valid) == pytest.approx(shift, abs=1.0)
    depth = depth_map(disp, focal_px=700.0, baseline_m=0.12)
    assert np.nanmedian(depth[20:-20, 60:-40]) == pytest.approx(700.0 * 0.12 / shift, rel=0.15)


def test_ground_homography_matches_the_pinhole_model_for_a_level_camera():
    H = ground_to_image_homography(K, camera_height_m=1.5, pitch_deg=0.0)
    u, v = project_ground(H, 0.0, 10.0)
    assert (u, v) == pytest.approx((640.0, 510.0))
    u, v = project_ground(H, 2.0, 8.0)
    assert (u, v) == pytest.approx((890.0, 547.5))
    assert image_to_ground(H, 890.0, 547.5) == pytest.approx((2.0, 8.0))


def test_pitched_camera_puts_the_horizon_above_the_image_center():
    H = ground_to_image_homography(K, camera_height_m=1.5, pitch_deg=10.0)
    _, v = project_ground(H, 0.0, 1e6)
    assert v < 360.0
    assert v == pytest.approx(360.0 - 1000.0 * np.tan(np.radians(10.0)), abs=0.1)


def test_bev_warp_moves_a_ground_mark_to_its_metric_cell():
    H = ground_to_image_homography(K, camera_height_m=1.5, pitch_deg=0.0)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    u, v = project_ground(H, 2.0, 8.0)
    cv2.circle(image, (round(u), round(v)), 4, (255, 255, 255), -1)
    grid = BevGrid(x_min=-5.0, x_max=5.0, z_min=0.0, z_max=20.0, px_per_m=10.0)
    bev = bev_warp(image, K, camera_height_m=1.5, pitch_deg=0.0, grid=grid)
    assert bev.shape == (200, 100, 3)
    ys, xs = np.nonzero(bev[:, :, 0] > 128)
    assert xs.mean() == pytest.approx(grid.x_to_col(2.0), abs=4.0)
    assert ys.mean() == pytest.approx(grid.z_to_row(8.0), abs=4.0)


def test_bev_from_four_points_is_a_plain_perspective_warp():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[40:60, 40:60] = 255
    src = np.float32([[40, 40], [60, 40], [60, 60], [40, 60]])
    dst = np.float32([[0, 0], [50, 0], [50, 50], [0, 50]])
    out = bev_from_points(image, src, dst, (50, 50))
    assert out.shape == (50, 50, 3)
    assert (out[:, :, 0] > 128).mean() > 0.95


def test_lucas_kanade_follows_a_global_shift():
    base = textured(300, 300, seed=1)
    frame0 = base[0:200, 0:200]
    frame1 = base[2:202, 5:205]
    points = np.float32([[50.0, 50.0], [100.0, 80.0], [150.0, 120.0]])
    moved, ok = flow_points(frame0, frame1, points)
    assert ok.all()
    assert moved == pytest.approx(points + np.float32([-5.0, -2.0]), abs=1.0)


def test_flow_tracker_keeps_a_cone_track_through_a_missed_detection():
    base = textured(300, 300, seed=2)
    frames = [base[k * 2 : k * 2 + 200, k * 5 : k * 5 + 200] for k in range(3)]
    truth = [(100.0 - 5.0 * k, 80.0 - 2.0 * k) for k in range(3)]

    def detection_at(x, y):
        return Detection(label="cone", confidence=0.9, box=Box(x - 10, y - 15, x + 10, y + 15))

    tracker = FlowTracker(max_lost=3, match_dist_px=20.0)
    tracker.update(frames[0], [detection_at(*truth[0])])
    tracker.update(frames[1], [])
    tracks = tracker.update(frames[2], [detection_at(*truth[2])])

    assert len(tracks) == 1
    track = tracks[0]
    assert track.label == "cone"
    assert len(track.points) == 3
    for got, want in zip(track.points, truth):
        assert got == pytest.approx(want, abs=1.5)
