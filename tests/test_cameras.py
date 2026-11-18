import pytest

from navdet.cameras import PRESETS, camera_height_for, intrinsics_for


def test_robot_camera_presets_match_their_datasheet_fields_of_view():
    assert PRESETS["zed2i"].intrinsics.hfov_deg == pytest.approx(110.0)
    assert PRESETS["d455"].intrinsics.hfov_deg == pytest.approx(90.0)
    assert PRESETS["d435i"].intrinsics.hfov_deg == pytest.approx(69.0)
    assert PRESETS["zed2i"].baseline_m == 0.120
    assert PRESETS["d455"].baseline_m == 0.095
    assert PRESETS["d435i"].baseline_m == 0.050


def test_preset_scales_to_the_frame_size():
    cam = intrinsics_for(640, 360, camera="d435i")
    assert cam.hfov_deg == pytest.approx(69.0)
    assert (cam.width, cam.height) == (640, 360)
    assert cam.fx == pytest.approx(PRESETS["d435i"].intrinsics.fx / 2.0)


def test_exact_intrinsics_override_everything():
    cam = intrinsics_for(
        1280, 720, camera="zed2i", hfov_deg=50.0, fx=700.0, fy=701.0, cx=630.0, cy=350.0
    )
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (700.0, 701.0, 630.0, 350.0)
    only_fx = intrinsics_for(1280, 720, fx=900.0)
    assert (only_fx.fy, only_fx.cx, only_fx.cy) == (900.0, 640.0, 360.0)


def test_field_of_view_overrides_the_preset():
    assert intrinsics_for(1280, 720, camera="zed2i", hfov_deg=90.0).fx == pytest.approx(640.0)


def test_camera_height_defaults_to_the_preset():
    assert camera_height_for("nuimages") == 1.5
    assert camera_height_for("d455") == 0.6
    assert camera_height_for("d455", 0.45) == 0.45
