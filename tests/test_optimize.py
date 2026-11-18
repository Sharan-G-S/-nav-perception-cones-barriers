from pathlib import Path

import cv2
import numpy as np
import pytest
from ultralytics.utils import ASSETS
from ultralytics.utils.downloads import attempt_download_asset

from navdet.optimize import (
    BenchmarkResult,
    benchmark_fps,
    export_onnx,
    file_size_mb,
    gzip_size_mb,
    markdown_table,
    merge_results,
    prune_model,
    sparsity,
)


@pytest.fixture(scope="module")
def weights():
    return Path(attempt_download_asset("yolov8n.pt"))


@pytest.fixture(scope="module")
def frames():
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    return [image, cv2.flip(image, 1)]


def test_onnx_export_runs_and_detects(weights, frames, tmp_path):
    onnx_path = export_onnx(weights, imgsz=320, out_dir=tmp_path)
    assert onnx_path.suffix == ".onnx" and onnx_path.exists()
    fps, latency_ms, labels = benchmark_fps(
        onnx_path, device="cpu", imgsz=320, frames=frames, warmup=1, runs=3
    )
    assert fps > 0 and latency_ms > 0
    assert "bus" in labels


def test_pruning_zeroes_weights_and_still_loads(weights, frames, tmp_path):
    pruned = prune_model(weights, amount=0.3, out_path=tmp_path / "pruned.pt")
    assert pruned.exists()
    before = sparsity(weights)
    after = sparsity(pruned)
    assert before < 0.05
    assert 0.25 < after < 0.35
    assert gzip_size_mb(pruned) < gzip_size_mb(weights)
    fps, _, labels = benchmark_fps(pruned, device="cpu", imgsz=320, frames=frames, warmup=1, runs=2)
    assert fps > 0
    assert "bus" in labels


def test_merge_replaces_rows_with_the_same_variant_and_device():
    old = [
        BenchmarkResult("a", "a.pt", "cpu", "fp32", 1.0, 1.0, 1.0, 0.1, 0.1),
        BenchmarkResult("b", "b.pt", "cpu", "fp32", 1.0, 1.0, 1.0, 0.1, 0.1),
    ]
    new = [
        BenchmarkResult("a", "a.pt", "cpu", "fp32", 2.0, 2.0, 2.0, 0.2, 0.2),
        BenchmarkResult("c", "c.pt", "mps", "fp32", 3.0, 3.0, 3.0, 0.3, 0.3),
    ]
    merged = merge_results(old, new)
    assert [(r.variant, r.size_mb) for r in merged] == [("a", 2.0), ("b", 1.0), ("c", 3.0)]


def test_markdown_table_has_one_row_per_result():
    rows = [
        BenchmarkResult("fp32 pt", "a.pt", "cpu", "fp32", 6.2, 12.5, 80.0, 0.5, 0.3),
        BenchmarkResult("int8 onnx", "a.onnx", "cpu", "int8", 3.1, 25.0, 40.0, 0.48, 0.28),
    ]
    text = markdown_table(rows)
    assert text.count("\n") == 3
    assert "| int8 onnx | cpu | int8 | 3.1 | 0.480 | 0.280 | 40.0 | 25.0 |" in text


def test_file_size_in_megabytes(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(np.zeros(2 * 1024 * 1024, dtype=np.uint8).tobytes())
    assert file_size_mb(p) == pytest.approx(2.0)
