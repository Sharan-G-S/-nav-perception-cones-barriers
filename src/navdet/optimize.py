"""Edge optimisation: ONNX export, INT8 quantisation, FP16, pruning, and a CPU/GPU FPS benchmark with mAP before and after."""

from __future__ import annotations

import gzip
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn.utils import prune

from navdet.detect import Detector
from navdet.train import evaluate

STEPS = ("cpu", "gpu", "prune", "backbones")


@dataclass(frozen=True)
class BenchmarkResult:
    variant: str
    path: str
    device: str
    precision: str
    size_mb: float
    fps: float
    latency_ms: float
    map50: float | None
    map50_95: float | None
    sparsity: float | None = None
    gzip_mb: float | None = None


def file_size_mb(path: Path | str) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def gzip_size_mb(path: Path | str) -> float:
    with open(path, "rb") as src:
        return len(gzip.compress(src.read(), compresslevel=6)) / (1024 * 1024)


def export_onnx(
    weights: Path | str,
    imgsz: int,
    out_dir: Path | None = None,
    int8: bool = False,
    data_yaml: Path | str | None = None,
) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs = {"format": "onnx", "imgsz": imgsz, "simplify": True, "dynamic": False, "device": "cpu"}
    if int8:
        kwargs.update(quantize=8, data=str(data_yaml))
    exported = Path(model.export(**kwargs))
    if out_dir is None:
        return exported
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (Path(weights).stem + ("-int8" if int8 else "-fp32") + ".onnx")
    shutil.move(str(exported), target)
    return target


def conv_layers(module: torch.nn.Module) -> dict[str, torch.nn.Conv2d]:
    """Every learned convolution; the DFL box decoder is a fixed Conv2d that must stay intact."""
    from ultralytics.nn.modules.block import DFL

    fixed = {f"{name}.conv" for name, m in module.named_modules() if isinstance(m, DFL)}
    return {
        name: m
        for name, m in module.named_modules()
        if isinstance(m, torch.nn.Conv2d) and name not in fixed
    }


def sparsity(weights: Path | str) -> float:
    from ultralytics import YOLO

    convs = conv_layers(YOLO(str(weights)).model).values()
    zeros = sum(int((m.weight == 0).sum()) for m in convs)
    total = sum(m.weight.numel() for m in convs)
    return zeros / total


def prune_model(weights: Path | str, amount: float, out_path: Path | str) -> Path:
    """L1 unstructured pruning of every Conv2d weight; the zeros are baked into the saved checkpoint."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    for conv in conv_layers(model.model).values():
        prune.l1_unstructured(conv, name="weight", amount=amount)
        prune.remove(conv, "weight")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    return out_path


def sparsity_masks(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: (m.weight != 0).float() for name, m in conv_layers(module).items()}


def apply_masks(module: torch.nn.Module, masks: dict[str, torch.Tensor]) -> None:
    for name, m in conv_layers(module).items():
        mask = masks.get(name)
        if mask is not None:
            m.weight.data.mul_(mask.to(m.weight.device, m.weight.dtype))


def sparse_finetune(
    pruned: Path, data_yaml: Path, epochs: int, imgsz: int, device: str, out_dir: Path
) -> Path:
    """Fine-tunes a pruned checkpoint while keeping its zeros in place, on the model and its EMA copy."""
    from ultralytics import YOLO

    from navdet.train import TrainConfig, train

    masks = sparsity_masks(YOLO(str(pruned)).model)

    def keep_sparse(trainer):
        apply_masks(trainer.model, masks)
        if trainer.ema is not None:
            apply_masks(trainer.ema.ema, masks)

    config = TrainConfig(
        data_yaml=Path(data_yaml),
        model=str(pruned),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=out_dir / "finetune",
        name="pruned",
        batch=16,
        patience=epochs,
    )
    return train(config, callbacks={"on_train_batch_end": keep_sparse, "on_train_end": keep_sparse})


def benchmark_fps(
    weights: Path | str,
    device: str,
    imgsz: int,
    frames: list[np.ndarray],
    warmup: int = 10,
    runs: int = 60,
    half: bool = False,
) -> tuple[float, float, set[str]]:
    """End-to-end frames per second (preprocess, inference, postprocess) over real frames."""
    detector = Detector(weights, device=device, imgsz=imgsz, conf=0.25, half=half)
    labels: set[str] = set()
    for i in range(warmup):
        detector.detect(frames[i % len(frames)])
    start = time.perf_counter()
    for i in range(runs):
        labels.update(d.label for d in detector.detect(frames[i % len(frames)]))
    elapsed = time.perf_counter() - start
    return runs / elapsed, 1000.0 * elapsed / runs, labels


def load_frames(image_dir: Path | str, limit: int = 16) -> list[np.ndarray]:
    paths = sorted(Path(image_dir).glob("*.jpg"))[:limit]
    return [cv2.imread(str(p)) for p in paths]


def gpu_device() -> str | None:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def measure(
    variant: str,
    path: Path,
    device: str,
    precision: str,
    imgsz: int,
    frames,
    data_yaml,
    half=False,
    eval_device=None,
    runs=60,
) -> BenchmarkResult:
    fps, latency_ms, _ = benchmark_fps(
        path, device=device, imgsz=imgsz, frames=frames, runs=runs, half=half
    )
    metrics = (
        evaluate(path, data_yaml, device=eval_device or device, imgsz=imgsz) if data_yaml else {}
    )
    return BenchmarkResult(
        variant=variant,
        path=str(path),
        device=device,
        precision=precision,
        size_mb=round(file_size_mb(path), 2),
        fps=round(fps, 1),
        latency_ms=round(latency_ms, 1),
        map50=metrics.get("map50"),
        map50_95=metrics.get("map50_95"),
        gzip_mb=round(gzip_size_mb(path), 2),
    )


def run_suite(
    weights: Path | str,
    data_yaml: Path | str,
    imgsz: int,
    out_dir: Path | str,
    prune_amount: float = 0.3,
    runs: int = 60,
    finetune_epochs: int = 0,
    extra_weights: dict[str, Path] | None = None,
    backbones: tuple[str, ...] = (),
    steps: tuple[str, ...] = STEPS,
) -> list[BenchmarkResult]:
    """Before/after table for a trained model; each step's rows merge into the table already on disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frames(Path(data_yaml).parent / "images" / "val")
    weights = Path(weights)
    gpu = gpu_device()
    results: list[BenchmarkResult] = []

    if "cpu" in steps:
        results.append(
            measure("pytorch fp32", weights, "cpu", "fp32", imgsz, frames, data_yaml, runs=runs)
        )
        results.extend(onnx_rows(weights, imgsz, out_dir, frames, data_yaml, runs))
    if "gpu" in steps and gpu:
        results.append(
            measure("pytorch fp32", weights, gpu, "fp32", imgsz, frames, data_yaml, runs=runs)
        )
        results.append(
            measure(
                "pytorch fp16", weights, gpu, "fp16", imgsz, frames, data_yaml, half=True, runs=runs
            )
        )
    if "prune" in steps:
        results.extend(
            prune_rows(
                weights, data_yaml, imgsz, out_dir, frames, prune_amount, runs, finetune_epochs, gpu
            )
        )
    for name, path in (extra_weights or {}).items():
        results.append(
            measure(name, Path(path), "cpu", "fp32", imgsz, frames, data_yaml, runs=runs)
        )
        if gpu:
            results.append(
                measure(name, Path(path), gpu, "fp32", imgsz, frames, data_yaml, runs=runs)
            )
    if "backbones" in steps:
        results.extend(backbone_rows(backbones, imgsz, frames, runs, gpu))

    merged = merge_results(load_results(out_dir / "optimization.json"), results)
    (out_dir / "optimization.json").write_text(json.dumps([asdict(r) for r in merged], indent=2))
    (out_dir / "optimization.md").write_text(markdown_table(merged) + "\n")
    return merged


def load_results(path: Path) -> list[BenchmarkResult]:
    if not path.exists():
        return []
    return [BenchmarkResult(**row) for row in json.loads(path.read_text())]


def merge_results(
    existing: list[BenchmarkResult], new: list[BenchmarkResult]
) -> list[BenchmarkResult]:
    """New rows replace existing rows with the same variant and device, in place; the rest are appended."""
    merged = list(existing)
    for row in new:
        for i, old in enumerate(merged):
            if (old.variant, old.device) == (row.variant, row.device):
                merged[i] = row
                break
        else:
            merged.append(row)
    return merged


def onnx_rows(weights, imgsz, out_dir, frames, data_yaml, runs) -> list[BenchmarkResult]:
    onnx_fp32 = export_onnx(weights, imgsz, out_dir)
    onnx_int8 = export_onnx(weights, imgsz, out_dir, int8=True, data_yaml=data_yaml)
    return [
        measure("onnx fp32", onnx_fp32, "cpu", "fp32", imgsz, frames, data_yaml, runs=runs),
        measure(
            "onnx int8 (static, calibrated)",
            onnx_int8,
            "cpu",
            "int8",
            imgsz,
            frames,
            data_yaml,
            runs=runs,
        ),
    ]


def prune_rows(
    weights, data_yaml, imgsz, out_dir, frames, prune_amount, runs, finetune_epochs, gpu
) -> list[BenchmarkResult]:
    pct = int(prune_amount * 100)
    pruned = prune_model(weights, prune_amount, out_dir / f"{weights.stem}-pruned{pct}.pt")
    row = measure(
        f"pruned {pct}% (L1, no finetune)",
        pruned,
        "cpu",
        "fp32",
        imgsz,
        frames,
        data_yaml,
        runs=runs,
    )
    results = [BenchmarkResult(**{**asdict(row), "sparsity": round(sparsity(pruned), 3)})]
    if finetune_epochs > 0:
        recovered = sparse_finetune(
            pruned, Path(data_yaml), finetune_epochs, imgsz, gpu or "cpu", out_dir
        )
        label = f"pruned {pct}% + {finetune_epochs} ep sparse finetune"
        row = measure(label, recovered, "cpu", "fp32", imgsz, frames, data_yaml, runs=runs)
        results.append(
            BenchmarkResult(**{**asdict(row), "sparsity": round(sparsity(recovered), 3)})
        )
    return results


def backbone_rows(backbones, imgsz, frames, runs, gpu) -> list[BenchmarkResult]:
    from ultralytics.utils.downloads import attempt_download_asset

    results = []
    for name in backbones:
        path = Path(attempt_download_asset(name))
        label = f"backbone {Path(name).stem} (COCO weights, speed only)"
        results.append(measure(label, path, "cpu", "fp32", imgsz, frames, None, runs=runs))
        if gpu:
            results.append(measure(label, path, gpu, "fp32", imgsz, frames, None, runs=runs))
    return results


def markdown_table(results: list[BenchmarkResult]) -> str:
    def fmt(v, digits):
        return "n/a" if v is None else f"{v:.{digits}f}"

    lines = [
        "| Variant | Device | Precision | Size (MB) | mAP50 | mAP50-95 | Latency (ms) | FPS |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.variant} | {r.device} | {r.precision} | {r.size_mb:.1f} | {fmt(r.map50, 3)} | {fmt(r.map50_95, 3)} | {r.latency_ms:.1f} | {r.fps:.1f} |"
        )
    return "\n".join(lines)


def before_after_figure(
    models: dict[str, Path | str],
    frame: np.ndarray,
    intrinsics,
    config,
    imgsz: int = 640,
    device: str = "cpu",
    conf: float = 0.3,
    tile_width: int = 960,
) -> np.ndarray:
    """One row per model: the same frame annotated with distances, headed by the model name and its measured latency."""
    from navdet.detect import annotate, with_distances

    tiles = []
    for name, path in models.items():
        detector = Detector(path, device=device, imgsz=imgsz, conf=conf)
        detector.detect(frame)
        start = time.perf_counter()
        dets = detector.detect(frame)
        latency_ms = 1000.0 * (time.perf_counter() - start)
        dets = with_distances(dets, intrinsics, config)
        drawn = annotate(frame, dets)
        scale = tile_width / drawn.shape[1]
        drawn = cv2.resize(drawn, (tile_width, round(drawn.shape[0] * scale)))
        header = np.full((36, tile_width, 3), 30, dtype=np.uint8)
        text = f"{name}  |  {file_size_mb(path):.1f} MB  |  {len(dets)} detections  |  {latency_ms:.0f} ms this frame"
        cv2.putText(
            header, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
        )
        tiles.append(np.vstack([header, drawn]))
    return np.vstack(tiles)
