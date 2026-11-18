"""Fine-tunes a pretrained YOLO detector on the cone, barrier and stop sign set, and evaluates checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    data_yaml: Path
    model: str = "yolov8n.pt"
    epochs: int = 30
    imgsz: int = 640
    batch: int = 16
    device: str = "cpu"
    project: Path = Path("runs/train")
    name: str = "yolov8n"
    seed: int = 0
    patience: int = 15
    workers: int = 4
    fraction: float = 1.0


def train(config: TrainConfig, callbacks: dict | None = None) -> Path:
    from ultralytics import YOLO

    model = YOLO(config.model)
    for event, fn in (callbacks or {}).items():
        model.add_callback(event, fn)
    data = Path(config.data_yaml)
    model.train(
        data=str(data.resolve()) if data.exists() else str(data),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        project=str(Path(config.project).resolve()),
        name=config.name,
        seed=config.seed,
        patience=config.patience,
        workers=config.workers,
        fraction=config.fraction,
        exist_ok=True,
        pretrained=True,
        plots=True,
        verbose=False,
    )
    run_dir = Path(model.trainer.save_dir)
    (run_dir / "train_config.json").write_text(
        json.dumps(
            {**asdict(config), "data_yaml": str(config.data_yaml), "project": str(config.project)},
            indent=2,
        )
    )
    return run_dir / "weights" / "best.pt"


def evaluate(
    weights: Path | str,
    data_yaml: Path | str,
    device: str = "cpu",
    imgsz: int = 640,
    split: str = "val",
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        split=split,
        plots=False,
        verbose=False,
        batch=8,
    )
    names = metrics.names
    per_class = (
        {
            names[i]: float(ap)
            for i, ap in zip(
                metrics.box.ap_class_index.tolist(),
                metrics.box.maps[metrics.box.ap_class_index].tolist(),
            )
        }
        if len(metrics.box.ap_class_index)
        else {}
    )
    per_class_ap50 = (
        {
            names[i]: float(ap)
            for i, ap in zip(metrics.box.ap_class_index.tolist(), metrics.box.ap50.tolist())
        }
        if len(metrics.box.ap_class_index)
        else {}
    )
    return {
        "weights": str(weights),
        "split": split,
        "imgsz": imgsz,
        "device": device,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_map50": per_class_ap50,
        "per_class_map50_95": per_class,
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
    }
