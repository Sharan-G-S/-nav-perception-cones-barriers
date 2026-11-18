import json

from navdet.train import TrainConfig, evaluate, train


def test_train_writes_the_run_where_the_config_says(tmp_path):
    best = train(
        TrainConfig(
            data_yaml="coco8.yaml",
            model="yolov8n.pt",
            epochs=1,
            imgsz=160,
            batch=4,
            device="cpu",
            project=tmp_path / "runs",
            name="smoke",
            workers=0,
        )
    )
    assert best == tmp_path / "runs" / "smoke" / "weights" / "best.pt"
    assert best.exists()
    config = json.loads((tmp_path / "runs" / "smoke" / "train_config.json").read_text())
    assert config["epochs"] == 1

    metrics = evaluate(best, "coco8.yaml", device="cpu", imgsz=160)
    assert set(metrics) >= {
        "map50",
        "map50_95",
        "precision",
        "recall",
        "per_class_map50_95",
        "speed_ms",
    }
    assert 0.0 <= metrics["map50"] <= 1.0
