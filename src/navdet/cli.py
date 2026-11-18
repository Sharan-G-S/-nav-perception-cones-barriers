"""Command line entry points: prepare-data, train, evaluate, detect, optimize, bev, stereo, flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from navdet.cameras import PRESETS, camera_height_for, intrinsics_for
from navdet.detect import Detector, DistanceConfig, annotate, detections_to_json, with_distances

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def add_camera_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--camera",
        choices=sorted(PRESETS),
        default="nuimages",
        help="intrinsics preset, scaled to the frame",
    )
    p.add_argument(
        "--hfov",
        type=float,
        default=None,
        help="horizontal field of view in degrees, overrides the preset",
    )
    p.add_argument(
        "--fx",
        type=float,
        default=None,
        help="exact focal length in pixels from the camera SDK, overrides everything",
    )
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument(
        "--camera-height", type=float, default=None, help="camera height above the ground in metres"
    )
    p.add_argument(
        "--pitch", type=float, default=0.0, help="camera pitch in degrees, positive looking down"
    )


def add_model_args(
    p: argparse.ArgumentParser, default_weights: str = "runs/train/yolov8n/weights/best.pt"
) -> None:
    p.add_argument("--weights", default=default_weights)
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.3)


def camera_height(args) -> float:
    return camera_height_for(args.camera, args.camera_height)


def frame_intrinsics(args, width: int, height: int):
    return intrinsics_for(width, height, args.camera, args.hfov, args.fx, args.fy, args.cx, args.cy)


def cmd_prepare_data(args) -> None:
    from navdet.data import BuildLimits, build_dataset

    limits = BuildLimits(
        nuimages=args.nuimages, lvis_per_class=args.lvis_per_class, bdd=args.bdd, seed=args.seed
    )
    summary = build_dataset(Path(args.raw), Path(args.out), limits)
    print(json.dumps(summary, indent=2))


def cmd_bdd_stats(args) -> None:
    from navdet.data import bdd_label_counts

    counts = bdd_label_counts(Path(args.samples))
    text = json.dumps(counts, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)


def cmd_train(args) -> None:
    from navdet.train import TrainConfig, train

    best = train(
        TrainConfig(
            data_yaml=Path(args.data),
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=Path(args.project),
            name=args.name,
            seed=args.seed,
            patience=args.patience,
            workers=args.workers,
        )
    )
    print(best)


def cmd_evaluate(args) -> None:
    from navdet.train import evaluate

    metrics = evaluate(
        args.weights, args.data, device=args.device, imgsz=args.imgsz, split=args.split
    )
    text = json.dumps(metrics, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)


def iter_sources(source: Path):
    if source.is_dir():
        for p in sorted(source.iterdir()):
            if p.suffix.lower() in IMAGE_SUFFIXES:
                yield p.stem, cv2.imread(str(p))
    elif source.suffix.lower() in VIDEO_SUFFIXES:
        cap = cv2.VideoCapture(str(source))
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"{source.stem}-{index:05d}", frame
            index += 1
        cap.release()
    else:
        yield source.stem, cv2.imread(str(source))


def cmd_detect(args) -> None:
    from navdet.extras.bev import BevGrid, bev_warp, draw_detections_on_bev

    detector = Detector(args.weights, device=args.device, imgsz=args.imgsz, conf=args.conf)
    config = DistanceConfig(
        method=args.method, camera_height_m=camera_height(args), pitch_deg=args.pitch
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    writer = None
    report: dict[str, list[dict]] = {}
    for name, frame in iter_sources(source):
        intrinsics = frame_intrinsics(args, frame.shape[1], frame.shape[0])
        dets = with_distances(detector.detect(frame), intrinsics, config)
        report[name] = detections_to_json(dets)
        drawn = annotate(frame, dets)
        if args.bev:
            k = np.array(
                [[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]]
            )
            grid = BevGrid(x_min=-10, x_max=10, z_min=0, z_max=40, px_per_m=frame.shape[0] / 40.0)
            bev = draw_detections_on_bev(
                bev_warp(frame, k, camera_height(args), args.pitch, grid), grid, dets
            )
            drawn = np.hstack([drawn, cv2.resize(bev, (bev.shape[1], drawn.shape[0]))])
        if source.suffix.lower() in VIDEO_SUFFIXES:
            if writer is None:
                fps = cv2.VideoCapture(str(source)).get(cv2.CAP_PROP_FPS) or 10.0
                writer = cv2.VideoWriter(
                    str(out_dir / f"{source.stem}-annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (drawn.shape[1], drawn.shape[0]),
                )
            writer.write(drawn)
        else:
            cv2.imwrite(str(out_dir / f"{name}.jpg"), drawn, [cv2.IMWRITE_JPEG_QUALITY, 85])
        for d in dets:
            print(
                f"{name}: {d.caption()} (conf {d.confidence:.2f}, bearing {d.bearing_deg:+.1f} deg)"
            )
    if writer is not None:
        writer.release()
    (out_dir / "detections.json").write_text(json.dumps(report, indent=2))


def cmd_optimize(args) -> None:
    from navdet.optimize import markdown_table, run_suite

    extra = {}
    for item in args.extra or []:
        name, path = item.split("=", 1)
        extra[name] = Path(path)
    results = run_suite(
        args.weights,
        args.data,
        imgsz=args.imgsz,
        out_dir=args.out,
        prune_amount=args.prune,
        runs=args.runs,
        finetune_epochs=args.finetune_epochs,
        extra_weights=extra,
        backbones=tuple(args.backbones or ()),
        steps=tuple(args.steps),
    )
    print(markdown_table(results))


def cmd_compare(args) -> None:
    from navdet.optimize import before_after_figure

    models = {}
    for item in args.models:
        name, path = item.split("=", 1)
        models[name] = Path(path)
    frame = cv2.imread(str(args.image))
    intrinsics = frame_intrinsics(args, frame.shape[1], frame.shape[0])
    config = DistanceConfig(
        method=args.method, camera_height_m=camera_height(args), pitch_deg=args.pitch
    )
    figure = before_after_figure(
        models, frame, intrinsics, config, imgsz=args.imgsz, device=args.device, conf=args.conf
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), figure, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(args.out)


def cmd_bev(args) -> None:
    from navdet.extras.bev import BevGrid, bev_warp, draw_detections_on_bev

    frame = cv2.imread(str(args.image))
    intrinsics = frame_intrinsics(args, frame.shape[1], frame.shape[0])
    k = np.array([[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]])
    grid = BevGrid(
        x_min=-args.half_width,
        x_max=args.half_width,
        z_min=0.0,
        z_max=args.depth,
        px_per_m=args.px_per_m,
    )
    bev = bev_warp(frame, k, camera_height(args), args.pitch, grid)
    dets = []
    if args.weights:
        detector = Detector(args.weights, device=args.device, imgsz=args.imgsz, conf=args.conf)
        dets = with_distances(
            detector.detect(frame),
            intrinsics,
            DistanceConfig(camera_height_m=camera_height(args), pitch_deg=args.pitch),
        )
    bev = draw_detections_on_bev(bev, grid, dets)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), bev)
    print(args.out)


def cmd_stereo(args) -> None:
    from navdet.extras.stereo import colorize_depth, depth_map, disparity_map

    left = cv2.imread(str(args.left), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(args.right), cv2.IMREAD_GRAYSCALE)
    if args.scale != 1.0:
        left = cv2.resize(left, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    focal = args.focal * args.scale
    disparity = disparity_map(
        left, right, num_disparities=args.num_disparities, block_size=args.block_size
    )
    offset = args.disparity_offset * args.scale
    depth = depth_map(np.where(disparity > 0, disparity + offset, disparity), focal, args.baseline)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    disp_vis = cv2.normalize(np.clip(disparity, 0, None), None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    cv2.imwrite(str(out / "disparity.png"), cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO))
    cv2.imwrite(str(out / "depth.png"), colorize_depth(depth, max_depth_m=args.max_depth))
    valid = disparity > 0
    stats = {
        "focal_px": focal,
        "baseline_m": args.baseline,
        "scale": args.scale,
        "disparity_offset_px": offset,
        "size": [int(left.shape[1]), int(left.shape[0])],
        "valid_fraction": float(valid.mean()),
        "disparity_px_median": float(np.median(disparity[valid])) if valid.any() else None,
        "depth_m_median": float(np.nanmedian(depth)) if valid.any() else None,
        "depth_m_min": float(np.nanmin(depth)) if valid.any() else None,
    }
    (out / "stereo.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def synthetic_approach(frame: np.ndarray, frames: int, zoom_end: float) -> list[np.ndarray]:
    """Simulates the robot driving forward by zooming into the frame center a little more each frame."""
    h, w = frame.shape[:2]
    out = []
    for i in range(frames):
        zoom = 1.0 + (zoom_end - 1.0) * i / max(1, frames - 1)
        cw, ch = int(w / zoom), int(h / zoom)
        x0, y0 = (w - cw) // 2, int((h - ch) * 0.6)
        out.append(
            cv2.resize(frame[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
        )
    return out


def cmd_flow(args) -> None:
    from navdet.extras.flow import FlowTracker, draw_tracks

    if args.video:
        frames = [f for _, f in iter_sources(Path(args.video))]
    else:
        frames = synthetic_approach(cv2.imread(str(args.image)), args.frames, args.zoom)
    detector = Detector(args.weights, device=args.device, imgsz=args.imgsz, conf=args.conf)
    tracker = FlowTracker(max_lost=args.max_lost, match_dist_px=args.match_dist)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out / "tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    history = []
    for i, frame in enumerate(frames):
        intrinsics = frame_intrinsics(args, w, h)
        dets = (
            with_distances(
                detector.detect(frame),
                intrinsics,
                DistanceConfig(camera_height_m=camera_height(args), pitch_deg=args.pitch),
            )
            if i % args.detect_every == 0
            else []
        )
        tracks = tracker.update(frame, dets)
        drawn = draw_tracks(annotate(frame, dets), tracks)
        writer.write(drawn)
        if i in (0, len(frames) // 2, len(frames) - 1):
            cv2.imwrite(str(out / f"frame-{i:03d}.jpg"), drawn, [cv2.IMWRITE_JPEG_QUALITY, 85])
        history.append(
            {
                "frame": i,
                "tracks": [
                    {
                        "id": t.track_id,
                        "label": t.label,
                        "x": t.position[0],
                        "y": t.position[1],
                        "lost": t.lost,
                    }
                    for t in tracks
                ],
            }
        )
    writer.release()
    (out / "tracks.json").write_text(json.dumps(history, indent=2))
    print(f"{len(frames)} frames, {tracker.next_id - 1} tracks -> {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="navdet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data", help="download the sources and write the YOLO dataset")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/yolo")
    p.add_argument("--nuimages", type=int, default=700)
    p.add_argument("--lvis-per-class", type=int, default=170)
    p.add_argument("--bdd", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("bdd-stats", help="label counts of the BDD100K validation split")
    p.add_argument("--samples", default="data/raw/bdd100k-val-samples.json")
    p.add_argument("--out", default="results/bdd100k-labels.json")
    p.set_defaults(func=cmd_bdd_stats)

    p = sub.add_parser("train", help="fine-tune a pretrained YOLO model")
    p.add_argument("--data", default="data/yolo/dataset.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="yolov8n")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="mAP of a checkpoint or exported model on a split")
    add_model_args(p)
    p.add_argument("--data", default="data/yolo/dataset.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("detect", help="detect objects and annotate them with their distance")
    add_model_args(p)
    add_camera_args(p)
    p.add_argument("--source", required=True, help="image, directory of images, or video")
    p.add_argument("--out", default="results/detections")
    p.add_argument("--method", choices=["height", "ground"], default="height")
    p.add_argument("--bev", action="store_true", help="append a bird's-eye view panel")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("optimize", help="export, quantize, prune and benchmark on CPU and GPU")
    p.add_argument("--weights", default="runs/train/yolov8n/weights/best.pt")
    p.add_argument("--data", default="data/yolo/dataset.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="results/optimization")
    p.add_argument("--prune", type=float, default=0.3)
    p.add_argument("--finetune-epochs", type=int, default=0)
    p.add_argument("--runs", type=int, default=60)
    p.add_argument("--extra", nargs="*", help="name=path of other checkpoints to put in the table")
    p.add_argument(
        "--backbones",
        nargs="*",
        help="pretrained model names to time for the backbone comparison, e.g. yolov8s.pt",
    )
    p.add_argument(
        "--steps",
        nargs="*",
        default=["cpu", "gpu", "prune", "backbones"],
        help="which parts to run; rows merge into the existing table",
    )
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser(
        "compare", help="same frame through several models, stacked, for before/after figures"
    )
    add_model_args(p)
    add_camera_args(p)
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="name=path pairs, e.g. fp32=best.pt int8=best-int8.onnx",
    )
    p.add_argument("--image", required=True)
    p.add_argument("--method", choices=["height", "ground"], default="height")
    p.add_argument("--out", default="results/before-after/compare.jpg")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("bev", help="warp an image to a bird's-eye view and plot detections on it")
    add_model_args(p)
    add_camera_args(p)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="results/bev/bev.jpg")
    p.add_argument("--half-width", type=float, default=10.0)
    p.add_argument("--depth", type=float, default=40.0)
    p.add_argument("--px-per-m", type=float, default=20.0)
    p.set_defaults(func=cmd_bev)

    p = sub.add_parser("stereo", help="disparity and depth maps for a rectified stereo pair")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--focal", type=float, required=True, help="focal length in pixels")
    p.add_argument("--baseline", type=float, required=True, help="baseline in metres")
    p.add_argument("--num-disparities", type=int, default=128)
    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="resize factor applied to both views before matching",
    )
    p.add_argument(
        "--disparity-offset",
        type=float,
        default=0.0,
        help="disparity removed by cropping (Middlebury dmin), in full-size pixels",
    )
    p.add_argument("--out", default="results/stereo")
    p.set_defaults(func=cmd_stereo)

    p = sub.add_parser("flow", help="track detections across frames with Lucas-Kanade optical flow")
    add_model_args(p)
    add_camera_args(p)
    p.add_argument("--video", default=None)
    p.add_argument("--image", default=None, help="synthesize an approach sequence from one image")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--zoom", type=float, default=1.8)
    p.add_argument("--detect-every", type=int, default=5)
    p.add_argument("--max-lost", type=int, default=8)
    p.add_argument("--match-dist", type=float, default=60.0)
    p.add_argument("--out", default="results/flow")
    p.set_defaults(func=cmd_flow)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
