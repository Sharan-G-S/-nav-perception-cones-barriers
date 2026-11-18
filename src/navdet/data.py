"""Builds the YOLO training set from nuImages (cones, barriers), LVIS (stop signs, cones) and BDD100K (driving backgrounds)."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import random
import time
import urllib.request
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from navdet.classes import CLASS_NAMES, class_index

NUIMAGES_URL = "https://huggingface.co/datasets/mviola/nuImages-cam-front/resolve/main/data/validation-00000-of-00001.parquet"
LVIS_URL = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
BDD_SAMPLES_URL = "https://huggingface.co/datasets/dgural/bdd100k/resolve/main/samples.json"
BDD_IMAGE_URL = "https://huggingface.co/datasets/dgural/bdd100k/resolve/main/{filepath}"

NUIMAGES_CLASSES = {8: "barrier", 11: "cone"}
LVIS_CLASSES = {"stop_sign": "stop_sign", "cone": "cone"}

RAW_FILES = {
    "nuimages": ("nuimages-front-validation.parquet", NUIMAGES_URL),
    "lvis": ("lvis_v1_val.json.zip", LVIS_URL),
    "bdd": ("bdd100k-val-samples.json", BDD_SAMPLES_URL),
}


@dataclass(frozen=True)
class LabeledBox:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Record:
    image_id: str
    source: str
    width: int
    height: int
    boxes: tuple[LabeledBox, ...]
    image_bytes: bytes | None = None
    image_url: str | None = None


@dataclass(frozen=True)
class BuildLimits:
    nuimages: int = 600
    lvis_per_class: int = 170
    bdd: int = 100
    val_fraction: float = 0.15
    min_box_px: int = 8
    seed: int = 0


def yolo_label_lines(boxes, width: int, height: int) -> list[str]:
    lines = []
    for b in boxes:
        x1, x2 = sorted((min(max(b.x1, 0.0), width), min(max(b.x2, 0.0), width)))
        y1, y2 = sorted((min(max(b.y1, 0.0), height), min(max(b.y2, 0.0), height)))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"{class_index(b.name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def split_for(image_id: str, val_fraction: float) -> str:
    digest = hashlib.sha1(image_id.encode()).digest()
    share = int.from_bytes(digest[:4], "big") / 2**32
    return "val" if share < val_fraction else "train"


def stable_sample(records: list[Record], limit: int, seed: int) -> list[Record]:
    ordered = sorted(records, key=lambda r: r.image_id)
    random.Random(seed).shuffle(ordered)
    return ordered[:limit]


def coco_box(name: str, xywh, min_box_px: int) -> LabeledBox | None:
    x, y, w, h = (float(v) for v in xywh)
    if h < min_box_px or w < 2:
        return None
    return LabeledBox(name, x, y, x + w, y + h)


def image_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        return im.size


def nuimages_batches(parquet_path: Path):
    reader = pq.ParquetFile(parquet_path)
    for batch in reader.iter_batches(batch_size=32, columns=["image", "objects"]):
        yield from batch.to_pylist()


def nuimages_index(parquet_path: Path, min_box_px: int) -> list[Record]:
    records = []
    for row in nuimages_batches(parquet_path):
        boxes = []
        for xywh, cat in zip(row["objects"]["bbox"], row["objects"]["categories"]):
            name = NUIMAGES_CLASSES.get(int(cat))
            if name is None:
                continue
            box = coco_box(name, xywh, min_box_px)
            if box is not None:
                boxes.append(box)
        if not boxes:
            continue
        width, height = image_size(row["image"]["bytes"])
        image_id = Path(row["image"]["path"]).stem
        records.append(Record(image_id, "nuimages", width, height, tuple(boxes)))
    return records


def nuimages_images(parquet_path: Path, wanted: set[str]) -> Iterator[tuple[str, bytes]]:
    for row in nuimages_batches(parquet_path):
        image_id = Path(row["image"]["path"]).stem
        if image_id in wanted:
            yield image_id, row["image"]["bytes"]


def lvis_index(zip_path: Path, min_box_px: int) -> list[Record]:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".json"))
        doc = json.loads(z.read(name))
    cat_names = {c["id"]: LVIS_CLASSES.get(c["name"]) for c in doc["categories"]}
    boxes_by_image: dict[int, list[LabeledBox]] = {}
    for ann in doc["annotations"]:
        name = cat_names.get(ann["category_id"])
        if name is None:
            continue
        box = coco_box(name, ann["bbox"], min_box_px)
        if box is not None:
            boxes_by_image.setdefault(ann["image_id"], []).append(box)
    records = []
    for img in doc["images"]:
        boxes = boxes_by_image.get(img["id"])
        if not boxes:
            continue
        records.append(
            Record(
                image_id=f"lvis-{img['id']}",
                source="lvis",
                width=img["width"],
                height=img["height"],
                boxes=tuple(boxes),
                image_url=img["coco_url"],
            )
        )
    return records


def bdd_index(samples_json: Path) -> list[Record]:
    doc = json.loads(Path(samples_json).read_text())
    records = []
    for sample in doc["samples"]:
        filepath = sample["filepath"]
        meta = sample["metadata"]
        records.append(
            Record(
                image_id=f"bdd-{Path(filepath).stem}",
                source="bdd100k",
                width=int(meta["width"]),
                height=int(meta["height"]),
                boxes=(),
                image_url=BDD_IMAGE_URL.format(filepath=filepath),
            )
        )
    return records


def bdd_label_counts(samples_json: Path) -> dict:
    """Images and boxes per label in the BDD100K split, which shows it has no cone, barrier or stop sign labels."""
    doc = json.loads(Path(samples_json).read_text())
    boxes: dict[str, int] = {}
    for sample in doc["samples"]:
        for det in (sample.get("detections") or {}).get("detections", []):
            boxes[det["label"]] = boxes.get(det["label"], 0) + 1
    return {
        "images": len(doc["samples"]),
        "boxes": dict(sorted(boxes.items(), key=lambda kv: -kv[1])),
    }


def fetch_bytes(url: str, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except (OSError, http.client.HTTPException) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def download(url: str, dest: Path) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def write_record(record: Record, root: Path, split: str) -> Path:
    image_path = root / "images" / split / f"{record.image_id}.jpg"
    label_path = root / "labels" / split / f"{record.image_id}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        data = (
            record.image_bytes if record.image_bytes is not None else fetch_bytes(record.image_url)
        )
        image_path.write_bytes(data)
    lines = yolo_label_lines(record.boxes, record.width, record.height)
    label_path.write_text("".join(f"{line}\n" for line in lines))
    return image_path


def write_dataset_yaml(root: Path) -> Path:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    path = root / "dataset.yaml"
    path.write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    return path


def select_lvis(records: list[Record], per_class: int, seed: int) -> list[Record]:
    chosen: dict[str, Record] = {}
    for name in LVIS_CLASSES.values():
        with_class = [r for r in records if any(b.name == name for b in r.boxes)]
        for r in stable_sample(with_class, per_class, seed):
            chosen[r.image_id] = r
    return list(chosen.values())


def summarize(records: list[Record], splits: dict[str, str]) -> dict:
    summary: dict = {"images": {}, "boxes": {}}
    for r in records:
        split = splits[r.image_id]
        summary["images"].setdefault(split, {}).setdefault(r.source, 0)
        summary["images"][split][r.source] += 1
        for b in r.boxes:
            summary["boxes"].setdefault(split, {}).setdefault(b.name, 0)
            summary["boxes"][split][b.name] += 1
    return summary


def build_dataset(raw_dir: Path, out_dir: Path, limits: BuildLimits = BuildLimits()) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    paths = {key: download(url, raw_dir / name) for key, (name, url) in RAW_FILES.items()}

    nu_all = nuimages_index(paths["nuimages"], limits.min_box_px)
    nu = stable_sample(nu_all, limits.nuimages, limits.seed)
    lvis = select_lvis(
        lvis_index(paths["lvis"], limits.min_box_px), limits.lvis_per_class, limits.seed
    )
    bdd = stable_sample(bdd_index(paths["bdd"]), limits.bdd, limits.seed)

    records = nu + lvis + bdd
    splits = {r.image_id: split_for(r.image_id, limits.val_fraction) for r in records}

    nu_bytes = dict(nuimages_images(paths["nuimages"], {r.image_id for r in nu}))
    nu_records = [
        Record(r.image_id, r.source, r.width, r.height, r.boxes, image_bytes=nu_bytes[r.image_id])
        for r in nu
    ]
    for r in nu_records:
        write_record(r, out_dir, splits[r.image_id])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda r: write_record(r, out_dir, splits[r.image_id]), lvis + bdd))

    write_dataset_yaml(out_dir)
    summary = summarize(records, splits)
    summary["available"] = {"nuimages": len(nu_all), "limits": limits.__dict__}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
