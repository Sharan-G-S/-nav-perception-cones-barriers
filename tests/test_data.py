import io
import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from navdet.data import (
    LabeledBox,
    Record,
    bdd_index,
    bdd_label_counts,
    lvis_index,
    nuimages_images,
    nuimages_index,
    split_for,
    write_dataset_yaml,
    write_record,
    yolo_label_lines,
)


def jpeg_bytes(width, height, color=(120, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_yolo_label_lines_are_normalized_center_width_height():
    boxes = [LabeledBox("cone", 100, 200, 300, 400), LabeledBox("stop_sign", 0, 0, 500, 250)]
    assert yolo_label_lines(boxes, width=1000, height=500) == [
        "0 0.200000 0.600000 0.200000 0.400000",
        "2 0.250000 0.250000 0.500000 0.500000",
    ]


def test_yolo_label_lines_clip_to_the_image():
    boxes = [LabeledBox("barrier", -50, 100, 150, 600)]
    assert yolo_label_lines(boxes, width=1000, height=500) == [
        "1 0.075000 0.600000 0.150000 0.800000"
    ]


def test_split_is_deterministic_and_roughly_balanced():
    ids = [f"img-{i}" for i in range(2000)]
    first = [split_for(i, val_fraction=0.15) for i in ids]
    assert first == [split_for(i, val_fraction=0.15) for i in ids]
    val_share = first.count("val") / len(first)
    assert 0.11 < val_share < 0.19


def test_nuimages_index_keeps_only_images_with_cones_or_barriers(tmp_path):
    cone_img = jpeg_bytes(160, 90)
    car_img = jpeg_bytes(160, 90, (0, 0, 90))
    table = pa.table(
        {
            "image": [
                {"bytes": cone_img, "path": "a.jpg"},
                {"bytes": car_img, "path": "b.jpg"},
            ],
            "objects": [
                {
                    "bbox": [
                        [10.0, 20.0, 30.0, 40.0],
                        [50.0, 50.0, 20.0, 20.0],
                        [0.0, 0.0, 5.0, 3.0],
                    ],
                    "categories": [11, 16, 8],
                },
                {"bbox": [[10.0, 20.0, 30.0, 40.0]], "categories": [16]},
            ],
        }
    )
    path = tmp_path / "nu.parquet"
    pq.write_table(table, path)

    records = nuimages_index(path, min_box_px=8)
    assert len(records) == 1
    rec = records[0]
    assert rec.image_id == "a"
    assert (rec.width, rec.height) == (160, 90)
    assert rec.boxes == (LabeledBox("cone", 10.0, 20.0, 40.0, 60.0),)

    images = dict(nuimages_images(path, {"a"}))
    assert images == {"a": cone_img}


def test_lvis_index_maps_stop_signs_and_cones(tmp_path):
    doc = {
        "categories": [
            {"id": 1019, "name": "stop_sign"},
            {"id": 298, "name": "cone"},
            {"id": 1, "name": "car"},
        ],
        "images": [
            {
                "id": 7,
                "coco_url": "http://images.cocodataset.org/val2017/000000000007.jpg",
                "width": 640,
                "height": 480,
            },
            {
                "id": 8,
                "coco_url": "http://images.cocodataset.org/val2017/000000000008.jpg",
                "width": 640,
                "height": 480,
            },
        ],
        "annotations": [
            {"image_id": 7, "category_id": 1019, "bbox": [10.0, 20.0, 30.0, 40.0]},
            {"image_id": 7, "category_id": 1, "bbox": [0.0, 0.0, 100.0, 100.0]},
            {"image_id": 8, "category_id": 1, "bbox": [0.0, 0.0, 100.0, 100.0]},
        ],
    }
    path = tmp_path / "lvis.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("lvis_v1_val.json", json.dumps(doc))

    records = lvis_index(path, min_box_px=8)
    assert len(records) == 1
    assert records[0].image_id == "lvis-7"
    assert records[0].image_url == "http://images.cocodataset.org/val2017/000000000007.jpg"
    assert records[0].boxes == (LabeledBox("stop_sign", 10.0, 20.0, 40.0, 60.0),)


def test_bdd_index_yields_background_records(tmp_path):
    doc = {
        "samples": [
            {
                "filepath": "data/b1c66a42-6f7d68ca.jpg",
                "metadata": {"width": 1280, "height": 720},
                "detections": {
                    "detections": [{"label": "traffic sign", "bounding_box": [0.1, 0.1, 0.1, 0.1]}]
                },
            }
        ]
    }
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(doc))

    records = bdd_index(path)
    assert records == [
        Record(
            image_id="bdd-b1c66a42-6f7d68ca",
            source="bdd100k",
            width=1280,
            height=720,
            boxes=(),
            image_url="https://huggingface.co/datasets/dgural/bdd100k/resolve/main/data/b1c66a42-6f7d68ca.jpg",
        )
    ]


def test_write_record_and_yaml_produce_a_yolo_dataset(tmp_path):
    rec = Record(
        image_id="x1",
        source="test",
        width=160,
        height=90,
        boxes=(LabeledBox("barrier", 16.0, 9.0, 32.0, 18.0),),
        image_bytes=jpeg_bytes(160, 90),
    )
    root = tmp_path / "ds"
    write_record(rec, root, "train")
    write_record(rec, root, "train")
    assert (root / "images" / "train" / "x1.jpg").read_bytes() == rec.image_bytes
    assert (
        root / "labels" / "train" / "x1.txt"
    ).read_text() == "1 0.150000 0.150000 0.100000 0.100000\n"

    yaml_path = write_dataset_yaml(root)
    text = yaml_path.read_text()
    assert "train: images/train" in text
    assert "0: cone" in text and "1: barrier" in text and "2: stop_sign" in text


def test_bdd_label_counts_count_boxes_per_label(tmp_path):
    doc = {
        "samples": [
            {
                "filepath": "data/a.jpg",
                "metadata": {"width": 1280, "height": 720},
                "detections": {
                    "detections": [
                        {"label": "traffic sign", "bounding_box": [0, 0, 0.1, 0.1]},
                        {"label": "car", "bounding_box": [0, 0, 0.1, 0.1]},
                    ]
                },
            },
            {
                "filepath": "data/b.jpg",
                "metadata": {"width": 1280, "height": 720},
                "detections": {"detections": [{"label": "car", "bounding_box": [0, 0, 0.1, 0.1]}]},
            },
        ]
    }
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(doc))
    assert bdd_label_counts(path) == {"images": 2, "boxes": {"car": 2, "traffic sign": 1}}
