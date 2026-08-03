from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _image(path: Path, value: int, size=(96, 80), channels=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 4:
        image = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        image[..., :3] = value
        image[..., 3] = min(255, value + 20)
    else:
        image = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def create_m3fd(root: Path, pairs: int = 6) -> Path:
    base = root / "M3FD"
    train, val = [], []
    for i in range(pairs):
        name = f"frame_{i:03d}.png"
        _image(base / "vi" / name, 30 + i)
        _image(base / "ir" / name, 130 + i)
        (base / "labels").mkdir(parents=True, exist_ok=True)
        (base / "labels" / f"frame_{i:03d}.txt").write_text(
            f"{i % 6} 0.5 0.5 0.3 0.4\n", encoding="utf-8")
        (val if i >= pairs - 2 else train).append(name)
    (base / "meta").mkdir(parents=True, exist_ok=True)
    (base / "meta/train.txt").write_text("\n".join(train) + "\n")
    (base / "meta/val.txt").write_text("\n".join(val) + "\n")
    return root


def create_vedai(root: Path, pairs: int = 6, variant="512") -> Path:
    ann = root / f"annotations{variant}"
    for i in range(pairs):
        stem = f"{i:08d}"
        _image(root / f"images{variant}" / "visible" / f"{stem}_rgb.png", 40 + i, size=(64, 64))
        _image(root / f"images{variant}" / "nir" / f"{stem}_nir.png", 140 + i, size=(64, 64))
        ann.mkdir(parents=True, exist_ok=True)
        # center, angle, class, instance, flag, x1..x4, y1..y4
        ann.joinpath(f"{stem}.txt").write_text(
            f"30 30 0 {i % 9 + 1} 1 0 20 40 40 20 20 20 40 40\n", encoding="utf-8")
    return root


def _coco_dataset(root: Path, class_names, rgb_name="visible", ir_name="thermal", pairs=6):
    for split, count in (("train", pairs - 2), ("val", 2)):
        images, annotations = [], []
        for i in range(count):
            global_i = i if split == "train" else pairs - 2 + i
            filename = f"frame_{global_i:03d}.jpg"
            _image(root / split / rgb_name / filename, 50 + global_i)
            _image(root / split / ir_name / filename, 150 + global_i)
            images.append({"id": global_i + 1, "file_name": f"{split}/{ir_name}/{filename}", "width": 96, "height": 80})
            annotations.append({
                "id": global_i + 1, "image_id": global_i + 1, "category_id": global_i % len(class_names) + 1,
                "bbox": [20, 15, 30, 35], "area": 1050, "iscrowd": 0,
            })
        payload = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
        }
        (root / f"{split}_thermal.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def create_flir(root: Path, pairs: int = 6) -> Path:
    return _coco_dataset(root, ("person", "car", "bicycle"), pairs=pairs)


def create_rgbtdrone(root: Path, pairs: int = 6) -> Path:
    return _coco_dataset(root, ("person", "rider", "crowd"), pairs=pairs)


def create_cvc14(root: Path, pairs: int = 6) -> Path:
    for i in range(pairs):
        split = "Train" if i < pairs - 2 else "Test"
        light = "Day" if i % 2 == 0 else "Night"
        name = f"frame_{i:03d}.png"
        _image(root / light / "Visible" / split / "FramesPos" / name, 60 + i)
        _image(root / light / "FIR" / split / "FramesPos" / name, 160 + i)
        label = root / light / "Visible" / split / "Annotations" / f"frame_{i:03d}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("20 15 30 35\n", encoding="utf-8")
    return root
