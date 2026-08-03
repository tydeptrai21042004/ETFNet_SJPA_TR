"""Create a tiny paired RGB/IR OBB fixture for pipeline smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()

    for modality in ("rgb", "ir"):
        for split in ("train", "val"):
            (root / modality / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, count in {"train": 4, "val": 2}.items():
        for idx in range(count):
            x1, y1, x2, y2 = 20 + idx, 24, 66 + idx, 70
            rgb = Image.new("RGB", (96, 96), (10 + idx, 0, 0))
            ir = Image.new("RGB", (96, 96), (0, 0, 30 + idx))
            ImageDraw.Draw(rgb).rectangle((x1, y1, x2, y2), fill=(240, 180, 20))
            ImageDraw.Draw(ir).rectangle((x1, y1, x2, y2), fill=(210, 210, 210))
            name = f"{idx:03d}.png"
            rgb.save(root / "rgb" / split / name)
            ir.save(root / "ir" / split / name)
            polygon = [x1 / 96, y1 / 96, x2 / 96, y1 / 96, x2 / 96, y2 / 96, x1 / 96, y2 / 96]
            (root / "labels" / split / f"{idx:03d}.txt").write_text(
                "0 " + " ".join(f"{value:.6f}" for value in polygon) + "\n", encoding="utf-8"
            )

    config = {
        "path": str(root),
        "train": "rgb/train",
        "val": "rgb/val",
        "test": "rgb/val",
        "train_ir": "ir/train",
        "val_ir": "ir/val",
        "test_ir": "ir/val",
        "train_labels": "labels/train",
        "val_labels": "labels/val",
        "test_labels": "labels/val",
        "pairing": {"strict": True, "resize_ir": False},
        "names": {0: "object"},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    videos = root / "videos"
    videos.mkdir(exist_ok=True)
    for modality in ("rgb", "ir"):
        frames = [cv2.imread(str(path)) for path in sorted((root / modality / "val").glob("*.png"))]
        frames.append(frames[0].copy())
        target = videos / f"{modality}.avi"
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (96, 96))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create fixture video")
        for frame in frames:
            writer.write(frame)
        writer.release()

    print(root / "data.yaml")


if __name__ == "__main__":
    main()
