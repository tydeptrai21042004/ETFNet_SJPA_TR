"""Utilities for creating a miniature archive with the official NII-CU layout."""
from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np


def create_nii_cu_archive(path: Path, *, include_rgb_t: bool = True, mismatch: bool = False) -> Path:
    source = path.parent / "nii_source"
    variants = ("4-channel", "rgb-t") if include_rgb_t else ("4-channel",)
    for variant in variants:
        for split, count in (("train", 4), ("val", 2)):
            for folder in ("images/rgb", "images/thermal", "labels"):
                (source / "NII_CU_MAPD_dataset" / variant / folder / split).mkdir(parents=True, exist_ok=True)
            for index in range(count):
                height, width = 80, 96
                rgb = np.zeros((height, width, 3), dtype=np.uint8)
                ir_height = height + 1 if mismatch and split == "train" and index == 0 else height
                ir = np.zeros((ir_height, width, 3), dtype=np.uint8)
                x1, y1, x2, y2 = 10 + index, 12, 42 + index, 58
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (20, 120, 240), -1)
                cv2.rectangle(ir, (x1, y1), (x2, min(y2, ir_height - 1)), (190, 190, 190), -1)
                name = f"frame_{index:03d}.png"
                rgb_path = source / "NII_CU_MAPD_dataset" / variant / "images/rgb" / split / name
                ir_path = source / "NII_CU_MAPD_dataset" / variant / "images/thermal" / split / name
                assert cv2.imwrite(str(rgb_path), rgb)
                assert cv2.imwrite(str(ir_path), ir)
                label = source / "NII_CU_MAPD_dataset" / variant / "labels" / split / f"frame_{index:03d}.txt"
                # Include header, one valid box, one filtered bad box and one clipped box.
                label.write_text(
                    "x1\ty1\tx2\ty2\ttype\toccluded\tbad\n"
                    f"{x1}\t{y1}\t{x2}\t{y2}\t0\t{index % 2}\t0\n"
                    "2\t3\t8\t9\t1\t0\t1\n"
                    f"-2\t4\t{width + 3}\t18\t2\t0\t0\n",
                    encoding="utf-8",
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(source.rglob("*")):
            if item.is_file():
                bundle.write(item, item.relative_to(source).as_posix())
    return path
