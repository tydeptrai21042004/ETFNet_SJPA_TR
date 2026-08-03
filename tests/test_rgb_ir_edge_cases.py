from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from ultralytics.data.base import BaseDataset
from ultralytics.data.rgb_ir import (
    bgr_hwc_to_rgb_chw,
    build_ir_file_list,
    paired_cache_path,
    read_rgb_ir_pair,
    resolve_ir_path,
    rgb_chw_to_bgr_hwc,
)
from ultralytics.data.rgb_ir_check import check_label, validate_dataset


def _write_image(path: Path, shape=(20, 24, 3), value=0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _dataset_yaml(root: Path, resize_ir: bool = False) -> Path:
    for split in ("train", "val"):
        _write_image(root / "rgb" / split / "a.png", value=10)
        _write_image(root / "ir" / split / "a.png", value=20)
        label = root / "labels" / split / "a.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n", encoding="utf-8")
    data = {
        "path": str(root),
        "train": "rgb/train",
        "val": "rgb/val",
        "train_ir": "ir/train",
        "val_ir": "ir/val",
        "train_labels": "labels/train",
        "val_labels": "labels/val",
        "pairing": {"strict": True, "resize_ir": resize_ir},
        "names": {0: "object"},
    }
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_pair_resize_policy(tmp_path: Path):
    rgb, ir = tmp_path / "rgb.png", tmp_path / "ir.png"
    _write_image(rgb, shape=(20, 24, 3), value=10)
    _write_image(ir, shape=(15, 12, 3), value=20)
    with pytest.raises(ValueError, match="size mismatch"):
        read_rgb_ir_pair(rgb, ir, resize_ir=False)
    pair = read_rgb_ir_pair(rgb, ir, resize_ir=True)
    assert pair.shape == (20, 24, 6)


def test_channel_conversion_roundtrip_for_both_modalities():
    image = np.arange(7 * 9 * 6, dtype=np.uint8).reshape(7, 9, 6)
    chw = bgr_hwc_to_rgb_chw(image)
    restored = rgb_chw_to_bgr_hwc(chw)
    np.testing.assert_array_equal(restored, image)


def test_exact_component_replacement_does_not_touch_substrings(tmp_path: Path):
    rgb = tmp_path / "my_images_backup" / "rgb" / "a.png"
    ir = tmp_path / "my_images_backup" / "ir" / "a.png"
    _write_image(rgb)
    _write_image(ir)
    data = {"pairing": {"rgb_token": "rgb", "ir_token": "ir"}}
    resolved = resolve_ir_path(rgb, data=data)
    assert resolved == ir.resolve()
    assert "my_images_backup" in str(resolved)


def test_explicit_ir_mapping_takes_precedence(tmp_path: Path):
    rgb = tmp_path / "rgb" / "a.png"
    ir = tmp_path / "chosen" / "thermal.png"
    _write_image(rgb)
    _write_image(ir)
    assert resolve_ir_path(rgb, explicit_ir=ir) == ir.resolve()


def test_strict_pairing_reports_missing_partner(tmp_path: Path):
    rgb_root = tmp_path / "rgb"
    _write_image(rgb_root / "a.png")
    data = {
        "path": str(tmp_path),
        "train": "rgb",
        "train_ir": "ir",
        "pairing": {"strict": True},
        "yaml_file": str(tmp_path / "data.yaml"),
    }
    with pytest.raises(FileNotFoundError, match="Missing 1 RGB--IR pair"):
        build_ir_file_list([str(rgb_root / "a.png")], rgb_root, data)


def test_corrupt_image_is_detected_by_preflight(tmp_path: Path):
    data_yaml = _dataset_yaml(tmp_path / "dataset")
    (tmp_path / "dataset/ir/val/a.png").write_bytes(b"not-an-image")
    report = validate_dataset(str(data_yaml), task="obb", fingerprint="none")
    assert not report["ok"]
    assert any("could not be read" in error for error in report["errors"])


@pytest.mark.parametrize(
    "row, message",
    [
        ("0 nan 0.2 0.8 0.2 0.8 0.8 0.2 0.8", "NaN or infinite"),
        ("2 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8", "outside [0,0]"),
        ("0 0.2 0.2 1.2 0.2 0.8 0.8 0.2 0.8", "outside [0,1]"),
        ("0 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2", "repeated vertices"),
    ],
)
def test_label_validation_edge_cases(tmp_path: Path, row: str, message: str):
    path = tmp_path / "label.txt"
    path.write_text(row + "\n", encoding="utf-8")
    errors, _, _ = check_label(str(path), nc=1, task="obb")
    assert any(message in error for error in errors)


def test_duplicate_label_is_warning_not_silent(tmp_path: Path):
    path = tmp_path / "label.txt"
    row = "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    errors, warnings, count = check_label(str(path), nc=1, task="obb")
    assert not errors and count == 2
    assert any("duplicate label row" in warning for warning in warnings)


def test_pair_specific_cache_name_changes_with_partner_and_policy(tmp_path: Path):
    rgb, ir1, ir2 = tmp_path / "rgb.png", tmp_path / "ir1.png", tmp_path / "ir2.png"
    for path in (rgb, ir1, ir2):
        path.write_bytes(path.name.encode())
    names = {
        paired_cache_path(rgb, ir_path=ir1, resize_ir=False).name,
        paired_cache_path(rgb, ir_path=ir2, resize_ir=False).name,
        paired_cache_path(rgb, ir_path=ir1, resize_ir=True).name,
    }
    assert len(names) == 3


def test_disk_cache_rebuilds_when_ir_source_is_newer(tmp_path: Path):
    root = tmp_path / "dataset"
    data_yaml = _dataset_yaml(root)

    class MinimalDataset(BaseDataset):
        def get_labels(self):
            return [
                {
                    "im_file": file,
                    "shape": (20, 24),
                    "cls": np.zeros((0, 1), dtype=np.float32),
                    "bboxes": np.zeros((0, 4), dtype=np.float32),
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
                for file in self.im_files
            ]

        def build_transforms(self, hyp=None):
            return lambda x: x

    data = yaml.safe_load(data_yaml.read_text())
    data["yaml_file"] = str(data_yaml)
    hyp = SimpleNamespace(ch=6)
    dataset = MinimalDataset(root / "rgb/train", imgsz=32, cache=False, hyp=hyp)
    dataset.data = data
    # BaseDataset resolves data during construction in real YOLODataset. Rebuild
    # the paired fields explicitly for this minimal unit fixture.
    dataset.la_files = [str(root / "ir/train/a.png")]
    dataset.npy_files = [paired_cache_path(dataset.im_files[0], ir_path=dataset.la_files[0], resize_ir=False)]
    dataset.cache_images_to_disk(0)
    cache = dataset.npy_files[0]
    first_mtime = cache.stat().st_mtime_ns
    time.sleep(0.01)
    os.utime(dataset.la_files[0], None)
    dataset.cache_images_to_disk(0)
    assert cache.stat().st_mtime_ns > first_mtime
    assert np.load(cache, allow_pickle=False).shape[2] == 6
