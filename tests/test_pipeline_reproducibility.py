from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from ultralytics import YOLO
from ultralytics.data.build import EpochRandomSampler
from ultralytics.data.loaders import LoadImages
from ultralytics.data.rgb_ir import (bgr_hwc_to_rgb_chw, paired_cache_path, read_rgb_ir_pair,
                                      resolve_ir_path)
from ultralytics.data.rgb_ir_check import validate_dataset
from ultralytics.nn.modules.block import SJPA
from ultralytics.utils.reproducibility import capture_manifest, sha256_file


ROOT = Path(__file__).resolve().parents[1]
TINY_MODEL = ROOT / "tests/fixtures/tiny_sjpa_obb.yaml"
FULL_MODEL = ROOT / "ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml"


def make_dataset(root: Path, train_count: int = 4, val_count: int = 2) -> Path:
    for modality in ("rgb", "ir"):
        for split in ("train", "val"):
            (root / modality / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split, count in (("train", train_count), ("val", val_count)):
        for idx in range(count):
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            ir = np.zeros_like(rgb)
            x1, y1, x2, y2 = 12 + idx, 14, 44 + idx, 48
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (11, 77, 231), -1)
            cv2.rectangle(ir, (x1, y1), (x2, y2), (181, 181, 181), -1)
            name = f"{idx:03d}.png"
            assert cv2.imwrite(str(root / "rgb" / split / name), rgb)
            assert cv2.imwrite(str(root / "ir" / split / name), ir)
            p = [x1 / 64, y1 / 64, x2 / 64, y1 / 64, x2 / 64, y2 / 64, x1 / 64, y2 / 64]
            (root / "labels" / split / f"{idx:03d}.txt").write_text(
                "0 " + " ".join(f"{v:.8f}" for v in p) + "\n", encoding="utf-8")
    config = {
        "path": str(root.resolve()),
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
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def nested_sum(value):
    if torch.is_tensor(value):
        return value.float().sum()
    if isinstance(value, dict):
        terms = [nested_sum(v) for v in value.values()]
    elif isinstance(value, (tuple, list)):
        terms = [nested_sum(v) for v in value]
    else:
        terms = []
    return sum(terms) if terms else None


def test_pairing_channel_order_and_manifests(tmp_path: Path):
    data_yaml = make_dataset(tmp_path / "dataset")
    rgb = tmp_path / "dataset/rgb/val/000.png"
    ir = tmp_path / "dataset/ir/val/000.png"
    pair = read_rgb_ir_pair(rgb, ir)
    chw = bgr_hwc_to_rgb_chw(pair)
    np.testing.assert_array_equal(chw[:3], pair[..., :3][..., ::-1].transpose(2, 0, 1))
    np.testing.assert_array_equal(chw[3:], pair[..., 3:][..., ::-1].transpose(2, 0, 1))
    assert resolve_ir_path(rgb, data=yaml.safe_load(data_yaml.read_text()),
                           rgb_roots=[tmp_path / "dataset/rgb/val"],
                           ir_roots=[tmp_path / "dataset/ir/val"]) == ir.resolve()

    rgb_manifest = tmp_path / "rgb.txt"
    ir_manifest = tmp_path / "ir.txt"
    rgb_manifest.write_text("dataset/rgb/val/001.png\ndataset/rgb/val/000.png\n", encoding="utf-8")
    ir_manifest.write_text("dataset/ir/val/001.png\ndataset/ir/val/000.png\n", encoding="utf-8")
    loader = LoadImages(rgb_manifest, ir_source=ir_manifest, data=data_yaml, ch=6)
    assert [Path(x).name for x in loader.files] == ["001.png", "000.png"]
    _, images, _, _ = next(iter(loader))
    assert images[0].shape == (64, 64, 6)


def test_pair_specific_cache_and_source_freshness_identity(tmp_path: Path):
    rgb = tmp_path / "rgb.png"
    ir1, ir2 = tmp_path / "ir1.png", tmp_path / "ir2.png"
    for path in (rgb, ir1, ir2):
        path.write_bytes(path.name.encode())
    a = paired_cache_path(rgb, ir_path=ir1, resize_ir=False)
    b = paired_cache_path(rgb, ir_path=ir2, resize_ir=False)
    c = paired_cache_path(rgb, ir_path=ir1, resize_ir=True)
    assert len({a, b, c}) == 3
    assert a.name.endswith(".rgbir.npy")


def test_dataset_checker_detects_invalid_obb(tmp_path: Path):
    data_yaml = make_dataset(tmp_path / "dataset")
    label = tmp_path / "dataset/labels/train/000.txt"
    label.write_text("1.5 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8")
    report = validate_dataset(str(data_yaml), task="obb", fingerprint="sha256")
    assert not report["ok"]
    assert any("class id must be an integer" in e for e in report["errors"])
    assert report["fingerprint"]["digest"]


def test_epoch_sampler_is_seed_and_epoch_deterministic():
    data = list(range(16))
    sampler = EpochRandomSampler(data, seed=41)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    assert first == list(sampler)
    sampler.set_epoch(4)
    assert first != list(sampler)
    other = EpochRandomSampler(data, seed=42)
    other.set_epoch(3)
    assert first != list(other)


def test_sjpa_translation_has_no_circular_wrap():
    x = torch.zeros(1, 1, 3, 4)
    x[0, 0, 0, 0] = 1
    down_right = SJPA._shift_no_wrap(x, 1, 1)
    assert down_right[0, 0, 1, 1] == 1
    assert down_right[0, 0, 0].sum() == 0
    up_left = SJPA._shift_no_wrap(x, -1, -1)
    assert up_left.sum() == 0  # source pixel leaves the field; it must not wrap to the opposite edge


def test_manifest_is_machine_readable_and_hashes_source(tmp_path: Path):
    data_yaml = make_dataset(tmp_path / "dataset")
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(str(data_yaml), autodownload=False)
    data["yaml_file"] = str(data_yaml)
    manifest = capture_manifest(args={"model": str(TINY_MODEL), "seed": 7}, data=data,
                                source_root=ROOT / "ultralytics", full_data_hash=True)
    json.dumps(manifest, default=str)
    assert manifest["source"]["digest"]
    assert manifest["dataset"]["digest"]
    assert manifest["model_file"]["sha256"] == sha256_file(TINY_MODEL)


@pytest.mark.parametrize("model_path,size", [(TINY_MODEL, 64), (FULL_MODEL, 96)])
def test_model_graph_forward_backward(model_path: Path, size: int):
    model = YOLO(str(model_path)).model.cpu().train()
    x = torch.rand(1, 6, size, size, requires_grad=True)
    output = model(x)
    scalar = nested_sum(output)
    assert scalar is not None
    scalar.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_all_shipped_etfnet_model_configs_build_and_forward():
    """No published ablation/configuration file may be left as dead or undefined code."""
    from ultralytics.nn.tasks import OBBModel, yaml_model_load

    config_dir = ROOT / "ultralytics/cfg/models/etfnet"
    failures = []
    for model_path in sorted(config_dir.glob("*.yaml")):
        try:
            model = OBBModel(yaml_model_load(model_path), verbose=False).eval()
            channels = int(model.yaml.get("ch", 3))
            with torch.inference_mode():
                model(torch.zeros(1, channels, 64, 64))
        except Exception as error:
            failures.append(f"{model_path.name}: {type(error).__name__}: {error}")
    assert not failures, "\n".join(failures)
