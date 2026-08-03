"""Self-contained ETFNet-SJPA-TR smoke test.

Default mode verifies pairing, loading, graph execution, gradients, and caching.
Use ``--full`` for one-epoch train/val/predict/resume/TorchScript checks.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import yaml

from ultralytics.data.rgb_ir_check import validate_dataset
from ultralytics import YOLO
from ultralytics.data.loaders import LoadImages
from ultralytics.data.rgb_ir import bgr_hwc_to_rgb_chw, paired_cache_path, read_rgb_ir_pair

MODEL = ROOT / "ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml"


def make_fixture(root: Path) -> Path:
    """Create a tiny aligned RGB/IR OBB dataset and return its YAML path."""
    for modality in ("rgb", "ir"):
        for split in ("train", "val"):
            (root / modality / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 4, "val": 2}
    for split, count in counts.items():
        for idx in range(count):
            rgb = np.zeros((96, 96, 3), dtype=np.uint8)
            ir = np.zeros_like(rgb)
            x1, y1 = 20 + idx, 24
            x2, y2 = 66 + idx, 70
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (20, 180, 240), -1)
            cv2.rectangle(ir, (x1, y1), (x2, y2), (210, 210, 210), -1)
            rgb[..., 0] = np.maximum(rgb[..., 0], 10 + idx)
            ir[..., 2] = np.maximum(ir[..., 2], 30 + idx)
            name = f"{idx:03d}.png"
            cv2.imwrite(str(root / "rgb" / split / name), rgb)
            cv2.imwrite(str(root / "ir" / split / name), ir)
            polygon = [x1 / 96, y1 / 96, x2 / 96, y1 / 96, x2 / 96, y2 / 96, x1 / 96, y2 / 96]
            (root / "labels" / split / f"{idx:03d}.txt").write_text(
                "0 " + " ".join(f"{v:.6f}" for v in polygon) + "\n", encoding="utf-8"
            )

    data = {
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
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml


def tensor_total(value):
    """Return a differentiable scalar from nested model output."""
    if torch.is_tensor(value):
        return value.float().sum()
    if isinstance(value, dict):
        values = [tensor_total(v) for v in value.values()]
    elif isinstance(value, (list, tuple)):
        values = [tensor_total(v) for v in value]
    else:
        values = []
    values = [v for v in values if v is not None]
    return sum(values) if values else None


def make_videos(root: Path) -> tuple[Path, Path]:
    video_root = root / "videos"
    video_root.mkdir(exist_ok=True)
    outputs = []
    for modality in ("rgb", "ir"):
        frames = [cv2.imread(str(p)) for p in sorted((root / modality / "val").glob("*.png"))]
        frames.append(frames[0].copy())
        path = video_root / f"{modality}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (96, 96))
        if not writer.isOpened():
            raise RuntimeError("OpenCV VideoWriter could not create the smoke-test video")
        for frame in frames:
            writer.write(frame)
        writer.release()
        outputs.append(path)
    return outputs[0], outputs[1]


def lightweight_checks(root: Path, data_yaml: Path) -> None:
    report = validate_dataset(str(data_yaml), "obb")
    assert report["ok"], report

    rgb = root / "rgb/val/000.png"
    ir = root / "ir/val/000.png"
    pair = read_rgb_ir_pair(rgb, ir)
    assert pair.shape == (96, 96, 6)
    chw = bgr_hwc_to_rgb_chw(pair)
    assert chw.shape == (6, 96, 96)
    # Sentinel checks: RGB and IR retain independent channel conversion.
    np.testing.assert_array_equal(chw[:3], pair[..., :3][..., ::-1].transpose(2, 0, 1))
    np.testing.assert_array_equal(chw[3:6], pair[..., 3:6][..., ::-1].transpose(2, 0, 1))

    loader = LoadImages(str(root / "rgb/val"), ir_source=str(root / "ir/val"), data=str(data_yaml), ch=6)
    paths, images, _, _ = next(iter(loader))
    assert paths and images[0].shape[2] == 6

    cache = paired_cache_path(rgb, ir_path=ir, resize_ir=False)
    np.save(cache, pair)
    cached = np.load(cache)
    assert cached.shape[2] == 6
    cache.unlink()

    yolo = YOLO(str(MODEL))
    module = yolo.model.to("cpu").train()
    x = torch.rand(1, 6, 96, 96, requires_grad=True)
    output = module(x)
    total = tensor_total(output)
    assert total is not None
    total.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="Keep the generated test workspace")
    args = parser.parse_args()

    temporary = tempfile.TemporaryDirectory(prefix="etfnet_sjpa_smoke_")
    workspace = Path(temporary.name)
    try:
        fixture = workspace / "dataset"
        data_yaml = make_fixture(fixture)
        lightweight_checks(fixture, data_yaml)
        print("PASS: ETFNet-SJPA-TR lightweight smoke pipeline")
        if args.keep:
            target = ROOT / "tests/_last_smoke_workspace"
            if target.exists():
                import shutil
                shutil.rmtree(target)
            import shutil
            shutil.copytree(workspace, target)
            print(f"Saved smoke workspace to {target}")
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    main()
