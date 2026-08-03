"""Preflight validation for paired RGB--IR detection datasets."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ultralytics.data.rgb_ir import (build_ir_file_list, build_label_file_list, pairing_options,
                                      read_rgb_ir_pair)
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths
from ultralytics.utils.reproducibility import dataset_fingerprint


def collect_images(value: Any) -> list[str]:
    """Expand a split directory, file, list, or text manifest into sorted image paths."""
    values = value if isinstance(value, (list, tuple)) else [value]
    files: list[Path] = []
    for item in values:
        if item in (None, ""):
            continue
        path = Path(str(item)).expanduser()
        if path.is_dir():
            files.extend(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text:
                    continue
                candidate = Path(text).expanduser()
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                if candidate.is_file() and candidate.suffix[1:].lower() in IMG_FORMATS:
                    files.append(candidate.resolve())
        elif path.is_file() and path.suffix[1:].lower() in IMG_FORMATS:
            files.append(path.resolve())
    return [str(p) for p in sorted(set(files), key=lambda p: p.as_posix())]


def _polygon_area(coords: list[float]) -> float:
    points = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def check_label(path: str, nc: int, task: str) -> tuple[list[str], list[str], int]:
    """Validate one label file and return errors, warnings, and object count."""
    errors: list[str] = []
    warnings: list[str] = []
    label_path = Path(path)
    if not label_path.is_file():
        return [f"missing label: {label_path}"], warnings, 0
    objects = 0
    seen: set[tuple[float, ...]] = set()
    expected = 9 if task == "obb" else 5
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        try:
            values = [float(x) for x in text.split()]
        except ValueError:
            errors.append(f"{label_path}:{line_number}: non-numeric label")
            continue
        if len(values) != expected:
            errors.append(f"{label_path}:{line_number}: expected {expected} values for task={task}, got {len(values)}")
            continue
        if not all(math.isfinite(v) for v in values):
            errors.append(f"{label_path}:{line_number}: NaN or infinite value")
            continue
        cls_value = values[0]
        if not cls_value.is_integer():
            errors.append(f"{label_path}:{line_number}: class id must be an integer, got {cls_value}")
            continue
        cls = int(cls_value)
        if cls < 0 or cls >= nc:
            errors.append(f"{label_path}:{line_number}: class {cls} outside [0,{nc - 1}]")
        coords = values[1:]
        if any(v < 0.0 or v > 1.0 for v in coords):
            errors.append(f"{label_path}:{line_number}: normalized coordinates outside [0,1]")
        if task == "detect":
            if coords[2] <= 0 or coords[3] <= 0:
                errors.append(f"{label_path}:{line_number}: box width and height must be positive")
        else:
            if len(set(zip(coords[0::2], coords[1::2]))) < 4:
                errors.append(f"{label_path}:{line_number}: OBB polygon has repeated vertices")
            if _polygon_area(coords) <= 1e-8:
                errors.append(f"{label_path}:{line_number}: OBB polygon has zero area")
        rounded = tuple(round(v, 8) for v in values)
        if rounded in seen:
            warnings.append(f"{label_path}:{line_number}: duplicate label row")
        seen.add(rounded)
        objects += 1
    return errors, warnings, objects


def validate_dataset(data_yaml: str, task: str = "obb", max_errors: int = 100,
                     fingerprint: str = "metadata") -> dict[str, Any]:
    """Validate all configured splits and return a serializable report.

    Args:
        data_yaml: Dataset YAML path.
        task: ``obb`` or ``detect``.
        max_errors: Stop detailed scanning after this many errors.
        fingerprint: ``none``, ``metadata``, or ``sha256``.
    """
    if task not in {"obb", "detect"}:
        raise ValueError(f"Unsupported task: {task}")
    if fingerprint not in {"none", "metadata", "sha256"}:
        raise ValueError("fingerprint must be one of: none, metadata, sha256")
    data = check_det_dataset(data_yaml, autodownload=False)
    data["yaml_file"] = str(Path(data_yaml).expanduser().resolve())
    nc = len(data["names"])
    report: dict[str, Any] = {
        "schema_version": 2,
        "data": data["yaml_file"],
        "task": task,
        "names": data["names"],
        "splits": {},
        "errors": [],
        "warnings": [],
    }
    options = pairing_options(data)
    for split in ("train", "val", "test"):
        if not data.get(split):
            continue
        rgb_files = collect_images(data[split])
        split_errors: list[str] = []
        split_warnings: list[str] = []
        if not rgb_files:
            split_errors.append(f"no RGB images found for split {split}")
            ir_files: list[str] = []
        else:
            try:
                ir_files = build_ir_file_list(rgb_files, data[split], data)
            except Exception as exc:
                split_errors.append(str(exc))
                ir_files = []
        try:
            explicit_labels = build_label_file_list(rgb_files, data[split], data)
            label_files = explicit_labels if explicit_labels is not None else img2label_paths(rgb_files)
        except Exception as exc:
            split_errors.append(str(exc))
            label_files = []

        readable = 0
        objects = 0
        checked = min(len(rgb_files), len(ir_files), len(label_files))
        if ir_files and len(ir_files) != len(rgb_files):
            split_errors.append(f"RGB/IR count mismatch: {len(rgb_files)} vs {len(ir_files)}")
        if label_files and len(label_files) != len(rgb_files):
            split_errors.append(f"RGB/label count mismatch: {len(rgb_files)} vs {len(label_files)}")
        for index in range(checked):
            try:
                pair = read_rgb_ir_pair(rgb_files[index], ir_files[index], resize_ir=options["resize_ir"])
                if pair.shape[2] != 6:
                    split_errors.append(f"{rgb_files[index]}: expected 6 channels, got {pair.shape}")
                readable += 1
            except Exception as exc:
                split_errors.append(str(exc))
            label_errors, label_warnings, count = check_label(label_files[index], nc, task)
            split_errors.extend(label_errors)
            split_warnings.extend(label_warnings)
            objects += count
            if len(report["errors"]) + len(split_errors) >= max_errors:
                split_warnings.append(f"detailed scan stopped after reaching max_errors={max_errors}")
                break
        report["splits"][split] = {
            "rgb_images": len(rgb_files),
            "ir_images": len(ir_files),
            "labels": len(label_files),
            "checked_pairs": checked,
            "readable_pairs": readable,
            "objects": objects,
            "errors": split_errors,
            "warnings": split_warnings,
        }
        report["errors"].extend(f"{split}: {message}" for message in split_errors)
        report["warnings"].extend(f"{split}: {message}" for message in split_warnings)
    if fingerprint != "none":
        report["fingerprint"] = dataset_fingerprint(data, content=fingerprint == "sha256")
    report["ok"] = not report["errors"]
    return report
