"""RGB--IR pairing and channel-order utilities for ETFNet.

The internal HWC representation is always::

    [B_rgb, G_rgb, R_rgb, B_ir, G_ir, R_ir]

The tensor representation is always::

    [R_rgb, G_rgb, B_rgb, R_ir, G_ir, B_ir]

Keeping this convention in one module prevents the train/predict modality swap
that affected the original repository.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from ultralytics.utils import LOGGER
from ultralytics.utils.patches import imread

_MODALITY_CHANNELS = 3


def is_rgb_ir_channels(channels: int | None) -> bool:
    """Return True when the model expects a paired 6-channel RGB--IR input."""
    return int(channels or 3) == 6


def load_data_yaml(data: Any) -> dict:
    """Load a dataset dictionary from a dict or YAML path; return an empty dict otherwise."""
    if isinstance(data, dict):
        return data
    if data in (None, "", False):
        return {}
    path = Path(str(data)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise TypeError(f"Dataset YAML must contain a mapping, got {type(obj).__name__}: {path}")
    obj.setdefault("yaml_file", str(path.resolve()))
    return obj


def _dataset_root(data: dict | None) -> Path:
    """Return the absolute dataset root used for relative split/pair paths."""
    data = data or {}
    yaml_parent = Path(data.get("yaml_file", ".")).expanduser().resolve().parent
    root_value = data.get("path", yaml_parent)
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = yaml_parent / root
    return root.resolve()


def _flatten_paths(value: Any, parent: Path | None = None) -> list[Path]:
    """Normalize a dataset split value into paths without expanding directories."""
    values = value if isinstance(value, (list, tuple)) else [value]
    out: list[Path] = []
    for item in values:
        if item in (None, ""):
            continue
        p = Path(str(item)).expanduser()
        if not p.is_absolute() and parent is not None:
            p = parent / p
        out.append(Path(os.path.abspath(str(p))))
    return out


def infer_split_name(img_path: Any, data: dict | None) -> str | None:
    """Infer train/val/test from the path passed to YOLODataset."""
    if not data:
        return None
    root = _dataset_root(data)
    current = {str(p) for p in _flatten_paths(img_path, root)}
    for split in ("train", "val", "test"):
        expected = {str(p) for p in _flatten_paths(data.get(split), root)}
        if current and current == expected:
            return split
    # Fallback to containment for configurations that resolve relative paths upstream.
    current_names = {Path(p).name.lower() for p in current}
    for split in ("train", "val", "test"):
        if split in current_names:
            return split
    return None


def _replace_path_component(path: Path, rgb_token: str, ir_token: str) -> Path | None:
    """Replace one exact path component, never an arbitrary substring."""
    parts = list(path.parts)
    lowered = [p.lower() for p in parts]
    try:
        idx = len(parts) - 1 - lowered[::-1].index(rgb_token.lower())
    except ValueError:
        return None
    parts[idx] = ir_token
    return Path(*parts)


def _relative_to_any(path: Path, roots: Iterable[Path]) -> tuple[Path, Path] | None:
    for root in roots:
        try:
            return root, path.relative_to(root)
        except ValueError:
            continue
    return None


def pairing_options(data: dict | None) -> dict:
    """Return normalized pairing options from a dataset dictionary."""
    data = data or {}
    raw = data.get("pairing", {}) or {}
    if not isinstance(raw, dict):
        raise TypeError("data.yaml 'pairing' must be a mapping")
    return {
        "strict": bool(raw.get("strict", data.get("pair_strict", True))),
        "resize_ir": bool(raw.get("resize_ir", data.get("pair_resize", False))),
        "rgb_token": str(raw.get("rgb_token", data.get("rgb_token", "images"))),
        "ir_token": str(raw.get("ir_token", data.get("ir_token", "infrared"))),
        "cache_suffix": str(raw.get("cache_suffix", ".rgbir.npy")),
    }


def resolve_ir_path(
    rgb_path: str | Path,
    *,
    data: dict | None = None,
    split: str | None = None,
    rgb_roots: Iterable[str | Path] | None = None,
    ir_roots: Iterable[str | Path] | None = None,
    explicit_ir: str | Path | None = None,
) -> Path:
    """Resolve the IR partner for one RGB image.

    Resolution order:
      1. ``explicit_ir`` for one-file prediction.
      2. Split-specific ``train_ir`` / ``val_ir`` / ``test_ir`` roots.
      3. Global ``rgb_root`` -> ``ir_root`` mapping.
      4. Exact path-component replacement configured under ``pairing``.
    """
    rgb = Path(rgb_path).expanduser().resolve()
    data = data or {}
    opts = pairing_options(data)
    dataset_root = _dataset_root(data)

    if explicit_ir not in (None, ""):
        candidate = Path(str(explicit_ir)).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Explicit IR source does not exist: {candidate}")
        return candidate.resolve()

    # Explicit source roots supplied by paired inference take precedence over
    # dataset YAML mappings and preserve arbitrary user-selected folder pairs.
    explicit_rgb_roots = [Path(p).expanduser().resolve() for p in (rgb_roots or [])]
    explicit_ir_roots = [Path(p).expanduser().resolve() for p in (ir_roots or [])]
    if explicit_rgb_roots and explicit_ir_roots:
        rel = _relative_to_any(rgb, explicit_rgb_roots)
        if rel:
            source_root, relative = rel
            source_index = explicit_rgb_roots.index(source_root)
            target_root = explicit_ir_roots[min(source_index, len(explicit_ir_roots) - 1)]
            return (target_root / relative).resolve()

    split = split or infer_split_name(rgb_roots or [], data)
    if split and data.get(f"{split}_ir") not in (None, ""):
        rroots = [Path(p).expanduser().resolve() for p in
                  (rgb_roots or _flatten_paths(data.get(split), dataset_root))]
        iroots = _flatten_paths(data[f"{split}_ir"], dataset_root)
        rel = _relative_to_any(rgb, rroots)
        if rel:
            source_root, relative = rel
            source_index = rroots.index(source_root)
            target_root = iroots[min(source_index, len(iroots) - 1)]
            return (target_root / relative).resolve()

    configured_rgb_roots = [Path(p).expanduser().resolve() for p in (rgb_roots or [])]
    configured_ir_roots = [Path(p).expanduser().resolve() for p in (ir_roots or [])]
    if not configured_rgb_roots:
        configured_rgb_roots = _flatten_paths(data.get("rgb_root"), dataset_root)
    if not configured_ir_roots:
        configured_ir_roots = _flatten_paths(data.get("ir_root"), dataset_root)
    if configured_rgb_roots and configured_ir_roots:
        rel = _relative_to_any(rgb, configured_rgb_roots)
        if rel:
            source_root, relative = rel
            source_index = configured_rgb_roots.index(source_root)
            target_root = configured_ir_roots[min(source_index, len(configured_ir_roots) - 1)]
            return (target_root / relative).resolve()

    replaced = _replace_path_component(rgb, opts["rgb_token"], opts["ir_token"])
    if replaced is not None:
        return replaced.resolve()

    # Compatibility fallbacks for common public-dataset layouts. These replace
    # only exact path components and are therefore safe against names such as
    # ``my_images_backup``.
    for rgb_token, ir_token in (("images", "image"), ("rgb", "ir"), ("visible", "infrared")):
        replaced = _replace_path_component(rgb, rgb_token, ir_token)
        if replaced is not None and replaced.exists():
            return replaced.resolve()

    raise FileNotFoundError(
        f"Cannot resolve IR partner for RGB image: {rgb}\n"
        "Define train_ir/val_ir/test_ir, rgb_root+ir_root, or pairing.rgb_token/pairing.ir_token in data.yaml."
    )


def build_ir_file_list(rgb_files: list[str], img_path: Any, data: dict | None) -> list[str]:
    """Resolve and validate all IR partners for a dataset split."""
    data = data or {}
    split = infer_split_name(img_path, data)
    dataset_root = _dataset_root(data)
    rgb_roots = _flatten_paths(data.get(split), dataset_root) if split else _flatten_paths(img_path, dataset_root)
    opts = pairing_options(data)
    ir_files: list[str] = []
    missing: list[tuple[str, str]] = []
    for rgb in rgb_files:
        try:
            ir = resolve_ir_path(rgb, data=data, split=split, rgb_roots=rgb_roots)
        except FileNotFoundError:
            ir = Path("<unresolved>")
        if not ir.is_file():
            missing.append((rgb, str(ir)))
        ir_files.append(str(ir))
    if missing and opts["strict"]:
        preview = "\n".join(f"  RGB: {r}\n  IR : {i}" for r, i in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} RGB--IR pair(s). First mismatches:\n{preview}\n"
            "Set pairing.strict: false only for diagnostic work; training requires complete pairs."
        )
    if missing:
        LOGGER.warning(f"WARNING ⚠️ {len(missing)} unresolved RGB--IR pairs will fail when loaded.")
    return ir_files


def read_rgb_ir_pair(
    rgb_path: str | Path,
    ir_path: str | Path,
    *,
    resize_ir: bool = False,
) -> np.ndarray:
    """Read and concatenate one pair in canonical HWC BGR order."""
    rgb = imread(str(rgb_path), cv2.IMREAD_COLOR)
    ir = imread(str(ir_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"RGB image could not be read: {rgb_path}")
    if ir is None:
        raise FileNotFoundError(f"IR image could not be read: {ir_path}")
    if rgb.shape[:2] != ir.shape[:2]:
        if not resize_ir:
            raise ValueError(
                f"RGB/IR size mismatch: RGB {rgb.shape[:2]} at {rgb_path}; IR {ir.shape[:2]} at {ir_path}. "
                "Align the dataset or set pairing.resize_ir: true."
            )
        ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.concatenate((rgb, ir), axis=2)


def validate_hwc_modalities(image: np.ndarray, channels: int) -> None:
    """Validate an HWC image against the configured channel count."""
    if image is None or image.ndim != 3:
        raise ValueError(f"Expected HWC image, got {None if image is None else image.shape}")
    if image.shape[2] != int(channels):
        raise ValueError(f"Expected {channels} channels, got image shape {image.shape}")


def bgr_hwc_to_rgb_chw(image: np.ndarray) -> np.ndarray:
    """Convert each 3-channel modality independently from BGR/HWC to RGB/CHW."""
    if image.ndim == 2:
        image = image[..., None]
    channels = image.shape[2]
    if channels == 1:
        return np.ascontiguousarray(image.transpose(2, 0, 1))
    if channels % _MODALITY_CHANNELS != 0:
        raise ValueError(
            f"Expected channels grouped by three (RGB or RGB--IR), got shape {image.shape}."
        )
    modalities = [image[..., start:start + 3][..., ::-1] for start in range(0, channels, 3)]
    rgb_hwc = np.concatenate(modalities, axis=2)
    return np.ascontiguousarray(rgb_hwc.transpose(2, 0, 1))


def rgb_chw_to_bgr_hwc(tensor_image: np.ndarray) -> np.ndarray:
    """Inverse of :func:`bgr_hwc_to_rgb_chw` for visualization."""
    if tensor_image.ndim != 3:
        raise ValueError(f"Expected CHW array, got {tensor_image.shape}")
    hwc = tensor_image.transpose(1, 2, 0)
    channels = hwc.shape[2]
    if channels % _MODALITY_CHANNELS != 0:
        raise ValueError(f"Expected CHW channels grouped by three, got {tensor_image.shape}")
    modalities = [hwc[..., start:start + 3][..., ::-1] for start in range(0, channels, 3)]
    return np.ascontiguousarray(np.concatenate(modalities, axis=2))


def rgb_view(image: np.ndarray) -> np.ndarray:
    """Return the RGB-camera BGR image used for visualization and saved predictions."""
    if image.ndim == 3 and image.shape[2] >= 3:
        return np.ascontiguousarray(image[..., :3])
    return image


def ir_view(image: np.ndarray) -> np.ndarray | None:
    """Return the IR-camera BGR image when present."""
    if image.ndim == 3 and image.shape[2] >= 6:
        return np.ascontiguousarray(image[..., 3:6])
    return None


def paired_cache_path(rgb_path: str | Path, suffix: str = ".rgbir.npy", *,
                      ir_path: str | Path | None = None, resize_ir: bool = False) -> Path:
    """Return a pair-specific cache path.

    The short signature prevents a cache produced for one IR partner or resize
    policy from being silently reused for another pairing configuration.
    """
    path = Path(rgb_path)
    identity = f"{Path(ir_path).expanduser().resolve() if ir_path else ''}|resize={int(resize_ir)}"
    signature = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]
    return path.with_name(f"{path.stem}.{signature}{suffix}")


def build_label_file_list(rgb_files: list[str], img_path: Any, data: dict | None) -> list[str] | None:
    """Build label paths from optional split-specific roots.

    Supported keys are ``train_labels``, ``val_labels``, ``test_labels`` or
    global ``labels_root``. Returning ``None`` requests the stock Ultralytics
    ``images`` -> ``labels`` convention.
    """
    data = data or {}
    split = infer_split_name(img_path, data)
    dataset_root = _dataset_root(data)
    label_value = data.get(f'{split}_labels') if split else None
    rgb_value = data.get(split) if split else img_path
    if label_value in (None, '') and data.get('labels_root') in (None, ''):
        return None
    rgb_roots = _flatten_paths(rgb_value, dataset_root)
    label_roots = _flatten_paths(label_value if label_value not in (None, '') else data.get('labels_root'), dataset_root)
    if not label_roots:
        return None
    output = []
    for rgb_file in rgb_files:
        rgb = Path(rgb_file).resolve()
        rel = _relative_to_any(rgb, rgb_roots)
        if rel is None:
            raise ValueError(f'Cannot make label path for {rgb}; it is outside configured RGB roots {rgb_roots}')
        source_root, relative = rel
        index = rgb_roots.index(source_root)
        target_root = label_roots[min(index, len(label_roots) - 1)]
        output.append(str((target_root / relative).with_suffix('.txt').resolve()))
    return output
