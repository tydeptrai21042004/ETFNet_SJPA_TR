"""Reproducibility utilities for ETFNet-SJPA-TR research runs.

The functions in this module are deliberately dependency-light and never access
network resources.  A training run writes a machine-readable manifest that
captures code, data, environment, and randomization state sufficiently to audit
or reproduce the experiment configuration.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from ultralytics.utils import ROOT

_RELEVANT_PACKAGES = (
    "numpy",
    "opencv-python",
    "Pillow",
    "PyYAML",
    "scipy",
    "torch",
    "torchvision",
    "pandas",
    "matplotlib",
    "seaborn",
    "tqdm",
    "psutil",
)
_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json"}
_EXCLUDED_PARTS = {".git", ".venv", "venv", "runs", "dist", "build", "__pycache__", ".pytest_cache"}
_IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_dataset_files(value: Any, kind: str) -> list[Path]:
    """Expand only immutable dataset source files, excluding generated caches.

    ``kind`` is ``rgb``, ``ir``, or ``labels``. Image manifests are included in
    the fingerprint together with the files they reference so list order/path
    edits are detectable, while generated ``*.npy`` and ``*.cache`` files are
    deliberately ignored.
    """
    values = value if isinstance(value, (list, tuple)) else [value]
    output: list[Path] = []
    allowed = {".txt"} if kind == "labels" else _IMAGE_SUFFIXES
    for item in values:
        if item in (None, ""):
            continue
        path = Path(str(item)).expanduser()
        if path.is_dir():
            output.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in allowed)
        elif path.is_file() and path.suffix.lower() == ".txt" and kind != "labels":
            output.append(path)  # hash the manifest itself as well as its targets
            for line in path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text:
                    continue
                candidate = Path(text).expanduser()
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                if candidate.is_file() and candidate.suffix.lower() in allowed:
                    output.append(candidate)
        elif path.is_file() and path.suffix.lower() in allowed:
            output.append(path)
    return sorted({p.resolve() for p in output}, key=lambda p: p.as_posix())


def fingerprint_paths(paths: Iterable[str | Path], *, root: str | Path | None = None,
                      content: bool = False) -> dict[str, Any]:
    """Create a stable metadata or content fingerprint for a collection of files."""
    root_path = Path(root).resolve() if root else None
    resolved = sorted({Path(p).expanduser().resolve() for p in paths if Path(p).expanduser().is_file()},
                      key=lambda p: p.as_posix())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in resolved:
        try:
            name = path.relative_to(root_path).as_posix() if root_path else path.as_posix()
        except ValueError:
            name = path.as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(name.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        if content:
            digest.update(sha256_file(path).encode())
            digest.update(b"\0")
    return {
        "algorithm": "sha256-content" if content else "sha256-path-size",
        "digest": digest.hexdigest(),
        "files": len(resolved),
        "bytes": total_bytes,
    }


def dataset_fingerprint(data: dict[str, Any], *, content: bool = False) -> dict[str, Any]:
    """Fingerprint all configured RGB, IR, and label splits in a resolved data dictionary."""
    root = Path(data.get("path", ".")).expanduser().resolve()
    result: dict[str, Any] = {"root": str(root), "mode": "content" if content else "metadata", "splits": {}}
    combined = hashlib.sha256()
    for split in ("train", "val", "test"):
        groups: dict[str, Any] = {}
        for name, key in (("rgb", split), ("ir", f"{split}_ir"), ("labels", f"{split}_labels")):
            files = _iter_dataset_files(data.get(key), name)
            if files:
                groups[name] = fingerprint_paths(files, root=root, content=content)
                combined.update(split.encode())
                combined.update(name.encode())
                combined.update(groups[name]["digest"].encode())
        if groups:
            result["splits"][split] = groups
    yaml_file = data.get("yaml_file")
    if yaml_file and Path(yaml_file).is_file():
        result["yaml"] = {"path": str(Path(yaml_file).resolve()), "sha256": sha256_file(yaml_file)}
        combined.update(result["yaml"]["sha256"].encode())
    result["digest"] = combined.hexdigest()
    return result


def source_tree_fingerprint(root: str | Path = ROOT) -> dict[str, Any]:
    """Fingerprint research source/configuration files, excluding generated directories."""
    root = Path(root).resolve()
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _SOURCE_SUFFIXES
             and not any(part in _EXCLUDED_PARTS for part in p.relative_to(root).parts)]
    return fingerprint_paths(files, root=root, content=True)


def _git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                                           text=True, timeout=5).strip()
        except Exception:
            return ""

    commit = run("rev-parse", "HEAD")
    if not commit:
        return {"available": False}
    return {
        "available": True,
        "commit": commit,
        "describe": run("describe", "--tags", "--always", "--dirty"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _RELEVANT_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def capture_manifest(*, args: Any = None, data: dict[str, Any] | None = None,
                     source_root: str | Path = ROOT, full_data_hash: bool = False) -> dict[str, Any]:
    """Capture a serializable run manifest."""
    args_dict = vars(args).copy() if args is not None and hasattr(args, "__dict__") else dict(args or {})
    root = Path(source_root).resolve()
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append({
                "index": index,
                "name": props.name,
                "total_memory": int(props.total_memory),
                "capability": list(torch.cuda.get_device_capability(index)),
            })
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "working_directory": str(Path.cwd()),
        "python": {"version": sys.version, "executable": sys.executable, "implementation": platform.python_implementation()},
        "platform": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "packages": _package_versions(),
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "devices": cuda_devices,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "environment": {
            key: os.environ.get(key) for key in (
                "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
            ) if os.environ.get(key) is not None
        },
        "arguments": args_dict,
        "git": _git_info(root),
        "source": source_tree_fingerprint(root),
    }
    if data:
        manifest["dataset"] = dataset_fingerprint(data, content=full_data_hash)
    model_value = args_dict.get("model")
    if model_value and Path(str(model_value)).is_file():
        manifest["model_file"] = {"path": str(Path(model_value).resolve()), "sha256": sha256_file(model_value)}
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically write a reproducibility manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
