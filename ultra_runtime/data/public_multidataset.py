"""Download and preprocess compact public RGB--IR/RGB--NIR detection datasets.

Supported datasets
------------------
- M3FD (official Google Drive folder)
- VEDAI 512/1024 (official GREYC multipart archives)
- FLIR-Aligned (authors' Google Drive file with public archive fallback)
- RGBTDronePerson (official Google Drive folder)
- CVC-14 (ModelScope mirror of the CVC release; local-archive fallback)

Every converter writes the same strict canonical layout used by ETFNet::

    processed/<variant>/rgb/images/{train,val,test}
    processed/<variant>/ir/images/{train,val,test}
    processed/<variant>/labels/{train,val,test}
    processed/<variant>/data.yaml

Network providers are optional dependencies. Install them with
``pip install -e '.[data]'``. Local archives/directories remain supported for
reproducibility and for providers that require browser authentication.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np
import yaml

from ultralytics import __version__
from ultralytics.data.public_rgb_ir import (
    DatasetPreparationError,
    PublicDatasetSpec,
    _image_files,
    _link_or_copy,
    _sha256,
    download_with_resume,
    safe_extract_zip,
)
from ultralytics.data.utils import IMG_FORMATS


M3FD_DRIVE_FOLDER = "1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6"
RGBT_DRIVE_FOLDER = "1Mi3NXQ-YG1iiIWkPbe3GQoDK68dARMN6"
FLIR_DRIVE_FILE = "1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y"
FLIR_FALLBACK_URL = (
    "https://huggingface.co/datasets/UserNae3/FLIR_aligned/resolve/main/aligned.zip?download=true"
)
CVC_MODELSCOPE_ID = "OmniData/CVC-14"
VEDAI_BASE = "https://downloads.greyc.fr/vedai"


ADDITIONAL_DATASETS: dict[str, PublicDatasetSpec] = {
    "m3fd": PublicDatasetSpec(
        key="m3fd",
        title="M3FD Multi-Scenario Multi-Modality Dataset",
        description="4,200 registered visible/infrared pairs with six detection classes.",
        homepage="https://github.com/JinyuanLiu-CV/TarDAL",
        citation_doi="10.1109/CVPR52688.2022.00571",
        dataset_doi=None,
        license_name="Official M3FD research-use terms",
        license_url="https://github.com/JinyuanLiu-CV/TarDAL",
        archive_name="M3FD-google-drive-folder",
        download_url=f"gdrive-folder:{M3FD_DRIVE_FOLDER}",
        expected_download_gb=3.5,
        variants=("default",),
        default_variant="default",
        classes=("person", "car", "bus", "motorcycle", "lamp", "truck"),
    ),
    "vedai": PublicDatasetSpec(
        key="vedai",
        title="Vehicle Detection in Aerial Imagery (VEDAI)",
        description="Small RGB/NIR aerial vehicle benchmark with oriented annotations.",
        homepage="https://downloads.greyc.fr/vedai/",
        citation_doi="10.1016/j.jvcir.2015.11.002",
        dataset_doi=None,
        license_name="VEDAI Terms and Conditions of Use",
        license_url=f"{VEDAI_BASE}/TermsandConditionsofUseVeDAI2014.pdf",
        archive_name="VEDAI-official-multipart",
        download_url=f"{VEDAI_BASE}/",
        expected_download_gb=1.3,
        variants=("512", "1024"),
        default_variant="512",
        classes=("plane", "boat", "camping-car", "car", "pickup", "tractor", "truck", "van", "other"),
    ),
    "flir-aligned": PublicDatasetSpec(
        key="flir-aligned",
        title="Aligned FLIR Visible-Thermal Dataset",
        description="Aligned visible/thermal ADAS pairs with person, car, and bicycle annotations.",
        homepage="https://huggingface.co/datasets/jsonhash/FLIR_aligned",
        citation_doi="10.48550/arXiv.2009.12664",
        dataset_doi=None,
        license_name="FLIR dataset terms (license acceptance required)",
        license_url="https://www.flir.com/oem/adas/adas-dataset-form/",
        archive_name="aligned.zip",
        download_url=f"gdrive-file:{FLIR_DRIVE_FILE}",
        expected_download_gb=2.3,
        variants=("aligned",),
        default_variant="aligned",
        classes=("person", "car", "bicycle"),
    ),
    "rgbtdroneperson": PublicDatasetSpec(
        key="rgbtdroneperson",
        title="RGBTDronePerson",
        description="6,125 UAV visible/thermal pairs with person, rider, and crowd annotations.",
        homepage="https://nnnnerd.github.io/RGBTDronePerson/",
        citation_doi="10.1016/j.isprsjprs.2023.08.016",
        dataset_doi=None,
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        archive_name="RGBTDronePerson-google-drive-folder",
        download_url=f"gdrive-folder:{RGBT_DRIVE_FOLDER}",
        expected_download_gb=2.0,
        variants=("default",),
        default_variant="default",
        classes=("person", "rider", "crowd"),
    ),
    "cvc-14": PublicDatasetSpec(
        key="cvc-14",
        title="CVC-14 Visible-FIR Pedestrian Dataset",
        description="Day/night grayscale-visible and FIR pedestrian pairs with residual misalignment.",
        homepage="https://www.mdpi.com/1424-8220/16/6/820",
        citation_doi="10.3390/s16060820",
        dataset_doi=None,
        license_name="CC BY-NC 4.0 (mirror metadata; review original dataset terms)",
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        archive_name="CVC-14-modelscope-snapshot",
        download_url=f"modelscope:{CVC_MODELSCOPE_ID}",
        expected_download_gb=6.0,
        variants=("default",),
        default_variant="default",
        classes=("person",),
    ),
}


@dataclass(frozen=True)
class ObjectLabel:
    class_id: int
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PairSample:
    key: str
    rgb: Path
    ir: Path
    labels: tuple[ObjectLabel, ...]
    split: str


def _require_optional(module: str, install_hint: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise DatasetPreparationError(
            f"Optional dependency '{module}' is required for this download provider. {install_hint}"
        ) from exc


def _gdown_major(gdown: Any) -> int:
    match = re.match(r"(\d+)", str(getattr(gdown, "__version__", "0")))
    return int(match.group(1)) if match else 0


def _download_gdrive_folder(folder_id: str, destination: Path) -> Path:
    gdown = _require_optional("gdown", "Install with: pip install -e '.[data]'")
    if _gdown_major(gdown) < 6:
        raise DatasetPreparationError(
            "Google Drive folder datasets require gdown>=6 because older releases are limited to 50 files. "
            "Use Python 3.10+ and `pip install -e '.[data]'`, or provide --archive with a local folder/archive."
        )
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        result = gdown.download_folder(url=url, output=str(destination), quiet=False, use_cookies=False)
    except Exception as exc:
        raise DatasetPreparationError(
            "Google Drive folder download failed. The provider may be rate-limited or require browser "
            "confirmation; download the official folder manually and pass --archive <folder-or-archive>."
        ) from exc
    if not result and not any(destination.rglob("*")):
        raise DatasetPreparationError("Google Drive folder download returned no files.")
    return destination


def _download_gdrive_file(file_id: str, destination: Path) -> Path:
    gdown = _require_optional("gdown", "Install with: pip install -e '.[data]'")
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        # gdown 6 accepts share URLs directly and removed the old fuzzy option.
        result = gdown.download(url=url, output=str(destination), quiet=False, resume=True)
    except TypeError:
        # Compatibility for gdown 5 when users supply a local environment manually.
        result = gdown.download(id=file_id, output=str(destination), quiet=False, fuzzy=True, resume=True)
    except Exception as exc:
        raise DatasetPreparationError("Google Drive file download failed.") from exc
    if not result or not destination.is_file() or destination.stat().st_size == 0:
        raise DatasetPreparationError("Google Drive file download failed or produced an empty file.")
    return destination


def _download_modelscope(dataset_id: str, destination: Path) -> Path:
    """Download a ModelScope dataset snapshot without importing training frameworks."""
    try:
        from modelscope.hub.snapshot_download import dataset_snapshot_download
    except (ImportError, AttributeError):
        try:
            from modelscope import dataset_snapshot_download  # type: ignore
        except ImportError as exc:
            raise DatasetPreparationError(
                "CVC-14 automatic download uses the ModelScope mirror. Install with "
                "`pip install -e '.[data]'`, or pass --archive with a local CVC-14 directory/archive."
            ) from exc
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = dataset_snapshot_download(dataset_id=dataset_id, local_dir=str(destination))
    root = Path(downloaded).resolve() if downloaded else destination.resolve()
    if not root.exists() or not any(root.rglob("*")):
        raise DatasetPreparationError("ModelScope returned an empty CVC-14 snapshot.")
    return root


def _safe_tar_member(root: Path, name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise DatasetPreparationError(f"Unsafe path in TAR archive: {name}")
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise DatasetPreparationError(f"TAR member escapes extraction root: {name}") from exc
    return target


def safe_extract_tar(archive: str | Path, destination: str | Path, *, force: bool = False) -> Path:
    archive = Path(archive).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    marker = destination / f".{archive.name}.extract-complete.json"
    checksum = _sha256(archive)
    if marker.is_file() and not force:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("archive_sha256") == checksum:
            return destination
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, mode="r:*") as bundle:
            for member in bundle.getmembers():
                target = _safe_tar_member(destination, member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise DatasetPreparationError(f"Links/devices are not accepted in TAR datasets: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise DatasetPreparationError(f"Cannot read TAR member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=4 << 20)
    except tarfile.TarError as exc:
        raise DatasetPreparationError(f"Invalid TAR archive: {archive}") from exc
    marker.write_text(json.dumps({"archive_sha256": checksum, "completed_unix": time.time()}, indent=2) + "\n")
    return destination


def safe_extract_7z(archive: str | Path, destination: str | Path, *, force: bool = False) -> Path:
    py7zr = _require_optional("py7zr", "Install with: pip install -e '.[data]'")
    archive = Path(archive).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    marker = destination / f".{archive.name}.extract-complete.json"
    checksum = _sha256(archive)
    if marker.is_file() and not force:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("archive_sha256") == checksum:
            return destination
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with py7zr.SevenZipFile(archive, mode="r") as bundle:
            names = bundle.getnames()
            for name in names:
                _safe_tar_member(destination, name)
            bundle.extractall(path=destination)
    except Exception as exc:
        raise DatasetPreparationError(f"Invalid or unsafe 7z archive: {archive}") from exc
    marker.write_text(json.dumps({"archive_sha256": checksum, "completed_unix": time.time()}, indent=2) + "\n")
    return destination


def _extract_archive(archive: Path, destination: Path, *, force: bool = False) -> Path:
    name = archive.name.lower()
    if name.endswith(".zip"):
        return safe_extract_zip(archive, destination, force=force)
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
        return safe_extract_tar(archive, destination, force=force)
    if name.endswith(".7z"):
        return safe_extract_7z(archive, destination, force=force)
    raise DatasetPreparationError(f"Unsupported archive format: {archive.name}")


def _extract_all_archives(source: Path, destination: Path, *, force: bool = False) -> Path:
    """Extract a top-level archive and any nested archives, with a bounded recursion depth."""
    if source.is_dir():
        root = source.resolve()
    else:
        root = _extract_archive(source, destination, force=force)
    for depth in range(3):
        archives = sorted(
            p for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(
                (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".7z")
            )
        )
        new_count = 0
        for index, archive in enumerate(archives):
            target = archive.parent / f"_{archive.name.replace('.', '_')}_extracted"
            marker_candidates = list(target.glob(".*extract-complete.json")) if target.exists() else []
            if marker_candidates and not force:
                continue
            _extract_archive(archive, target, force=force)
            new_count += 1
        if new_count == 0:
            break
    return root


def _concat_parts(parts: Sequence[Path], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with partial.open("wb") as output:
        for part in parts:
            if not part.is_file():
                raise FileNotFoundError(part)
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 << 20)
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, destination)
    return destination


def _download_vedai(variant: str, downloads: Path, *, force: bool = False) -> Path:
    count = 2 if variant == "512" else 5
    prefix = f"Vehicules{variant}.tar"
    parts: list[Path] = []
    for index in range(1, count + 1):
        name = f"{prefix}.{index:03d}"
        path = downloads / name
        if force:
            path.unlink(missing_ok=True)
        if not path.is_file():
            download_with_resume(f"{VEDAI_BASE}/{name}", path, validate_magic=False)
        parts.append(path)
    annotations = downloads / f"Annotations{variant}.tar"
    if force:
        annotations.unlink(missing_ok=True)
    if not annotations.is_file():
        download_with_resume(f"{VEDAI_BASE}/Annotations{variant}.tar", annotations, validate_magic=False)
    combined = downloads / prefix
    if force or not combined.is_file():
        _concat_parts(parts, combined)
    raw = downloads.parent / "raw"
    safe_extract_tar(combined, raw, force=force)
    safe_extract_tar(annotations, raw, force=force)
    return raw


def _normal_stem(path: Path) -> str:
    stem = path.stem.lower()
    # common pair suffixes are deliberately removed only at token boundaries
    stem = re.sub(r"(?:[_-](?:visible|thermal|infrared|rgb|ir|fir|nir|vi|co|lwir))+$", "", stem)
    return stem


def _modality_score(path: Path, modality: str) -> int:
    tokens = set(re.split(r"[^a-z0-9]+", path.as_posix().lower()))
    visible = {"visible", "vis", "rgb", "vi", "color", "colour", "co"}
    infrared = {"thermal", "infrared", "ir", "fir", "nir", "lwir"}
    wanted = visible if modality == "rgb" else infrared
    rejected = infrared if modality == "rgb" else visible
    return 3 * len(tokens & wanted) - 3 * len(tokens & rejected)


def _pair_from_modality_dirs(root: Path) -> list[tuple[str, Path, Path]]:
    files = _image_files(root)
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        relative = path.relative_to(root)
        context = "/".join(relative.parts[:-1]).lower()
        key = f"{re.sub(r'(visible|thermal|infrared|rgb|ir|fir|nir|vi|color|colour)', '', context)}::{_normal_stem(path)}"
        key = re.sub(r"/+", "/", key)
        groups[key].append(path)
    pairs: list[tuple[str, Path, Path]] = []
    for key, candidates in sorted(groups.items()):
        if len(candidates) < 2:
            continue
        rgb = max(candidates, key=lambda p: (_modality_score(p, "rgb"), -len(p.as_posix())))
        ir = max(candidates, key=lambda p: (_modality_score(p, "ir"), -len(p.as_posix())))
        if rgb == ir or _modality_score(rgb, "rgb") <= 0 or _modality_score(ir, "ir") <= 0:
            continue
        pairs.append((_normal_stem(rgb), rgb, ir))
    return pairs


def _find_named_dir(root: Path, names: Sequence[str]) -> Path | None:
    names_l = {n.lower() for n in names}
    candidates = [p for p in [root, *root.rglob("*")] if p.is_dir() and p.name.lower() in names_l]
    return min(candidates, key=lambda p: len(p.parts)) if candidates else None


def _deterministic_split(key: str, val_fraction: float, seed: int) -> str:
    token = hashlib.sha1(f"{seed}:{key}".encode()).digest()
    value = int.from_bytes(token[:8], "big") / float(1 << 64)
    return "val" if value < val_fraction else "train"


def _limit_samples(samples: Sequence[PairSample], limit: int | None) -> list[PairSample]:
    result = sorted(samples, key=lambda s: (s.split, s.key))
    if limit is None:
        return result
    if limit <= 0:
        raise ValueError("limit must be positive")
    counters: dict[str, int] = defaultdict(int)
    selected: list[PairSample] = []
    for sample in result:
        if counters[sample.split] < limit:
            selected.append(sample)
            counters[sample.split] += 1
    return selected


def _read_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise DatasetPreparationError(f"Unreadable image: {path}")
    return int(image.shape[1]), int(image.shape[0])


def _clip_points(points: Iterable[tuple[float, float]], width: int, height: int) -> tuple[tuple[float, float], ...]:
    return tuple((max(0.0, min(float(width), x)), max(0.0, min(float(height), y))) for x, y in points)


def _hbb_points(x1: float, y1: float, x2: float, y2: float) -> tuple[tuple[float, float], ...]:
    x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.asarray([p[0] for p in points], dtype=np.float64)
    y = np.asarray([p[1] for p in points], dtype=np.float64)
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _write_yolo_label(path: Path, labels: Sequence[ObjectLabel], width: int, height: int, task: str) -> int:
    lines: list[str] = []
    for obj in labels:
        points = _clip_points(obj.points, width, height)
        if _polygon_area(points) < 1.0:
            continue
        if task == "obb":
            if len(points) != 4:
                rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
                points = tuple(tuple(map(float, p)) for p in cv2.boxPoints(rect))
            coords = [value for x, y in points for value in (x / width, y / height)]
        elif task == "detect":
            xs, ys = zip(*points)
            x1, x2 = min(xs), max(xs); y1, y2 = min(ys), max(ys)
            coords = [((x1 + x2) * 0.5) / width, ((y1 + y2) * 0.5) / height,
                      (x2 - x1) / width, (y2 - y1) / height]
        else:
            raise ValueError("task must be obb or detect")
        if all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in coords):
            lines.append(f"{obj.class_id} " + " ".join(f"{v:.8f}" for v in coords))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _parse_yolo_source(path: Path | None, width: int, height: int, class_count: int) -> tuple[ObjectLabel, ...]:
    if path is None or not path.is_file():
        return ()
    labels: list[ObjectLabel] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        fields = raw.strip().split()
        if not fields:
            continue
        try:
            values = [float(v) for v in fields]
        except ValueError as exc:
            raise DatasetPreparationError(f"Non-numeric YOLO label at {path}:{number}") from exc
        cls = int(values[0])
        if cls < 0 or cls >= class_count:
            raise DatasetPreparationError(f"Class {cls} outside [0,{class_count - 1}] at {path}:{number}")
        coords = values[1:]
        if len(coords) == 4:
            cx, cy, w, h = coords
            points = _hbb_points((cx - w / 2) * width, (cy - h / 2) * height,
                                 (cx + w / 2) * width, (cy + h / 2) * height)
        elif len(coords) == 8:
            points = tuple((coords[i] * width, coords[i + 1] * height) for i in range(0, 8, 2))
        else:
            raise DatasetPreparationError(f"Expected 5 or 9 YOLO fields at {path}:{number}")
        labels.append(ObjectLabel(cls, points))
    return tuple(labels)


def _read_split_manifest(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        keys.add(Path(text.split()[0]).stem.lower())
    return keys


def _coco_records(json_path: Path, class_names: Sequence[str]) -> dict[str, tuple[tuple[ObjectLabel, ...], int, int]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    categories = {int(c["id"]): str(c.get("name", c["id"])).lower() for c in payload.get("categories", [])}
    name_to_id = {name.lower(): i for i, name in enumerate(class_names)}
    category_map: dict[int, int] = {}
    for raw_id, name in categories.items():
        normalized = {"people": "person", "pedestrian": "person", "bike": "bicycle"}.get(name, name)
        if normalized in name_to_id:
            category_map[raw_id] = name_to_id[normalized]
    images = {int(item["id"]): item for item in payload.get("images", [])}
    grouped: dict[int, list[ObjectLabel]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        if ann.get("ignore") or ann.get("iscrowd", 0) not in (0, False):
            # Crowd is a semantic class in RGBTDronePerson, so only ignore if the category is not mapped.
            if int(ann.get("category_id", -1)) not in category_map:
                continue
        cid = category_map.get(int(ann.get("category_id", -1)))
        if cid is None:
            continue
        if "segmentation" in ann and isinstance(ann["segmentation"], list) and ann["segmentation"]:
            seg = ann["segmentation"][0]
            if isinstance(seg, list) and len(seg) >= 8:
                points = tuple((float(seg[i]), float(seg[i + 1])) for i in range(0, len(seg) - 1, 2))
            else:
                points = ()
        else:
            points = ()
        if not points:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = map(float, bbox)
            points = _hbb_points(x, y, x + w, y + h)
        grouped[int(ann["image_id"])].append(ObjectLabel(cid, points))
    records: dict[str, tuple[tuple[ObjectLabel, ...], int, int]] = {}
    for image_id, item in images.items():
        filename = str(item["file_name"]).replace("\\", "/")
        records[filename] = (tuple(grouped.get(image_id, [])), int(item.get("width", 0)), int(item.get("height", 0)))
        records[Path(filename).name] = records[filename]
        records[Path(filename).stem] = records[filename]
    return records


def _find_coco_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as stream:
                head = stream.read(4096)
            if '"images"' in head or '"annotations"' in head:
                files.append(path)
        except (OSError, UnicodeDecodeError):
            pass
    return sorted(files)


def _split_hint(path: Path) -> str | None:
    """Infer a source split without leaking unrelated absolute-path names.

    The filename may encode a split (for example ``train_thermal.json``).
    Parent directories are accepted only when the whole directory name is a
    standard split token, so a temporary directory such as
    ``test_public_converter0`` cannot force every sample into the test split.
    """
    filename_tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
    parent_names = {parent.name.lower() for parent in list(path.parents)[:4]}
    test_names = {"test", "testing", "test2017", "test2019"}
    val_names = {"val", "valid", "validation", "val2017", "val2019"}
    train_names = {"train", "training", "train2017", "train2019"}
    if filename_tokens & {"test", "testing"} or parent_names & test_names:
        return "test"
    if filename_tokens & {"val", "valid", "validation"} or parent_names & val_names:
        return "val"
    if filename_tokens & {"train", "training"} or parent_names & train_names:
        return "train"
    return None


def _prepare_m3fd(raw: Path, spec: PublicDatasetSpec, val_fraction: float, seed: int) -> list[PairSample]:
    roots = [p for p in [raw, *raw.rglob("*")] if p.is_dir() and (p / "vi").is_dir() and (p / "ir").is_dir()]
    if not roots:
        raise DatasetPreparationError("M3FD layout not found: expected sibling vi/, ir/, labels/, and meta/ directories.")
    root = min(roots, key=lambda p: len(p.parts))
    vi, ir = root / "vi", root / "ir"
    labels_dir, meta = root / "labels", root / "meta"
    train_keys = _read_split_manifest(meta / "train.txt")
    val_keys = _read_split_manifest(meta / "val.txt")
    samples: list[PairSample] = []
    for rgb in _image_files(vi):
        relative = rgb.relative_to(vi)
        partner = (ir / relative)
        if not partner.is_file():
            matches = [p for p in ir.rglob(rgb.stem + ".*") if p.suffix[1:].lower() in IMG_FORMATS]
            partner = matches[0] if matches else Path()
        if not partner.is_file():
            raise DatasetPreparationError(f"M3FD infrared partner missing for {relative}")
        width, height = _read_image_size(rgb)
        label = labels_dir / relative.with_suffix(".txt")
        labels = _parse_yolo_source(label if label.is_file() else None, width, height, len(spec.classes))
        key = rgb.stem.lower()
        if key in val_keys:
            split = "val"
        elif key in train_keys:
            split = "train"
        else:
            split = _deterministic_split(key, val_fraction, seed)
        samples.append(PairSample(relative.with_suffix("").as_posix(), rgb, partner, labels, split))
    return samples


def _parse_vedai_labels(path: Path, width: int, height: int, class_count: int) -> tuple[ObjectLabel, ...]:
    labels: list[ObjectLabel] = []
    if not path.is_file():
        return ()
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 14:
            raise DatasetPreparationError(f"VEDAI expects 14 fields at {path}:{number}, got {len(fields)}")
        values = [float(v) for v in fields]
        raw_class = int(values[3])
        cls = raw_class - 1 if 1 <= raw_class <= class_count else raw_class
        if cls < 0 or cls >= class_count:
            raise DatasetPreparationError(f"Unknown VEDAI class id {raw_class} at {path}:{number}")
        # Official records place x1..x4 first, followed by y1..y4.
        xs, ys = values[6:10], values[10:14]
        points = tuple(zip(xs, ys))
        if _polygon_area(points) < 1.0:
            # Some third-party conversions interleave x/y; accept it only when the official interpretation is invalid.
            alternate = tuple((values[i], values[i + 1]) for i in range(6, 14, 2))
            if _polygon_area(alternate) > _polygon_area(points):
                points = alternate
        labels.append(ObjectLabel(cls, tuple((float(x), float(y)) for x, y in points)))
    return tuple(labels)


def _split_multiband_vedai(image: Path, cache_root: Path) -> tuple[Path, Path] | None:
    data = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    if data is None or data.ndim != 3 or data.shape[2] < 4:
        return None
    rgb_path = cache_root / "rgb" / f"{image.stem}.png"
    ir_path = cache_root / "ir" / f"{image.stem}.png"
    rgb_path.parent.mkdir(parents=True, exist_ok=True); ir_path.parent.mkdir(parents=True, exist_ok=True)
    if not rgb_path.is_file():
        cv2.imwrite(str(rgb_path), data[:, :, :3])
    if not ir_path.is_file():
        nir = data[:, :, 3]
        cv2.imwrite(str(ir_path), cv2.cvtColor(nir, cv2.COLOR_GRAY2BGR))
    return rgb_path, ir_path


def _prepare_vedai(raw: Path, spec: PublicDatasetSpec, variant: str, val_fraction: float, seed: int) -> list[PairSample]:
    ann_dirs = [p for p in raw.rglob("*") if p.is_dir() and p.name.lower() == f"annotations{variant}".lower()]
    if not ann_dirs:
        ann_dirs = [p for p in raw.rglob("*") if p.is_dir() and "annotation" in p.name.lower()]
    if not ann_dirs:
        raise DatasetPreparationError(f"VEDAI annotations{variant}/ directory not found.")
    ann_dir = min(ann_dirs, key=lambda p: len(p.parts))
    images = [p for p in _image_files(raw) if "annotation" not in p.as_posix().lower()]
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        by_stem[_normal_stem(image)].append(image)
    split_cache = raw / "_vedai_split_modalities"
    samples: list[PairSample] = []
    for ann in sorted(ann_dir.glob("*.txt")):
        candidates = by_stem.get(ann.stem.lower(), [])
        rgb: Path | None = None; ir: Path | None = None
        if len(candidates) >= 2:
            rgb = max(candidates, key=lambda p: _modality_score(p, "rgb"))
            ir = max(candidates, key=lambda p: _modality_score(p, "ir"))
            if rgb == ir:
                rgb = ir = None
        if rgb is None and candidates:
            split = _split_multiband_vedai(candidates[0], split_cache)
            if split:
                rgb, ir = split
        if rgb is None or ir is None:
            # Some releases have one color image only. VEDAI's extra spectral band may be embedded; refuse silent duplication.
            raise DatasetPreparationError(
                f"Could not find distinct RGB/NIR data for VEDAI sample {ann.stem}. "
                "Supply the official multispectral archive, not a color-only derivative."
            )
        width, height = _read_image_size(rgb)
        labels = _parse_vedai_labels(ann, width, height, len(spec.classes))
        split_name = _split_hint(rgb) or _deterministic_split(ann.stem, val_fraction, seed)
        if split_name == "test":
            split_name = "val"  # labels are available; keep a train/val setup for the project
        samples.append(PairSample(ann.stem, rgb, ir, labels, split_name))
    return samples


def _match_pair_file(root: Path, reference: Path, wanted: str) -> Path | None:
    stem = _normal_stem(reference)
    candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS
                  and _normal_stem(p) == stem and p != reference]
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda p: (_modality_score(p, wanted), -len(p.parts)), reverse=True)
    return ranked[0] if _modality_score(ranked[0], wanted) > 0 else None


def _prepare_coco_paired(raw: Path, spec: PublicDatasetSpec, val_fraction: float, seed: int,
                         reference_modality: str = "ir") -> list[PairSample]:
    coco_files = _find_coco_files(raw)
    if not coco_files:
        raise DatasetPreparationError(f"No COCO annotation JSON found for {spec.key}.")
    all_images = _image_files(raw)
    image_index: dict[str, list[Path]] = defaultdict(list)
    for image in all_images:
        image_index[image.name.lower()].append(image)
        image_index[image.stem.lower()].append(image)
        image_index[image.relative_to(raw).as_posix().lower()].append(image)
    samples: dict[str, PairSample] = {}
    for json_path in coco_files:
        records = _coco_records(json_path, spec.classes)
        split_hint = _split_hint(json_path)
        for name, (labels, _, _) in records.items():
            if "/" not in name and "." not in name:
                continue  # stem aliases are handled by filename records
            candidates = image_index.get(name.lower(), []) or image_index.get(Path(name).name.lower(), [])
            if not candidates:
                continue
            reference = max(candidates, key=lambda p: _modality_score(p, reference_modality))
            partner = _match_pair_file(raw, reference, "rgb" if reference_modality == "ir" else "ir")
            if partner is None:
                continue
            rgb, ir = (partner, reference) if reference_modality == "ir" else (reference, partner)
            key = f"{split_hint or 'auto'}/{_normal_stem(reference)}"
            split = split_hint or _split_hint(reference) or _deterministic_split(key, val_fraction, seed)
            samples[key] = PairSample(key, rgb, ir, labels, split)
    if not samples:
        raise DatasetPreparationError(
            f"COCO files were found for {spec.key}, but no visible/infrared image pairs matched their file names."
        )
    # Guarantee a validation split when a release only provides train/test or train.
    values = list(samples.values())
    if not any(s.split == "val" for s in values):
        values = [PairSample(s.key, s.rgb, s.ir, s.labels,
                             _deterministic_split(s.key, val_fraction, seed) if s.split == "train" else s.split)
                  for s in values]
    return values


def _parse_simple_text_boxes(path: Path | None, width: int, height: int) -> tuple[ObjectLabel, ...]:
    if path is None or not path.is_file():
        return ()
    labels: list[ObjectLabel] = []
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        fields = re.findall(r"[-+]?\d*\.?\d+", raw)
        if len(fields) < 4:
            continue
        values = [float(v) for v in fields]
        # CVC derivatives commonly use x y w h; choose XYXY only when it is unambiguous.
        x, y, a, b = values[:4]
        if a > x and b > y and a <= width and b <= height and (a - x) * (b - y) > a * b * 0.05:
            x1, y1, x2, y2 = x, y, a, b
        else:
            x1, y1, x2, y2 = x, y, x + a, y + b
        labels.append(ObjectLabel(0, _hbb_points(x1, y1, x2, y2)))
    return tuple(labels)


def _find_annotation_for_image(root: Path, image: Path) -> Path | None:
    stem = image.stem
    candidates = [p for p in root.rglob(stem + ".*") if p.is_file() and p.suffix.lower() in {".txt", ".xml"}]
    candidates = [p for p in candidates if "annotation" in p.as_posix().lower() or "label" in p.as_posix().lower()]
    return min(candidates, key=lambda p: len(p.parts)) if candidates else None


def _parse_voc(path: Path, class_names: Sequence[str]) -> tuple[ObjectLabel, ...]:
    tree = ET.parse(path)
    name_to_id = {n.lower(): i for i, n in enumerate(class_names)}
    labels: list[ObjectLabel] = []
    for obj in tree.findall(".//object"):
        name = (obj.findtext("name") or "person").strip().lower()
        name = {"pedestrian": "person", "people": "person"}.get(name, name)
        if name not in name_to_id:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        values = [float(box.findtext(tag, "0")) for tag in ("xmin", "ymin", "xmax", "ymax")]
        labels.append(ObjectLabel(name_to_id[name], _hbb_points(*values)))
    return tuple(labels)


def _prepare_cvc(raw: Path, spec: PublicDatasetSpec, val_fraction: float, seed: int) -> list[PairSample]:
    pairs = _pair_from_modality_dirs(raw)
    if not pairs:
        raise DatasetPreparationError(
            "CVC-14 visible/FIR pairs were not found. Expected modality directories named Visible and FIR/Thermal."
        )
    samples: list[PairSample] = []
    for key, rgb, ir in pairs:
        width, height = _read_image_size(rgb)
        annotation = _find_annotation_for_image(raw, rgb) or _find_annotation_for_image(raw, ir)
        if annotation and annotation.suffix.lower() == ".xml":
            labels = _parse_voc(annotation, spec.classes)
        else:
            labels = _parse_simple_text_boxes(annotation, width, height)
        split = _split_hint(rgb) or _split_hint(ir) or _deterministic_split(key, val_fraction, seed)
        if split == "test":
            split = "val"
        samples.append(PairSample(key, rgb, ir, labels, split))
    return samples


def _prepare_flir(raw: Path, spec: PublicDatasetSpec, val_fraction: float, seed: int) -> list[PairSample]:
    try:
        return _prepare_coco_paired(raw, spec, val_fraction, seed, reference_modality="ir")
    except DatasetPreparationError as coco_error:
        # A common aligned release uses YOLO labels with paired visible/thermal folders.
        pairs = _pair_from_modality_dirs(raw)
        if not pairs:
            raise coco_error
        samples: list[PairSample] = []
        for key, rgb, ir in pairs:
            width, height = _read_image_size(rgb)
            label = _find_annotation_for_image(raw, ir) or _find_annotation_for_image(raw, rgb)
            labels = _parse_yolo_source(label, width, height, len(spec.classes)) if label else ()
            split = _split_hint(rgb) or _deterministic_split(key, val_fraction, seed)
            if split == "test": split = "val"
            samples.append(PairSample(key, rgb, ir, labels, split))
        return samples


def _prepare_rgbtdrone(raw: Path, spec: PublicDatasetSpec, val_fraction: float, seed: int) -> list[PairSample]:
    return _prepare_coco_paired(raw, spec, val_fraction, seed, reference_modality="ir")


def _canonicalize(samples: Sequence[PairSample], output_root: Path, spec: PublicDatasetSpec, variant: str,
                  task: str, link_mode: str, limit: int | None, source_info: dict[str, Any],
                  force: bool = False) -> Path:
    processed = output_root / "processed" / variant
    marker = processed / ".preprocess-complete.json"
    if marker.is_file() and not force:
        data_yaml = processed / "data.yaml"
        if data_yaml.is_file():
            return data_yaml
    if force and processed.exists():
        shutil.rmtree(processed)
    selected = _limit_samples(samples, limit)
    split_counts: dict[str, int] = defaultdict(int)
    objects = 0
    link_methods: dict[str, int] = defaultdict(int)
    seen_targets: set[str] = set()
    for sample in selected:
        split = sample.split if sample.split in {"train", "val", "test"} else "train"
        safe_key = re.sub(r"[^A-Za-z0-9_./-]+", "_", sample.key).strip("/.") or sample.rgb.stem
        rgb_suffix = sample.rgb.suffix.lower() or ".png"
        ir_suffix = sample.ir.suffix.lower() or ".png"
        # The paired loader resolves the IR path from the RGB relative path. Keep
        # both canonical filenames identical. If source encodings differ, convert
        # both to PNG instead of copying bytes under a misleading extension.
        common_suffix = rgb_suffix if rgb_suffix == ir_suffix else ".png"
        relative = Path(safe_key).with_suffix(common_suffix)
        target_id = f"{split}/{relative.as_posix().lower()}"
        if target_id in seen_targets:
            digest = hashlib.sha1(str(sample.rgb).encode()).hexdigest()[:8]
            relative = relative.with_name(f"{relative.stem}_{digest}{relative.suffix}")
            target_id = f"{split}/{relative.as_posix().lower()}"
        seen_targets.add(target_id)
        rgb_target = processed / "rgb/images" / split / relative
        ir_target = processed / "ir/images" / split / relative
        if rgb_suffix == ir_suffix:
            for source, target in ((sample.rgb, rgb_target), (sample.ir, ir_target)):
                method = _link_or_copy(source, target, link_mode)
                link_methods[method] += 1
        else:
            for source, target in ((sample.rgb, rgb_target), (sample.ir, ir_target)):
                image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise DatasetPreparationError(f"Cannot decode image for canonical conversion: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), image):
                    raise DatasetPreparationError(f"Cannot write canonical PNG: {target}")
                link_methods["converted"] += 1
        rgb_size = _read_image_size(sample.rgb); ir_size = _read_image_size(sample.ir)
        if rgb_size != ir_size:
            raise DatasetPreparationError(
                f"Pair dimensions differ for {sample.key}: RGB={rgb_size}, IR={ir_size}. "
                "Use an aligned release or explicitly preprocess registration before training."
            )
        label_target = processed / "labels" / split / relative.with_suffix(".txt")
        objects += _write_yolo_label(label_target, sample.labels, rgb_size[0], rgb_size[1], task)
        split_counts[split] += 1
    if split_counts.get("train", 0) == 0 or split_counts.get("val", 0) == 0:
        raise DatasetPreparationError(f"Prepared {spec.key} requires non-empty train and val splits; got {dict(split_counts)}")
    config: dict[str, Any] = {
        "path": str(processed),
        "train": "rgb/images/train",
        "val": "rgb/images/val",
        "train_ir": "ir/images/train",
        "val_ir": "ir/images/val",
        "train_labels": "labels/train",
        "val_labels": "labels/val",
        "pairing": {"strict": True, "resize_ir": False, "cache_suffix": ".rgbir.npy"},
        "names": {i: name for i, name in enumerate(spec.classes)},
        "source": {
            "dataset": spec.title,
            "homepage": spec.homepage,
            "citation_doi": spec.citation_doi,
            "license": spec.license_name,
            "variant": variant,
            "task": task,
        },
    }
    if split_counts.get("test"):
        config.update({"test": "rgb/images/test", "test_ir": "ir/images/test", "test_labels": "labels/test"})
    data_yaml = processed / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "prepared_by": f"etfnet-sjpa-tr {__version__}",
        "prepared_unix": time.time(),
        "dataset": asdict(spec),
        "variant": variant,
        "task": task,
        "limit_per_split": limit,
        "source": source_info,
        "statistics": {"pairs": sum(split_counts.values()), "objects": objects,
                       "splits": dict(split_counts), "link_methods": dict(link_methods)},
        "data_yaml": str(data_yaml),
    }
    (processed / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    marker.write_text(json.dumps({"schema_version": 2, "data_yaml": str(data_yaml),
                                  "completed_unix": time.time()}, indent=2) + "\n")
    return data_yaml


def _download_source(spec: PublicDatasetSpec, root: Path, variant: str, force: bool) -> Path:
    downloads = root / "downloads"; downloads.mkdir(parents=True, exist_ok=True)
    if spec.key == "m3fd":
        target = downloads / "m3fd"
        if force and target.exists(): shutil.rmtree(target)
        return _download_gdrive_folder(M3FD_DRIVE_FOLDER, target) if not any(target.rglob("*")) else target
    if spec.key == "rgbtdroneperson":
        target = downloads / "rgbtdroneperson"
        if force and target.exists(): shutil.rmtree(target)
        return _download_gdrive_folder(RGBT_DRIVE_FOLDER, target) if not any(target.rglob("*")) else target
    if spec.key == "vedai":
        return _download_vedai(variant, downloads, force=force)
    if spec.key == "flir-aligned":
        target = downloads / "aligned.zip"
        if force: target.unlink(missing_ok=True)
        if not target.is_file():
            try:
                _download_gdrive_file(FLIR_DRIVE_FILE, target)
            except DatasetPreparationError:
                download_with_resume(FLIR_FALLBACK_URL, target)
        return target
    if spec.key == "cvc-14":
        target = downloads / "cvc14"
        if force and target.exists(): shutil.rmtree(target)
        return _download_modelscope(CVC_MODELSCOPE_ID, target) if not any(target.rglob("*")) else target
    raise KeyError(spec.key)


def prepare_additional_dataset(
    name: str,
    output_dir: str | Path = "datasets",
    *,
    variant: str | None = None,
    task: str = "obb",
    accept_license: bool = False,
    archive: str | Path | None = None,
    force_download: bool = False,
    force_extract: bool = False,
    force_preprocess: bool = False,
    keep_archive: bool = True,
    link_mode: str = "auto",
    visibility: str = "all",
    exclude_bad: bool = False,
    limit: int | None = None,
    val_fraction: float = 0.2,
    split_seed: int = 0,
    reference_modality: str = "ir",
) -> Path:
    del visibility, exclude_bad, reference_modality  # options retained for a uniform public-data API
    key = name.strip().lower()
    if key not in ADDITIONAL_DATASETS:
        raise KeyError(key)
    spec = ADDITIONAL_DATASETS[key]
    variant = variant or spec.default_variant
    if variant not in spec.variants:
        raise ValueError(f"Variant '{variant}' is invalid for {key}; choose one of {spec.variants}")
    if not 0.01 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be between 0.01 and 0.5")
    root = Path(output_dir).expanduser().resolve() / key
    existing = root / "processed" / variant / "data.yaml"
    if existing.is_file() and not force_preprocess and archive is None:
        return existing
    if not accept_license:
        raise PermissionError(
            f"{spec.title} uses {spec.license_name}. Review {spec.license_url}, then re-run with --accept-license."
        )
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".prepare.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, f"pid={os.getpid()} time={time.time()}\n".encode()); os.close(descriptor)
    except FileExistsError as exc:
        if time.time() - lock.stat().st_mtime > 24 * 3600:
            lock.unlink()
            return prepare_additional_dataset(
                name, output_dir, variant=variant, task=task, accept_license=accept_license, archive=archive,
                force_download=force_download, force_extract=force_extract, force_preprocess=force_preprocess,
                keep_archive=keep_archive, link_mode=link_mode, limit=limit, val_fraction=val_fraction,
                split_seed=split_seed,
            )
        raise DatasetPreparationError(f"Another preparation process holds {lock}") from exc
    try:
        source = Path(archive).expanduser().resolve() if archive else _download_source(spec, root, variant, force_download)
        if not source.exists():
            raise FileNotFoundError(source)
        raw = root / "raw"
        extracted = _extract_all_archives(source, raw, force=force_extract)
        source_info = {"path": str(source), "kind": "directory" if source.is_dir() else "archive"}
        if source.is_file():
            source_info.update({"bytes": source.stat().st_size, "sha256": _sha256(source)})
        if key == "m3fd":
            samples = _prepare_m3fd(extracted, spec, val_fraction, split_seed)
        elif key == "vedai":
            samples = _prepare_vedai(extracted, spec, variant, val_fraction, split_seed)
        elif key == "flir-aligned":
            samples = _prepare_flir(extracted, spec, val_fraction, split_seed)
        elif key == "rgbtdroneperson":
            samples = _prepare_rgbtdrone(extracted, spec, val_fraction, split_seed)
        else:
            samples = _prepare_cvc(extracted, spec, val_fraction, split_seed)
        data_yaml = _canonicalize(samples, root, spec, variant, task, link_mode, limit, source_info,
                                  force=force_preprocess)
        if not keep_archive and archive is None and source.is_file():
            source.unlink(missing_ok=True)
        return data_yaml
    finally:
        lock.unlink(missing_ok=True)
