"""Public RGB--IR dataset acquisition and reproducible preprocessing.

The default supported benchmark is NII-CU MAPD, a public paired RGB/FIR
UAV person-detection dataset associated with DOI 10.1002/rob.22082.
The downloader uses the official dataset link, validates the ZIP archive,
and converts the tab-separated boxes into YOLO detect or OBB labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2
import requests
import yaml
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from ultralytics import __version__
from ultralytics.data.utils import IMG_FORMATS


@dataclass(frozen=True)
class PublicDatasetSpec:
    """Immutable metadata for one supported public dataset."""

    key: str
    title: str
    description: str
    homepage: str
    citation_doi: str
    dataset_doi: str | None
    license_name: str
    license_url: str
    archive_name: str
    download_url: str
    expected_download_gb: float
    variants: tuple[str, ...]
    default_variant: str
    classes: tuple[str, ...]


# Official source page: https://www.nii-cu-multispectral.org/
# The `preview` argument selects the labelled 9.1 GB archive from the official
# Dropbox folder rather than the separate 15.2 GB raw-video archive.
_NII_CU_DOWNLOAD = (
    "https://www.dropbox.com/scl/fo/g1unaotzqxs71o978236n/"
    "ADoTYur1wKZRvaiwLAw48II?rlkey=bw9zyqlvlco73a8cmoxaoimck"
    "&preview=NII_CU_MAPD_dataset.zip&dl=1"
)

PUBLIC_DATASETS: dict[str, PublicDatasetSpec] = {
    "nii-cu-mapd": PublicDatasetSpec(
        key="nii-cu-mapd",
        title="NII-CU Multispectral Aerial Person Detection Dataset",
        description="5,880 aligned drone RGB/FIR pairs with person bounding boxes.",
        homepage="https://www.nii-cu-multispectral.org/",
        citation_doi="10.1002/rob.22082",
        dataset_doi=None,
        license_name="CC BY-NC-SA 3.0",
        license_url="https://creativecommons.org/licenses/by-nc-sa/3.0/",
        archive_name="NII_CU_MAPD_dataset.zip",
        download_url=_NII_CU_DOWNLOAD,
        expected_download_gb=9.1,
        variants=("4-channel", "rgb-t"),
        default_variant="4-channel",
        classes=("person",),
    ),
}


class DatasetPreparationError(RuntimeError):
    """Raised when public-dataset acquisition or conversion cannot be completed safely."""


def _dataset_registry() -> dict[str, PublicDatasetSpec]:
    """Build the registry lazily to avoid a circular import with converters."""
    registry = dict(PUBLIC_DATASETS)
    try:
        from ultralytics.data.public_multidataset import ADDITIONAL_DATASETS
    except ImportError:
        # public_multidataset imports shared safe-I/O helpers from this module; while
        # that module is being initialized the NII-CU entry is the only safe view.
        return registry
    registry.update(ADDITIONAL_DATASETS)
    return registry


def list_public_datasets() -> list[dict[str, Any]]:
    """Return serializable metadata for every supported dataset."""
    return [asdict(spec) for spec in _dataset_registry().values()]


def get_public_dataset(name: str) -> PublicDatasetSpec:
    """Resolve a public dataset name and provide a useful error for unknown names."""
    key = name.strip().lower()
    registry = _dataset_registry()
    if key not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown public dataset '{name}'. Available: {available}")
    return registry[key]


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _requests_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET", "HEAD")),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": f"ETFNet-SJPA-TR/{__version__} public-dataset-downloader"})
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _with_download_flag(url: str) -> str:
    """Ensure a Dropbox-style URL requests binary download rather than HTML preview."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["dl"] = "1"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def download_with_resume(
    url: str,
    destination: str | Path,
    *,
    expected_size: int | None = None,
    timeout: tuple[float, float] = (20.0, 120.0),
    chunk_size: int = 4 << 20,
    session: requests.Session | None = None,
    validate_magic: bool = True,
) -> Path:
    """Download a file atomically with HTTP range resume and response validation.

    Existing ``.part`` bytes are reused only if the server accepts a Range
    request. A server that ignores Range causes a safe restart from byte zero.
    """
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    session = session or _requests_session()
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}

    response = session.get(_with_download_flag(url), headers=headers, stream=True, timeout=timeout,
                           allow_redirects=True)
    response.raise_for_status()
    if offset and response.status_code != 206:
        # Provider ignored Range. Restart instead of appending duplicate bytes.
        response.close()
        partial.unlink(missing_ok=True)
        offset = 0
        response = session.get(_with_download_flag(url), stream=True, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    disposition = response.headers.get("content-disposition", "").lower()
    if "text/html" in content_type and "attachment" not in disposition:
        response.close()
        raise DatasetPreparationError(
            "The dataset provider returned an HTML page instead of the archive. "
            "The public link may have changed; use --archive with a manually downloaded official ZIP."
        )

    remaining = response.headers.get("content-length")
    total = offset + int(remaining) if remaining and remaining.isdigit() else expected_size
    mode = "ab" if offset and response.status_code == 206 else "wb"
    with partial.open(mode) as output, tqdm(
        total=total,
        initial=offset,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Downloading {destination.name}",
    ) as progress:
        for block in response.iter_content(chunk_size=chunk_size):
            if block:
                output.write(block)
                progress.update(len(block))
        output.flush()
        os.fsync(output.fileno())
    response.close()

    size = partial.stat().st_size
    if expected_size is not None and size != expected_size:
        raise DatasetPreparationError(f"Downloaded size mismatch: expected {expected_size}, got {size}")
    if validate_magic:
        with partial.open("rb") as check_stream:
            magic = check_stream.read(4)
        if size < 4 or magic != b"PK\x03\x04":
            raise DatasetPreparationError(f"Downloaded file is not a ZIP archive: {partial}")
    os.replace(partial, destination)
    return destination


def _safe_member_path(root: Path, name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise DatasetPreparationError(f"Unsafe path in ZIP archive: {name}")
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise DatasetPreparationError(f"ZIP member escapes extraction root: {name}") from exc
    return target


def safe_extract_zip(archive: str | Path, destination: str | Path, *, force: bool = False) -> Path:
    """Validate CRCs and extract a ZIP without path traversal or archive symlinks."""
    archive = Path(archive).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    marker = destination / ".extract-complete.json"
    if marker.is_file() and not force:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("archive_sha256") == _sha256(archive):
            return destination
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive) as bundle:
            corrupt = bundle.testzip()
            if corrupt:
                raise DatasetPreparationError(f"ZIP CRC validation failed at member: {corrupt}")
            for info in bundle.infolist():
                target = _safe_member_path(destination, info.filename)
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(unix_mode):
                    raise DatasetPreparationError(f"Symbolic links are not accepted in dataset ZIPs: {info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=4 << 20)
    except zipfile.BadZipFile as exc:
        raise DatasetPreparationError(f"Invalid ZIP archive: {archive}") from exc

    marker.write_text(json.dumps({
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "completed_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    return destination


def _find_variant_root(extracted: Path, variant: str) -> Path | None:
    candidates: list[Path] = []
    for path in [extracted, *extracted.rglob(variant)]:
        if path.is_dir() and (path / "images" / "rgb").is_dir() and (path / "labels").is_dir():
            candidates.append(path)
    if not candidates:
        for path in extracted.rglob("images"):
            parent = path.parent
            if path.is_dir() and (path / "rgb").is_dir() and (parent / "labels").is_dir():
                if variant.lower() in parent.as_posix().lower():
                    candidates.append(parent)
    return min(candidates, key=lambda p: len(p.parts)) if candidates else None


def _unpack_nested_dataset_archive(extracted: Path, *, force: bool = False) -> Path:
    """Handle providers that wrap the desired ZIP inside a folder-download ZIP."""
    nested = sorted(extracted.rglob("NII_CU_MAPD_dataset.zip"))
    if not nested:
        return extracted
    nested_root = extracted / "_dataset_archive"
    return safe_extract_zip(nested[0], nested_root, force=force)


def _image_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS),
        key=lambda p: p.relative_to(root).as_posix(),
    )


def _find_partner(root: Path, relative: Path) -> Path | None:
    exact = root / relative
    if exact.is_file():
        return exact
    parent = root / relative.parent
    if not parent.is_dir():
        return None
    matches = [p for p in parent.glob(relative.stem + ".*") if p.suffix[1:].lower() in IMG_FORMATS]
    return sorted(matches)[0] if matches else None


def _find_label(root: Path, relative: Path) -> Path | None:
    parent = root / relative.parent
    for suffix in (".txt", ".csv", ".tsv"):
        candidate = parent / f"{relative.stem}{suffix}"
        if candidate.is_file():
            return candidate
    if parent.is_dir():
        matches = sorted(p for p in parent.glob(relative.stem + ".*") if p.is_file())
        return matches[0] if matches else None
    return None


def _link_or_copy(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == source.stat().st_size:
            return "existing"
        target.unlink()
    attempts = ("hardlink", "symlink", "copy") if mode == "auto" else (mode,)
    last_error: Exception | None = None
    for candidate in attempts:
        try:
            if candidate == "hardlink":
                os.link(source, target)
            elif candidate == "symlink":
                target.symlink_to(source.resolve())
            elif candidate == "copy":
                shutil.copy2(source, target)
            else:
                raise ValueError("link_mode must be auto, hardlink, symlink, or copy")
            return candidate
        except (OSError, ValueError) as exc:
            last_error = exc
            target.unlink(missing_ok=True)
    raise DatasetPreparationError(f"Cannot place {source} at {target}: {last_error}")


def _parse_annotation_rows(path: Path) -> Iterable[tuple[float, float, float, float, int, int, int]]:
    """Yield NII-CU rows as x1,y1,x2,y2,type,occluded,bad."""
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        fields = [field for field in re.split(r"[\t,; ]+", text) if field]
        try:
            numeric = [float(field) for field in fields]
        except ValueError:
            if line_number == 1 and any(token.lower().startswith("x") for token in fields):
                continue
            raise DatasetPreparationError(f"Non-numeric annotation at {path}:{line_number}: {raw}")
        if len(numeric) < 4:
            raise DatasetPreparationError(f"Expected at least 4 fields at {path}:{line_number}, got {len(numeric)}")
        x1, y1, x2, y2 = numeric[:4]
        visibility = int(numeric[4]) if len(numeric) > 4 else 0
        occluded = int(numeric[5]) if len(numeric) > 5 else 0
        bad = int(numeric[6]) if len(numeric) > 6 else 0
        yield x1, y1, x2, y2, visibility, occluded, bad


def _convert_label(
    source: Path | None,
    destination: Path,
    width: int,
    height: int,
    *,
    task: str,
    visibility: str,
    exclude_bad: bool,
) -> dict[str, int]:
    stats = {"rows": 0, "written": 0, "clipped": 0, "invalid": 0, "filtered": 0, "occluded": 0}
    lines: list[str] = []
    if source is not None:
        visibility_values = {"all": {0, 1, 2}, "both": {0}, "thermal": {0, 1}, "rgb": {0, 2}}[visibility]
        for x1, y1, x2, y2, visible_type, occluded, bad in _parse_annotation_rows(source):
            stats["rows"] += 1
            if visible_type not in visibility_values or (exclude_bad and bad):
                stats["filtered"] += 1
                continue
            original = (x1, y1, x2, y2)
            x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
            y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
            if original != (x1, y1, x2, y2):
                stats["clipped"] += 1
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                stats["invalid"] += 1
                continue
            if task == "obb":
                values = (x1 / width, y1 / height, x2 / width, y1 / height,
                          x2 / width, y2 / height, x1 / width, y2 / height)
            elif task == "detect":
                values = (((x1 + x2) * 0.5) / width, ((y1 + y2) * 0.5) / height,
                          (x2 - x1) / width, (y2 - y1) / height)
            else:
                raise ValueError("task must be obb or detect")
            lines.append("0 " + " ".join(f"{value:.8f}" for value in values))
            stats["written"] += 1
            stats["occluded"] += int(bool(occluded))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return stats


def preprocess_nii_cu_mapd(
    extracted_root: str | Path,
    output_root: str | Path,
    *,
    variant: str = "4-channel",
    task: str = "obb",
    link_mode: str = "auto",
    visibility: str = "all",
    exclude_bad: bool = False,
    limit: int | None = None,
    force: bool = False,
    archive_metadata: dict[str, Any] | None = None,
) -> Path:
    """Convert an extracted NII-CU archive into the canonical paired pipeline layout."""
    spec = PUBLIC_DATASETS["nii-cu-mapd"]
    if variant not in spec.variants:
        raise ValueError(f"variant must be one of {spec.variants}")
    if visibility not in {"all", "both", "thermal", "rgb"}:
        raise ValueError("visibility must be all, both, thermal, or rgb")
    extracted = Path(extracted_root).expanduser().resolve()
    dataset_root = Path(output_root).expanduser().resolve()
    processed = dataset_root / "processed" / variant
    marker = processed / ".preprocess-complete.json"
    if marker.is_file() and not force:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        yaml_path = processed / "data.yaml"
        if saved.get("schema_version") == 1 and yaml_path.is_file():
            return yaml_path
    if force and processed.exists():
        shutil.rmtree(processed)

    variant_root = _find_variant_root(extracted, variant)
    if variant_root is None:
        nested_root = _unpack_nested_dataset_archive(extracted, force=force)
        variant_root = _find_variant_root(nested_root, variant)
    if variant_root is None:
        raise DatasetPreparationError(
            f"Could not locate '{variant}/images/rgb' and '{variant}/labels' below {extracted}. "
            "Confirm that the official NII_CU_MAPD_dataset.zip was supplied."
        )

    totals: dict[str, Any] = {"pairs": 0, "objects": 0, "splits": {}, "link_methods": {}}
    for split in ("train", "val"):
        rgb_source = variant_root / "images" / "rgb" / split
        ir_source = variant_root / "images" / "thermal" / split
        if not ir_source.is_dir():
            ir_source = variant_root / "images" / "ir" / split
        label_source = variant_root / "labels" / split
        if not rgb_source.is_dir() or not ir_source.is_dir() or not label_source.is_dir():
            raise DatasetPreparationError(
                f"Incomplete split '{split}' below {variant_root}: expected RGB, thermal, and labels directories."
            )
        rgb_files = _image_files(rgb_source)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            rgb_files = rgb_files[:limit]
        split_stats: dict[str, Any] = {"pairs": 0, "objects": 0, "missing_ir": [], "missing_labels": [],
                                      "label_stats": {}}
        aggregate = {"rows": 0, "written": 0, "clipped": 0, "invalid": 0, "filtered": 0, "occluded": 0}
        for rgb_path in rgb_files:
            relative = rgb_path.relative_to(rgb_source)
            ir_path = _find_partner(ir_source, relative)
            label_path = _find_label(label_source, relative)
            if ir_path is None:
                split_stats["missing_ir"].append(relative.as_posix())
                continue
            if label_path is None:
                # Empty annotations are valid, but a truly missing file is recorded for auditability.
                split_stats["missing_labels"].append(relative.as_posix())
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
            if rgb is None or ir is None:
                raise DatasetPreparationError(f"Unreadable image pair: {rgb_path} | {ir_path}")
            if rgb.shape[:2] != ir.shape[:2]:
                raise DatasetPreparationError(
                    f"Official aligned pair has inconsistent dimensions: RGB {rgb.shape[:2]} vs IR {ir.shape[:2]} "
                    f"for {relative}. Use the 4-channel variant or inspect the source archive."
                )

            rgb_target = processed / "rgb" / "images" / split / relative
            ir_target = processed / "ir" / "images" / split / relative
            label_target = processed / "labels" / split / relative.with_suffix(".txt")
            for source, target in ((rgb_path, rgb_target), (ir_path, ir_target)):
                used = _link_or_copy(source, target, link_mode)
                totals["link_methods"][used] = totals["link_methods"].get(used, 0) + 1
            label_stats = _convert_label(label_path, label_target, rgb.shape[1], rgb.shape[0], task=task,
                                         visibility=visibility, exclude_bad=exclude_bad)
            for key, value in label_stats.items():
                aggregate[key] += value
            split_stats["pairs"] += 1
            split_stats["objects"] += label_stats["written"]

        if split_stats["missing_ir"]:
            preview = ", ".join(split_stats["missing_ir"][:5])
            raise DatasetPreparationError(f"Missing {len(split_stats['missing_ir'])} IR partners in {split}: {preview}")
        split_stats["label_stats"] = aggregate
        totals["splits"][split] = split_stats
        totals["pairs"] += split_stats["pairs"]
        totals["objects"] += split_stats["objects"]

    config = {
        "path": str(processed),
        "train": "rgb/images/train",
        "val": "rgb/images/val",
        "train_ir": "ir/images/train",
        "val_ir": "ir/images/val",
        "train_labels": "labels/train",
        "val_labels": "labels/val",
        "pairing": {"strict": True, "resize_ir": False, "cache_suffix": ".rgbir.npy"},
        "names": {0: "person"},
        "source": {
            "dataset": spec.title,
            "homepage": spec.homepage,
            "citation_doi": spec.citation_doi,
            "license": spec.license_name,
            "variant": variant,
            "task": task,
        },
    }
    yaml_path = processed / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "prepared_by": f"etfnet-sjpa-tr {__version__}",
        "prepared_unix": time.time(),
        "dataset": asdict(spec),
        "variant": variant,
        "task": task,
        "visibility": visibility,
        "exclude_bad": exclude_bad,
        "limit_per_split": limit,
        "source_variant_root": str(variant_root),
        "archive": archive_metadata or {},
        "statistics": totals,
        "data_yaml": str(yaml_path),
    }
    (processed / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                     encoding="utf-8")
    marker.write_text(json.dumps({"schema_version": 1, "data_yaml": str(yaml_path),
                                  "completed_unix": time.time()}, indent=2) + "\n", encoding="utf-8")
    return yaml_path


def _prepare_nii_cu_dataset(
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
) -> Path:
    """Download, validate, extract, preprocess, and return a generated data YAML."""
    spec = get_public_dataset(name)
    variant = variant or spec.default_variant
    root = Path(output_dir).expanduser().resolve() / spec.key
    existing = root / "processed" / variant / "data.yaml"
    if existing.is_file() and not force_preprocess and archive is None:
        return existing
    if not accept_license:
        raise PermissionError(
            f"{spec.title} is distributed under {spec.license_name}. Re-run with --accept-license after reviewing "
            f"{spec.license_url}. The dataset is restricted to non-commercial share-alike use."
        )

    downloads = root / "downloads"
    raw = root / "raw"
    downloads.mkdir(parents=True, exist_ok=True)
    lock = root / ".prepare.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, f"pid={os.getpid()} time={time.time()}\n".encode())
        os.close(descriptor)
    except FileExistsError as exc:
        age = time.time() - lock.stat().st_mtime
        if age > 24 * 3600:
            lock.unlink()
            return _prepare_nii_cu_dataset(
                name, output_dir, variant=variant, task=task, accept_license=accept_license, archive=archive,
                force_download=force_download, force_extract=force_extract, force_preprocess=force_preprocess,
                keep_archive=keep_archive, link_mode=link_mode, visibility=visibility,
                exclude_bad=exclude_bad, limit=limit,
            )
        raise DatasetPreparationError(f"Another preparation process holds {lock}") from exc

    try:
        if archive is None:
            archive_path = downloads / spec.archive_name
            if force_download:
                archive_path.unlink(missing_ok=True)
                archive_path.with_suffix(archive_path.suffix + ".part").unlink(missing_ok=True)
            if not archive_path.is_file():
                download_with_resume(spec.download_url, archive_path)
        else:
            archive_path = Path(archive).expanduser().resolve()
            if not archive_path.is_file():
                raise FileNotFoundError(f"Dataset archive does not exist: {archive_path}")
        archive_info = {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        }
        extracted = safe_extract_zip(archive_path, raw, force=force_extract)
        yaml_path = preprocess_nii_cu_mapd(
            extracted, root, variant=variant, task=task, link_mode=link_mode, visibility=visibility,
            exclude_bad=exclude_bad, limit=limit, force=force_preprocess, archive_metadata=archive_info,
        )
        if archive is None and not keep_archive:
            archive_path.unlink(missing_ok=True)
        return yaml_path
    finally:
        lock.unlink(missing_ok=True)



def prepare_public_dataset(
    name: str,
    output_dir: str | Path = "datasets",
    **kwargs,
) -> Path:
    """Prepare any supported public dataset through one stable API."""
    key = name.strip().lower()
    if key == "nii-cu-mapd":
        kwargs.pop("val_fraction", None)
        kwargs.pop("split_seed", None)
        kwargs.pop("reference_modality", None)
        return _prepare_nii_cu_dataset(key, output_dir, **kwargs)
    from ultralytics.data.public_multidataset import ADDITIONAL_DATASETS, prepare_additional_dataset
    if key in ADDITIONAL_DATASETS:
        return prepare_additional_dataset(key, output_dir, **kwargs)
    available = ", ".join(sorted(_dataset_registry()))
    raise KeyError(f"Unknown public dataset '{name}'. Available: {available}")


def parse_public_reference(value: str) -> str | None:
    """Return the dataset key from ``public:<key>`` or None for a normal YAML path."""
    prefix = "public:"
    return value[len(prefix):].strip().lower() if value.lower().startswith(prefix) else None


def resolve_data_reference(
    value: str,
    *,
    datasets_dir: str | Path = "datasets",
    variant: str | None = None,
    task: str = "obb",
    accept_license: bool = False,
    limit: int | None = None,
    val_fraction: float = 0.2,
    split_seed: int = 0,
) -> str:
    """Resolve a local YAML or automatically prepare a ``public:<name>`` reference."""
    key = parse_public_reference(value)
    if key is None:
        return value
    return str(prepare_public_dataset(
        key, datasets_dir, variant=variant, task=task, accept_license=accept_license,
        limit=limit, val_fraction=val_fraction, split_seed=split_seed,
    ))
