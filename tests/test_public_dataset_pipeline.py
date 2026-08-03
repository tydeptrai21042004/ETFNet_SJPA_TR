from __future__ import annotations

import http.server
import json
import socketserver
import threading
import zipfile
from pathlib import Path

import pytest
import yaml

from etfnet_cli import build_parser
from tests.public_dataset_fixture import create_nii_cu_archive
from ultralytics.data.public_rgb_ir import (DatasetPreparationError, PUBLIC_DATASETS,
                                             download_with_resume, prepare_public_dataset,
                                             resolve_data_reference, safe_extract_zip)
from ultralytics.data.rgb_ir_check import validate_dataset


def test_registry_has_public_source_doi_and_license():
    spec = PUBLIC_DATASETS["nii-cu-mapd"]
    assert spec.homepage.startswith("https://")
    assert spec.citation_doi == "10.1002/rob.22082"
    assert spec.license_name == "CC BY-NC-SA 3.0"
    assert spec.default_variant in spec.variants


def test_license_acceptance_is_required(tmp_path: Path):
    archive = create_nii_cu_archive(tmp_path / "source.zip")
    with pytest.raises(PermissionError, match="accept-license"):
        prepare_public_dataset("nii-cu-mapd", tmp_path / "datasets", archive=archive)


@pytest.mark.parametrize("task,expected_fields", [("obb", 9), ("detect", 5)])
def test_official_layout_is_preprocessed_and_validated(tmp_path: Path, task: str, expected_fields: int):
    archive = create_nii_cu_archive(tmp_path / "source.zip")
    data_yaml = prepare_public_dataset(
        "nii-cu-mapd", tmp_path / "datasets", archive=archive, accept_license=True,
        task=task, exclude_bad=True, link_mode="hardlink",
    )
    assert data_yaml.is_file()
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert config["source"]["citation_doi"] == "10.1002/rob.22082"
    assert config["names"] == {0: "person"}
    label = data_yaml.parent / "labels/train/frame_000.txt"
    rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
    assert rows and all(len(row) == expected_fields for row in rows)
    assert all(0.0 <= float(value) <= 1.0 for row in rows for value in row[1:])
    report = validate_dataset(str(data_yaml), task=task, fingerprint="sha256")
    assert report["ok"], report["errors"]
    assert report["splits"]["train"]["rgb_images"] == 4
    assert report["splits"]["val"]["rgb_images"] == 2
    manifest = json.loads((data_yaml.parent / "SOURCE_MANIFEST.json").read_text())
    assert manifest["archive"]["sha256"]
    assert manifest["statistics"]["pairs"] == 6


def test_preprocessing_is_idempotent_and_public_alias_resolves(tmp_path: Path):
    archive = create_nii_cu_archive(tmp_path / "source.zip")
    datasets = tmp_path / "datasets"
    first = prepare_public_dataset("nii-cu-mapd", datasets, archive=archive, accept_license=True)
    mtime = first.stat().st_mtime_ns
    second = prepare_public_dataset("nii-cu-mapd", datasets, accept_license=False)
    assert second == first
    assert second.stat().st_mtime_ns == mtime
    assert resolve_data_reference("public:nii-cu-mapd", datasets_dir=datasets) == str(first)


def test_dimension_mismatch_is_rejected(tmp_path: Path):
    archive = create_nii_cu_archive(tmp_path / "bad.zip", mismatch=True)
    with pytest.raises(DatasetPreparationError, match="inconsistent dimensions"):
        prepare_public_dataset("nii-cu-mapd", tmp_path / "datasets", archive=archive, accept_license=True)


def test_zip_traversal_and_corruption_are_rejected(tmp_path: Path):
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(DatasetPreparationError, match="Unsafe path"):
        safe_extract_zip(traversal, tmp_path / "out")
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"PK\x03\x04not-a-valid-zip")
    with pytest.raises(DatasetPreparationError, match="Invalid ZIP"):
        safe_extract_zip(corrupt, tmp_path / "corrupt-out")


def test_resumable_http_download(tmp_path: Path):
    archive = create_nii_cu_archive(tmp_path / "served/source.zip")
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(archive.parent), **kwargs)

        def log_message(self, format, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), QuietHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        destination = tmp_path / "downloaded.zip"
        partial = destination.with_suffix(".zip.part")
        payload = archive.read_bytes()
        partial.write_bytes(payload[: max(1, len(payload) // 3)])
        result = download_with_resume(f"http://127.0.0.1:{server.server_address[1]}/source.zip", destination)
        server.shutdown()
        thread.join(timeout=5)
    assert result.read_bytes() == payload


def test_cli_exposes_download_and_public_training_reference():
    parser = build_parser()
    args = parser.parse_args(["download-data", "--accept-license", "--limit", "2"])
    assert args.dataset == "nii-cu-mapd" and args.limit == 2
    train = parser.parse_args(["train", "--data", "public:nii-cu-mapd", "--accept-dataset-license"])
    assert train.data == "public:nii-cu-mapd" and train.accept_dataset_license
