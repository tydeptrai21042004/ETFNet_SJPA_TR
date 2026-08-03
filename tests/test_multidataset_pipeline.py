from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from etfnet_cli import build_parser
from tests.multidataset_fixtures import create_cvc14, create_flir, create_m3fd, create_rgbtdrone, create_vedai
from ultralytics.data.public_multidataset import ADDITIONAL_DATASETS, safe_extract_tar
from ultralytics.data.public_rgb_ir import PUBLIC_DATASETS, prepare_public_dataset, resolve_data_reference
from ultralytics.data.rgb_ir_check import validate_dataset


@pytest.mark.parametrize(
    "name,fixture,variant,names",
    [
        ("m3fd", create_m3fd, "default", 6),
        ("vedai", create_vedai, "512", 9),
        ("flir-aligned", create_flir, "aligned", 3),
        ("rgbtdroneperson", create_rgbtdrone, "default", 3),
        ("cvc-14", create_cvc14, "default", 1),
    ],
)
@pytest.mark.parametrize("task", ["obb", "detect"])
def test_all_public_datasets_preprocess_and_validate(tmp_path: Path, name, fixture, variant, names, task):
    source = fixture(tmp_path / f"source-{name}")
    data_yaml = prepare_public_dataset(
        name, tmp_path / "datasets", archive=source, variant=variant, task=task,
        accept_license=True, link_mode="copy", force_preprocess=True,
    )
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert len(config["names"]) == names
    assert Path(config["path"]).is_dir()
    report = validate_dataset(str(data_yaml), task=task, fingerprint="sha256")
    assert report["ok"], report["errors"]
    assert report["splits"]["train"]["rgb_images"] >= 1
    assert report["splits"]["val"]["rgb_images"] >= 1
    manifest = json.loads((data_yaml.parent / "SOURCE_MANIFEST.json").read_text())
    assert manifest["statistics"]["pairs"] == 6
    assert manifest["dataset"]["citation_doi"]


def test_license_is_required_for_every_added_dataset(tmp_path: Path):
    for name, spec in ADDITIONAL_DATASETS.items():
        with pytest.raises(PermissionError, match="accept-license"):
            prepare_public_dataset(name, tmp_path / name, variant=spec.default_variant, archive=tmp_path)


def test_public_alias_resolves_existing_dataset_without_reacceptance(tmp_path: Path):
    source = create_m3fd(tmp_path / "source")
    data_yaml = prepare_public_dataset("m3fd", tmp_path / "datasets", archive=source, accept_license=True)
    assert resolve_data_reference("public:m3fd", datasets_dir=tmp_path / "datasets") == str(data_yaml)


def test_vedai_official_x_then_y_coordinate_order(tmp_path: Path):
    source = create_vedai(tmp_path / "vedai")
    data_yaml = prepare_public_dataset("vedai", tmp_path / "datasets", archive=source, variant="512",
                                       accept_license=True, task="obb")
    label = next((data_yaml.parent / "labels/train").rglob("*.txt"))
    values = [float(x) for x in label.read_text().split()]
    assert len(values) == 9
    # Four distinct normalized vertices from x1..x4 followed by y1..y4.
    assert len(set(zip(values[1::2], values[2::2]))) == 4


def test_tar_traversal_is_rejected(tmp_path: Path):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as bundle:
        data = b"no"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        bundle.addfile(info, io.BytesIO(data))
    with pytest.raises(Exception, match="Unsafe path"):
        safe_extract_tar(archive, tmp_path / "out")


def test_cli_lists_and_accepts_every_dataset():
    parser = build_parser()
    for name, spec in PUBLIC_DATASETS.items():
        args = parser.parse_args(["download-data", "--dataset", name, "--variant", spec.default_variant,
                                  "--accept-license", "--limit", "2", "--no-validate"])
        assert args.dataset == name
        train = parser.parse_args(["train", "--data", f"public:{name}", "--dataset-variant", spec.default_variant,
                                   "--dataset-limit", "2", "--accept-dataset-license"])
        assert train.data == f"public:{name}"


def test_split_inference_ignores_test_harness_ancestor():
    from ultralytics.data.public_multidataset import _split_hint

    path = Path('/tmp/pytest/test_converter0/source/images/visible/frame.png')
    assert _split_hint(path) is None
    assert _split_hint(path.parent.parent / 'train' / 'frame.png') == 'train'
    assert _split_hint(path.parent.parent / 'val_thermal.json') == 'val'


def test_gdown_v6_folder_provider_and_file_provider(tmp_path: Path, monkeypatch):
    import sys
    import types
    from ultralytics.data.public_multidataset import _download_gdrive_file, _download_gdrive_folder

    fake = types.ModuleType('gdown')
    fake.__version__ = '6.0.0'

    def download_folder(**kwargs):
        output = Path(kwargs['output'])
        output.mkdir(parents=True, exist_ok=True)
        (output / 'payload.txt').write_text('ok', encoding='utf-8')
        return [str(output / 'payload.txt')]

    def download(**kwargs):
        output = Path(kwargs['output'])
        output.write_bytes(b'PK\x03\x04fixture')
        return str(output)

    fake.download_folder = download_folder
    fake.download = download
    monkeypatch.setitem(sys.modules, 'gdown', fake)
    folder = _download_gdrive_folder('folder-id', tmp_path / 'folder')
    archive = _download_gdrive_file('file-id', tmp_path / 'file.zip')
    assert (folder / 'payload.txt').read_text() == 'ok'
    assert archive.read_bytes().startswith(b'PK')


def test_old_gdown_is_rejected_for_large_folder_download(tmp_path: Path, monkeypatch):
    import sys
    import types
    from ultralytics.data.public_multidataset import _download_gdrive_folder

    fake = types.ModuleType('gdown')
    fake.__version__ = '5.2.0'
    monkeypatch.setitem(sys.modules, 'gdown', fake)
    with pytest.raises(Exception, match='gdown>=6'):
        _download_gdrive_folder('folder-id', tmp_path / 'folder')


def test_modelscope_dataset_provider_dispatch(tmp_path: Path, monkeypatch):
    import sys
    import types
    from ultralytics.data.public_multidataset import _download_modelscope

    package = types.ModuleType('modelscope'); package.__path__ = []
    hub = types.ModuleType('modelscope.hub'); hub.__path__ = []
    snapshot = types.ModuleType('modelscope.hub.snapshot_download')

    def dataset_snapshot_download(*, dataset_id, local_dir):
        assert dataset_id == 'OmniData/CVC-14'
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / 'README.md').write_text('fixture', encoding='utf-8')
        return str(target)

    snapshot.dataset_snapshot_download = dataset_snapshot_download
    monkeypatch.setitem(sys.modules, 'modelscope', package)
    monkeypatch.setitem(sys.modules, 'modelscope.hub', hub)
    monkeypatch.setitem(sys.modules, 'modelscope.hub.snapshot_download', snapshot)
    root = _download_modelscope('OmniData/CVC-14', tmp_path / 'cvc')
    assert (root / 'README.md').is_file()


def test_mixed_image_extensions_are_canonicalized_to_identical_pair_names(tmp_path: Path):
    import cv2
    import numpy as np
    from ultralytics.data.public_multidataset import PairSample, _canonicalize

    rgb = tmp_path / 'rgb.jpg'; ir = tmp_path / 'ir.png'
    cv2.imwrite(str(rgb), np.zeros((32, 32, 3), dtype=np.uint8))
    cv2.imwrite(str(ir), np.ones((32, 32, 3), dtype=np.uint8))
    spec = ADDITIONAL_DATASETS['m3fd']
    samples = [
        PairSample('a', rgb, ir, (), 'train'),
        PairSample('b', rgb, ir, (), 'val'),
    ]
    data_yaml = _canonicalize(samples, tmp_path / 'out', spec, 'default', 'obb', 'copy', None,
                              {'kind': 'fixture'}, force=True)
    root = data_yaml.parent
    for split in ('train', 'val'):
        rgb_names = {p.relative_to(root / 'rgb/images' / split).as_posix() for p in (root / 'rgb/images' / split).rglob('*') if p.is_file()}
        ir_names = {p.relative_to(root / 'ir/images' / split).as_posix() for p in (root / 'ir/images' / split).rglob('*') if p.is_file()}
        assert rgb_names == ir_names
        assert all(name.endswith('.png') for name in rgb_names)
