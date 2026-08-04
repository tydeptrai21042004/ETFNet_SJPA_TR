from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml


def _write_image(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((64, 64, 3), value, dtype=np.uint8))


def test_official_vedai_sparse_ids_and_metadata_file(tmp_path: Path):
    from ultralytics.data.public_rgb_ir import prepare_public_dataset

    source = tmp_path / 'source'
    ann = source / 'Annotations512'
    ann.mkdir(parents=True)
    # Official metadata file must not be interpreted as an image annotation.
    (ann / 'annotation512.txt').write_text('metadata only\n', encoding='utf-8')
    raw_ids = [31, 6, 23, 5, 1, 11, 4, 2, 9, 7, 8, 10, 12]
    for index, raw_id in enumerate(raw_ids):
        stem = f'{index:08d}'
        _write_image(source / 'Vehicules512' / f'{stem}_co.png', 30 + index)
        _write_image(source / 'Vehicules512' / f'{stem}_ir.png', 130 + index)
        (ann / f'{stem}.txt').write_text(
            f'30 30 0 {raw_id} 1 0 20 40 40 20 20 20 40 40\n', encoding='utf-8')

    data_yaml = prepare_public_dataset('vedai', tmp_path / 'datasets', archive=source,
                                       variant='512', task='obb', accept_license=True,
                                       force_preprocess=True)
    config = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    assert tuple(config['names'].values()) == (
        'plane', 'boat', 'camping-car', 'car', 'pickup', 'tractor', 'truck', 'van', 'other')
    label_files = list((data_yaml.parent / 'labels').rglob('*.txt'))
    assert len(label_files) == len(raw_ids)
    classes = [int(path.read_text().split()[0]) for path in label_files]
    assert set(classes) == set(range(9))


def test_sjpa_outer_autocast_keeps_linalg_in_fp32():
    from ultralytics.nn.modules.block import SJPA

    module = SJPA(128, 32, 8).train()
    rgb = torch.randn(2, 128, 16, 16, requires_grad=True)
    ir = torch.randn(2, 128, 16, 16, requires_grad=True)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        output = module([rgb, ir])
        loss = output.float().square().mean()
    loss.backward()
    assert output.shape == (2, 256, 16, 16)
    assert torch.isfinite(output.float()).all()
    assert rgb.grad is not None and ir.grad is not None
    assert torch.isfinite(rgb.grad).all() and torch.isfinite(ir.grad).all()
    assert module.running_cov_r.dtype == torch.float32


def test_running_statistics_accept_mixed_input_dtype():
    from ultralytics.nn.modules.block import GOCI

    module = GOCI(128, 32, 8).train()
    shape_mu = module.running_mu_r.shape
    shape_cov = module.running_cov_r.shape
    stats = (
        torch.zeros(shape_mu, dtype=torch.float16),
        torch.zeros(shape_mu, dtype=torch.float16),
        torch.eye(module.group_width, dtype=torch.float16).repeat(module.groups, 1, 1),
        torch.eye(module.group_width, dtype=torch.float16).repeat(module.groups, 1, 1),
        torch.eye(module.group_width, dtype=torch.float16).repeat(module.groups, 1, 1),
    )
    module._update_running(stats)
    assert module.running_cov_r.dtype == torch.float32
    assert int(module.num_updates) == 1


def test_ray_callback_is_noop_for_new_api_without_session(monkeypatch):
    import ultralytics.utils.callbacks.raytune as callback

    fake_tune = SimpleNamespace()  # deliberately has no is_session_enabled
    monkeypatch.setattr(callback, 'ray', SimpleNamespace(tune=fake_tune))
    monkeypatch.setattr(callback, 'tune', fake_tune)
    monkeypatch.setattr(callback, 'air_session', None)
    trainer = SimpleNamespace(metrics={'mAP50': 0.1}, epoch=0)
    callback.on_fit_epoch_end(trainer)  # must not raise


def test_wandb_project_path_is_sanitized():
    from ultralytics.utils.callbacks.wb import _safe_project_name

    assert _safe_project_name('/kaggle/working/etfnet/results') == 'results'
    assert _safe_project_name('project:bad/name') == 'name'


def test_cli_infers_rgb_and_rgb_ir_model_channels():
    from etfnet_cli import _model_channels
    from ultralytics import YOLO

    root = Path(__file__).resolve().parents[1]
    rgb = YOLO(str(root / 'ultralytics/cfg/models/etfnet/etfnet_yolo11.yaml'))
    rgb_ir = YOLO(str(root / 'ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml'))
    assert _model_channels(rgb) == 3
    assert _model_channels(rgb_ir) == 6
