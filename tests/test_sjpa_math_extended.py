from __future__ import annotations

import copy
import math

import pytest
import torch

from ultralytics.nn.modules.block import GOCI, SJPA


def _block_orthogonal(groups: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    blocks = []
    for _ in range(groups):
        q, r = torch.linalg.qr(torch.randn(width, width, generator=generator))
        sign = torch.sign(torch.diagonal(r))
        sign[sign == 0] = 1
        blocks.append(q * sign)
    return torch.stack(blocks)


def _apply_group_rotation(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    groups, group_width, _ = q.shape
    tokens = x.reshape(batch, groups, group_width, height * width).permute(0, 1, 3, 2)
    tokens = tokens @ q.unsqueeze(0)
    return tokens.permute(0, 1, 3, 2).reshape(batch, channels, height, width)


def _fit_running_stats(module: GOCI, steps: int = 4) -> None:
    module.train()
    generator = torch.Generator().manual_seed(100)
    mixing = _block_orthogonal(module.groups, module.group_width, 12)
    for _ in range(steps):
        rgb = torch.randn(5, module.channels, 12, 10, generator=generator)
        ir = _apply_group_rotation(rgb, mixing) + 0.05 * torch.randn(
            rgb.shape, generator=generator
        )
        module([rgb, ir])
    module.eval()


def test_constructor_rejects_invalid_grouping():
    with pytest.raises(ValueError):
        SJPA(channels=10, groups=3)
    with pytest.raises(ValueError):
        SJPA(channels=0, groups=1)
    with pytest.raises(ValueError):
        SJPA(channels=8, groups=2, max_shift=-1)


@pytest.mark.parametrize("shape", [(1, 8, 32, 32), (2, 8, 31, 27), (3, 8, 16, 24)])
def test_forward_backward_multiple_shapes(shape):
    module = SJPA(8, groups=2, anchors=4, max_shift=1).train()
    rgb = torch.randn(shape, requires_grad=True)
    ir = torch.randn(shape, requires_grad=True)
    output = module([rgb, ir])
    assert output.shape == (shape[0], 16, shape[2], shape[3])
    output.square().mean().backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert ir.grad is not None and torch.isfinite(ir.grad).all()


@pytest.mark.parametrize("fill", [0.0, 1.0, -2.5])
def test_constant_inputs_are_finite(fill: float):
    module = SJPA(8, groups=2, anchors=4).eval()
    rgb = torch.full((2, 8, 20, 20), fill)
    ir = torch.full_like(rgb, fill)
    output, diagnostics = module.forward_with_diagnostics([rgb, ir])
    assert torch.isfinite(output).all()
    assert torch.isfinite(diagnostics["probability"]).all()
    assert torch.allclose(diagnostics["probability"].sum(1), torch.ones(2, 1, 1, 1), atol=1e-7)


def test_procrustes_transform_is_orthogonal_after_updates():
    module = GOCI(8, groups=2, anchors=4, momentum=0.2)
    _fit_running_stats(module)
    *_, q = module._get_transforms(torch.empty(0), torch.empty(0))
    identity = torch.eye(module.group_width).expand(module.groups, -1, -1)
    assert torch.allclose(q.transpose(-1, -2) @ q, identity, atol=2e-5, rtol=2e-5)


def test_running_whitening_is_close_to_identity():
    module = GOCI(8, groups=2, anchors=4, momentum=1.0, eps=1e-4)
    generator = torch.Generator().manual_seed(9)
    base = torch.randn(48, 8, 12, 12, generator=generator)
    scales = torch.tensor([0.4, 0.8, 1.2, 1.8, 0.6, 1.1, 1.5, 2.0]).view(1, 8, 1, 1)
    rgb = base * scales
    ir = base * scales.flip(1)
    module.train()
    module([rgb, ir])
    module.eval()
    mu_r, _, wr, _, _ = module._get_transforms(rgb, ir)
    whitened = module._apply_whitening(rgb, mu_r, wr)
    pooled = module.pool(whitened)
    tokens = module._group_tokens(pooled.float())
    centered = tokens - tokens.mean(1, keepdim=True)
    covariance = centered.transpose(-1, -2) @ centered / (tokens.shape[1] - 1)
    identity = torch.eye(module.group_width).expand(module.groups, -1, -1)
    assert torch.allclose(covariance, identity, atol=0.08, rtol=0.08)


def test_reliability_simplex_and_trust_bound():
    module = GOCI(8, groups=2, reliability=True).eval()
    rgb = torch.randn(4, 8, 10, 10)
    ir = 4.0 * torch.randn_like(rgb)
    probability, trigger, _, _ = module._reliability_terms(rgb, ir)
    output = module._reliability_output(rgb, ir)
    out_rgb, out_ir = output.chunk(2, 1)
    assert torch.allclose(probability.sum(1), torch.ones_like(trigger), atol=1e-7)
    assert (probability >= 0).all() and (probability <= 1).all()
    for source, corrected in ((rgb, out_rgb), (ir, out_ir)):
        lhs = (corrected - source).flatten(1).norm(dim=1)
        rhs = trigger.flatten(1) * source.flatten(1).norm(dim=1) + 1e-6
        assert torch.all(lhs <= rhs)


def test_missing_modality_probability_favors_present_stream():
    module = GOCI(8, groups=2, reliability=True).eval()
    rgb = torch.randn(3, 8, 12, 12)
    ir = torch.zeros_like(rgb)
    probability, trigger, *_ = module._reliability_terms(rgb, ir)
    assert torch.all(probability[:, 0] > probability[:, 1])
    assert torch.all(trigger > 0.5)


def test_known_translation_is_recovered_without_wrap():
    module = SJPA(
        8,
        groups=2,
        anchors=8,
        reliability=False,
        trigger_tau=100.0,
        max_shift=1,
        shift_penalty=0.0,
        score_threshold=100.0,
    ).eval()
    rgb = torch.zeros(1, 8, 16, 16)
    generator = torch.Generator().manual_seed(4)
    rgb[..., 4:12, 4:12] = torch.randn(1, 8, 8, 8, generator=generator)
    ir = SJPA._shift_no_wrap(rgb, 1, -1)
    _, diagnostics = module.forward_with_diagnostics([rgb, ir])
    assert diagnostics["selected_shift"].tolist() == [[-1, 1]]


@pytest.mark.parametrize(
    "applied_shift",
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
)
def test_all_unit_translations_are_recovered_without_wrap(applied_shift):
    module = SJPA(
        8,
        groups=2,
        anchors=8,
        reliability=False,
        trigger_tau=100.0,
        max_shift=1,
        shift_penalty=0.0,
        score_threshold=100.0,
    ).eval()
    dy, dx = applied_shift
    rgb = torch.zeros(1, 8, 24, 24)
    generator = torch.Generator().manual_seed(100 + (dy + 1) * 3 + (dx + 1))
    rgb[..., 6:18, 6:18] = torch.randn(1, 8, 12, 12, generator=generator)
    ir = SJPA._shift_no_wrap(rgb, dy, dx)
    _, diagnostics = module.forward_with_diagnostics([rgb, ir])
    assert diagnostics["selected_shift"].tolist() == [[-dy, -dx]]


def test_torchscript_trace_matches_export_mode_on_new_input():
    module = SJPA(
        8,
        groups=2,
        anchors=6,
        reliability=True,
        trigger_tau=100.0,
        max_shift=1,
        shift_penalty=0.05,
        score_threshold=100.0,
    )
    _fit_running_stats(module, steps=3)
    module.prepare_for_export().eval()
    rgb = torch.randn(1, 8, 20, 20)
    ir = SJPA._shift_no_wrap(rgb, 1, 0)
    traced = torch.jit.trace(module, ([rgb, ir],), strict=False)

    rgb_new = torch.randn(1, 8, 20, 20)
    ir_new = SJPA._shift_no_wrap(rgb_new, -1, 1)
    with torch.inference_mode():
        eager = module([rgb_new, ir_new])
        scripted = traced([rgb_new, ir_new])
    assert torch.isfinite(scripted).all()
    assert torch.allclose(scripted, eager, atol=1e-6, rtol=1e-6)


def test_identical_pair_prefers_zero_shift():
    module = SJPA(
        8,
        groups=2,
        anchors=6,
        reliability=False,
        trigger_tau=100.0,
        max_shift=1,
        shift_penalty=0.2,
        score_threshold=100.0,
    ).eval()
    rgb = torch.randn(3, 8, 18, 18)
    _, diagnostics = module.forward_with_diagnostics([rgb, rgb.clone()])
    assert torch.equal(diagnostics["selected_shift"], torch.zeros(3, 2, dtype=torch.long))


def test_groupwise_gauge_equivariance_of_decisions_and_output():
    groups, width = 2, 4
    base = SJPA(
        8,
        groups=groups,
        anchors=6,
        reliability=True,
        trigger_tau=100.0,
        max_shift=1,
        shift_penalty=0.05,
        score_threshold=100.0,
    ).eval()
    rotated = copy.deepcopy(base)
    q_rgb = _block_orthogonal(groups, width, 40)
    q_ir = _block_orthogonal(groups, width, 41)
    rotated.running_cross.copy_(q_rgb.transpose(-1, -2) @ q_ir)
    rotated._cached_eval = None
    rgb = torch.randn(2, 8, 20, 20)
    ir = SJPA._shift_no_wrap(rgb, 1, 0) + 0.01 * torch.randn_like(rgb)
    output, diag = base.forward_with_diagnostics([rgb, ir])
    rgb_rot = _apply_group_rotation(rgb, q_rgb)
    ir_rot = _apply_group_rotation(ir, q_ir)
    output_rot, diag_rot = rotated.forward_with_diagnostics([rgb_rot, ir_rot])
    expected = torch.cat(
        (
            _apply_group_rotation(output[:, :8], q_ir),
            _apply_group_rotation(output[:, 8:], q_ir),
        ),
        dim=1,
    )
    assert torch.equal(diag["selected_shift"], diag_rot["selected_shift"])
    assert torch.allclose(diag["probability"], diag_rot["probability"], atol=2e-5, rtol=2e-5)
    assert torch.allclose(output_rot, expected, atol=3e-4, rtol=3e-4)


def test_state_dict_roundtrip_preserves_eval_output():
    module = SJPA(8, groups=2, anchors=4, max_shift=0)
    _fit_running_stats(module)
    rgb = torch.randn(2, 8, 20, 20)
    ir = torch.randn_like(rgb)
    expected = module([rgb, ir])
    restored = SJPA(8, groups=2, anchors=4, max_shift=0).eval()
    restored.load_state_dict(module.state_dict())
    actual = restored([rgb, ir])
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert int(restored.num_updates) == int(module.num_updates)


def test_prepare_for_export_matches_eval_for_zero_shift():
    module = SJPA(8, groups=2, anchors=4, max_shift=0)
    _fit_running_stats(module)
    rgb = torch.randn(2, 8, 18, 18)
    ir = torch.randn_like(rgb)
    expected = module([rgb, ir])
    module.prepare_for_export().eval()
    actual = module([rgb, ir])
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_cpu_bfloat16_autocast_is_finite():
    if not hasattr(torch, "autocast"):
        pytest.skip("autocast unavailable")
    module = SJPA(8, groups=2, anchors=4, max_shift=1).train()
    rgb = torch.randn(2, 8, 24, 24, requires_grad=True)
    ir = torch.randn_like(rgb, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = module([rgb, ir])
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output.float()).all()
    assert torch.isfinite(rgb.grad).all() and torch.isfinite(ir.grad).all()


def test_eval_transform_cache_is_invalidated_by_training_update():
    module = GOCI(8, groups=2, anchors=4, momentum=0.5)
    _fit_running_stats(module, steps=2)
    rgb = torch.randn(2, 8, 12, 12)
    ir = torch.randn_like(rgb)
    module.eval()
    module([rgb, ir])
    old_version = module._cached_version
    old_cache = module._cached_eval
    module.train()
    module([rgb + 2.0, ir - 1.0])
    assert module._cached_eval is None
    module.eval()
    module([rgb, ir])
    assert module._cached_version != old_version
    assert module._cached_eval is not old_cache
