from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules.block import MPCRF


def _features(batch=3, channels=16, height=24, width=24):
    torch.manual_seed(19)
    base = F.avg_pool2d(torch.randn(batch, channels, height + 4, width + 4), 5, 1)
    rgb = F.silu(1.1 * base + 0.25 * torch.randn_like(base))
    ir = F.silu(0.8 * base + 0.35 * torch.randn_like(base) + 0.1)
    return rgb, ir


def _sample_moments(module, x):
    tokens = module._tokens(module._point_sample(x.float()))
    mean = tokens.mean(-2)
    centered = tokens - tokens.mean(-2, keepdim=True)
    covariance = centered.transpose(-1, -2) @ centered / (tokens.shape[-2] - 1)
    return mean, covariance


def test_mpcrf_is_exact_concat_at_initialization():
    rgb, ir = _features()
    module = MPCRF(16, groups=4, stat_grid=8)
    output = module([rgb, ir])
    assert torch.equal(output, torch.cat((rgb, ir), dim=1))


def test_mpcrf_candidates_preserve_sampled_first_and_second_moments():
    rgb, ir = _features(batch=2)
    module = MPCRF(16, groups=4, stat_grid=8, eps=1e-5)
    _, diagnostics = module.forward_with_diagnostics([rgb, ir])
    for raw, candidate in (
        (rgb, diagnostics['rgb_candidate']),
        (ir, diagnostics['ir_candidate']),
    ):
        mean_raw, cov_raw = _sample_moments(module, raw)
        mean_candidate, cov_candidate = _sample_moments(module, candidate)
        assert torch.allclose(mean_raw, mean_candidate, atol=2e-5, rtol=2e-5)
        assert torch.allclose(cov_raw, cov_candidate, atol=2e-4, rtol=2e-3)


def test_mpcrf_output_is_independent_of_batch_companions():
    rgb, ir = _features(batch=3)
    module = MPCRF(16, groups=4, stat_grid=8)
    with torch.no_grad():
        module.rgb_residual_weight.normal_(0.0, 0.2)
        module.ir_residual_weight.normal_(0.0, 0.2)
        first = module([rgb, ir])[:1]
        second = module([
            torch.cat((rgb[:1], torch.randn_like(rgb[1:])), dim=0),
            torch.cat((ir[:1], torch.randn_like(ir[1:])), dim=0),
        ])[:1]
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_mpcrf_trust_bound_and_finite_gradients():
    rgb, ir = _features()
    rgb.requires_grad_()
    ir.requires_grad_()
    module = MPCRF(16, groups=4, stat_grid=8, trust_radius=0.2)
    with torch.no_grad():
        module.rgb_residual_weight.normal_(0.0, 0.5)
        module.ir_residual_weight.normal_(0.0, 0.5)
    output, diagnostics = module.forward_with_diagnostics([rgb, ir])
    raw = diagnostics['raw']
    ratio = (output - raw).flatten(1).norm(dim=1) / (raw.flatten(1).norm(dim=1) + 1e-8)
    assert float(ratio.max().detach()) <= 0.2001
    output.square().mean().backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert ir.grad is not None and torch.isfinite(ir.grad).all()
    assert torch.isfinite(module.rgb_residual_weight.grad).all()
    assert torch.isfinite(module.ir_residual_weight.grad).all()


def test_mpcrf_correlated_pair_has_higher_coherence_than_independent_pair():
    rgb, ir = _features(batch=4)
    module = MPCRF(16, groups=4, stat_grid=8)
    with torch.no_grad():
        _, correlated = module.forward_with_diagnostics([rgb, ir])
        _, independent = module.forward_with_diagnostics([rgb, ir.flip(0)])
    assert correlated['coherence'].mean() > independent['coherence'].mean()


def test_mpcrf_model_yaml_builds_and_forwards():
    model = YOLO('ultralytics/cfg/models/etfnet/etfnet_P2_C3k2_MPCRF.yaml')
    model.model.eval()
    with torch.no_grad():
        output = model.model(torch.randn(1, 6, 64, 64))
    assert output is not None



def test_mpcrf_does_not_shift_downstream_rng_or_initial_predictions():
    base_yaml = 'ultralytics/cfg/models/etfnet/etfnet_dualstream_noCAFEM_noTGF.yaml'
    mpcrf_yaml = 'ultralytics/cfg/models/etfnet/etfnet_P2_C3k2_MPCRF.yaml'
    torch.manual_seed(123)
    baseline = YOLO(base_yaml).model.eval()
    torch.manual_seed(123)
    proposal = YOLO(mpcrf_yaml).model.eval()

    baseline_state = baseline.state_dict()
    proposal_state = proposal.state_dict()
    for key, value in baseline_state.items():
        if key in proposal_state and value.shape == proposal_state[key].shape:
            assert torch.equal(value, proposal_state[key]), key

    x = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        baseline_output = baseline(x)
        proposal_output = proposal(x)

    def tensors(value):
        if torch.is_tensor(value):
            return [value]
        result = []
        if isinstance(value, (list, tuple)):
            for item in value:
                result.extend(tensors(item))
        return result

    for left, right in zip(tensors(baseline_output), tensors(proposal_output)):
        assert torch.equal(left, right)


def test_mpcrf_effective_procrustes_residual_is_stable_near_degeneracy():
    torch.manual_seed(77)
    module = MPCRF(16, groups=4, stat_grid=8, trust_radius=0.1)
    rgb = torch.randn(1, 16, 24, 24)
    ir = rgb + 1e-3 * torch.randn_like(rgb)
    perturb_rgb = 1e-6 * torch.randn_like(rgb)
    perturb_ir = 1e-6 * torch.randn_like(ir)

    with torch.no_grad():
        _, first = module.forward_with_diagnostics([rgb, ir])
        _, second = module.forward_with_diagnostics(
            [rgb + perturb_rgb, ir + perturb_ir]
        )

    def effective(diagnostics, raw_rgb, raw_ir):
        gate = diagnostics['gate'].unsqueeze(-1).unsqueeze(-1)
        gate = gate.repeat_interleave(module.group_width, dim=1)
        return torch.cat((
            gate * (diagnostics['rgb_candidate'] - raw_rgb),
            gate * (diagnostics['ir_candidate'] - raw_ir),
        ), dim=1)

    left = effective(first, rgb, ir)
    right = effective(second, rgb + perturb_rgb, ir + perturb_ir)
    input_change = torch.cat((perturb_rgb, perturb_ir), dim=1).norm()
    amplification = (right - left).norm() / (input_change + 1e-12)
    assert float(amplification) < 5.0
