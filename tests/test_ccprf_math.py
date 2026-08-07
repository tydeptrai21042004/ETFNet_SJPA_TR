from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules.block import CCPRF


def _features(batch=3, channels=16, height=24, width=24):
    torch.manual_seed(31)
    base = F.avg_pool2d(torch.randn(batch, channels, height + 4, width + 4), 5, 1)
    rgb = F.silu(1.15 * base + 0.20 * torch.randn_like(base))
    ir = F.silu(0.85 * base + 0.30 * torch.randn_like(base) + 0.08)
    return rgb, ir


def _flatten_outputs(value):
    if torch.is_tensor(value):
        return [value]
    result = []
    if isinstance(value, (list, tuple)):
        for item in value:
            result.extend(_flatten_outputs(item))
    elif isinstance(value, dict):
        for item in value.values():
            result.extend(_flatten_outputs(item))
    return result


def test_ccprf_is_exact_concat_at_initialization():
    rgb, ir = _features()
    module = CCPRF(16, groups=2, stat_grid=8)
    output = module([rgb, ir])
    assert torch.equal(output, torch.cat((rgb, ir), dim=1))


def test_ccprf_hybrid_basis_vanishes_when_cross_covariance_is_zero():
    torch.manual_seed(3)
    rw = torch.randn(2, 3, 11, 4)
    iw = torch.randn_like(rw)
    cross = torch.zeros(2, 3, 4, 4)
    basis_r, basis_i, reliability_r, reliability_i = CCPRF._hybrid_whitened_basis(
        rw, iw, cross
    )
    assert torch.count_nonzero(basis_r) == 0
    assert torch.count_nonzero(basis_i) == 0
    assert torch.count_nonzero(reliability_r) == 0
    assert torch.count_nonzero(reliability_i) == 0


def test_ccprf_hybrid_basis_has_directionwise_canonical_shrinkage():
    # In a matched canonical basis, direction j follows
    # H_r,j = sigma_j(1+sigma_j^2) I_j - sigma_j^2 R_j.
    rw = torch.tensor([[[[2.0, 2.0, 2.0]]]])
    iw = torch.tensor([[[[3.0, 3.0, 3.0]]]])
    sigma = torch.tensor([0.0, 0.2, 1.0])
    cross = torch.diag(sigma).view(1, 1, 3, 3)
    basis_r, basis_i, _, _ = CCPRF._hybrid_whitened_basis(rw, iw, cross)
    expected_r = sigma * (1.0 + sigma.square()) * 3.0 - sigma.square() * 2.0
    expected_i = sigma * (1.0 + sigma.square()) * 2.0 - sigma.square() * 3.0
    assert torch.allclose(basis_r.flatten(), expected_r, atol=1e-7, rtol=1e-7)
    assert torch.allclose(basis_i.flatten(), expected_i, atol=1e-7, rtol=1e-7)
    assert basis_r.flatten()[0] == 0
    assert basis_i.flatten()[0] == 0


def test_ccprf_group_layout_pairs_corresponding_rgb_and_ir_groups():
    module = CCPRF(8, groups=2, stat_grid=4)
    rgb = torch.arange(8.0).view(1, 8, 1, 1)
    ir = (100.0 + torch.arange(8.0)).view(1, 8, 1, 1)
    grouped = module._group_major_pair(rgb, ir)
    expected = torch.tensor(
        [0, 1, 2, 3, 100, 101, 102, 103,
         4, 5, 6, 7, 104, 105, 106, 107],
        dtype=torch.float32,
    ).view(1, 16, 1, 1)
    assert torch.equal(grouped, expected)
    assert torch.equal(module._modality_major_pair(grouped), torch.cat((rgb, ir), 1))

    # Each grouped output can directly read the matching IR subgroup.
    with torch.no_grad():
        module.innovation_weight.zero_()
        d = module.group_width
        module.innovation_weight[0, d, 0, 0] = 1.0
        module.innovation_weight[2 * d, d, 0, 0] = 1.0
    grouped_output = F.conv2d(grouped, module.innovation_weight, groups=module.groups)
    output = module._modality_major_pair(grouped_output)
    assert output[0, 0, 0, 0] == 100
    assert output[0, 4, 0, 0] == 104


def test_ccprf_cross_operator_and_coherence_are_bounded():
    rgb, ir = _features(batch=4)
    module = CCPRF(16, groups=2, stat_grid=8)
    with torch.no_grad():
        stats = module._statistics(rgb, ir)
        cross = stats[-2]
        coherence = stats[-1]
        singular = torch.linalg.svdvals(cross)
    assert float(singular.max()) <= 1.0001
    assert float(coherence.min()) >= 0.0
    assert float(coherence.max()) <= 1.0


def test_ccprf_coherence_tracks_cross_modal_correlation():
    rgb, ir = _features(batch=4)
    module = CCPRF(16, groups=2, stat_grid=8)
    with torch.no_grad():
        _, correlated = module.forward_with_diagnostics([rgb, ir])
        _, independent = module.forward_with_diagnostics([rgb, ir.flip(0)])
    assert correlated['coherence'].mean() > independent['coherence'].mean()


def test_ccprf_basis_is_modality_scale_equivariant():
    rgb, ir = _features(batch=2)
    module = CCPRF(16, groups=2, stat_grid=8)
    with torch.no_grad():
        basis_r, basis_i, *_ = module._cross_confirmed_basis(rgb, ir)
        scaled_r, scaled_i, *_ = module._cross_confirmed_basis(3.0 * rgb, 0.5 * ir)
    assert torch.allclose(scaled_r, 3.0 * basis_r, atol=2e-5, rtol=2e-4)
    assert torch.allclose(scaled_i, 0.5 * basis_i, atol=2e-5, rtol=2e-4)


def test_ccprf_rank_deficient_zero_and_extreme_inputs_are_finite():
    module = CCPRF(16, groups=2, stat_grid=8)
    with torch.no_grad():
        module.innovation_weight.normal_(0.0, 0.1)
    cases = []
    for value in (0.0, 1.0, -2.0, 1e-6, 1e6):
        rgb = torch.full((2, 16, 20, 20), value)
        ir = torch.full_like(rgb, value)
        cases.append((rgb, ir))
    # Exactly rank-one spatial/channel structure.
    axis = torch.linspace(-1, 1, 20).view(1, 1, 20, 1)
    rank_one = axis.expand(2, 16, 20, 20)
    cases.append((rank_one, 2.0 * rank_one))

    for rgb, ir in cases:
        output, diagnostics = module.forward_with_diagnostics([rgb, ir])
        assert torch.isfinite(output).all()
        for key in ('rgb_basis', 'ir_basis', 'cross_prediction', 'coherence'):
            assert torch.isfinite(diagnostics[key]).all(), key


def test_ccprf_output_is_independent_of_batch_companions():
    rgb, ir = _features(batch=3)
    module = CCPRF(16, groups=2, stat_grid=8)
    with torch.no_grad():
        module.innovation_weight.normal_(0.0, 0.2)
        first = module([rgb, ir])[:1]
        second = module([
            torch.cat((rgb[:1], torch.randn_like(rgb[1:])), dim=0),
            torch.cat((ir[:1], torch.randn_like(ir[1:])), dim=0),
        ])[:1]
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_ccprf_trust_bound_and_finite_gradients():
    rgb, ir = _features()
    rgb.requires_grad_()
    ir.requires_grad_()
    module = CCPRF(16, groups=2, stat_grid=8, trust_radius=0.05)
    with torch.no_grad():
        module.innovation_weight.normal_(0.0, 0.5)
    output, diagnostics = module.forward_with_diagnostics([rgb, ir])
    raw = diagnostics['raw']
    ratio = (output - raw).flatten(1).norm(dim=1) / (
        raw.flatten(1).norm(dim=1) + 1e-8
    )
    assert float(ratio.max().detach()) <= 0.0501
    output.square().mean().backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert ir.grad is not None and torch.isfinite(ir.grad).all()
    assert module.innovation_weight.grad is not None
    assert torch.isfinite(module.innovation_weight.grad).all()
    assert float(module.innovation_weight.grad.abs().sum()) > 0.0


def test_ccprf_rejects_mismatched_modalities():
    module = CCPRF(16, groups=2, stat_grid=8)
    rgb = torch.randn(1, 16, 20, 20)
    ir = torch.randn(1, 16, 19, 20)
    try:
        module([rgb, ir])
    except ValueError as error:
        assert 'matching RGB/IR shapes' in str(error)
    else:
        raise AssertionError('mismatched modality shapes must be rejected')


def test_ccprf_model_yaml_builds_and_forwards_zero_and_random_inputs():
    model = YOLO('ultralytics/cfg/models/etfnet/etfnet_P2_C3k2_CCPRF.yaml')
    model.model.eval()
    with torch.no_grad():
        random_output = model.model(torch.randn(1, 6, 64, 64))
        zero_output = model.model(torch.zeros(1, 6, 64, 64))
    for tensor in _flatten_outputs(random_output) + _flatten_outputs(zero_output):
        assert torch.isfinite(tensor).all()


def test_ccprf_does_not_shift_downstream_rng_or_initial_predictions():
    base_yaml = 'ultralytics/cfg/models/etfnet/etfnet_dualstream_noCAFEM_noTGF.yaml'
    ccprf_yaml = 'ultralytics/cfg/models/etfnet/etfnet_P2_C3k2_CCPRF.yaml'
    torch.manual_seed(123)
    baseline = YOLO(base_yaml).model.eval()
    torch.manual_seed(123)
    proposal = YOLO(ccprf_yaml).model.eval()

    baseline_state = baseline.state_dict()
    proposal_state = proposal.state_dict()
    for key, value in baseline_state.items():
        if key in proposal_state and value.shape == proposal_state[key].shape:
            assert torch.equal(value, proposal_state[key]), key

    x = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        baseline_output = baseline(x)
        proposal_output = proposal(x)
    baseline_tensors = _flatten_outputs(baseline_output)
    proposal_tensors = _flatten_outputs(proposal_output)
    assert len(baseline_tensors) == len(proposal_tensors)
    for left, right in zip(baseline_tensors, proposal_tensors):
        assert torch.equal(left, right)
