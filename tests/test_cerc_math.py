from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from ultra_modeling.nn.modules.cerc import (
    CERCBackbone,
    CERCModel,
    ConvolutionalCERC,
    UnifiedEvidenceAdapter,
)


def _small_cerc() -> ConvolutionalCERC:
    torch.manual_seed(0)
    return ConvolutionalCERC(
        channels=16,
        group_width=4,
        stat_grid=4,
        trust_radius=0.05,
        kernel_size=3,
    )


def test_cerc_single_evidence_is_exact_identity_at_initialization():
    module = _small_cerc().eval()
    x = torch.randn(2, 16, 13, 11)
    output, diagnostics = module.forward_with_diagnostics(x)
    assert torch.equal(output, x)
    assert diagnostics["evidence_count"] == 1
    assert torch.equal(
        diagnostics["consensus_weights"],
        torch.ones_like(diagnostics["consensus_weights"]),
    )


def test_cerc_two_evidence_consensus_is_permutation_invariant():
    module = _small_cerc().eval()
    a = torch.randn(2, 16, 12, 10)
    b = torch.randn_like(a)
    first = module((a, b))
    second = module((b, a))
    assert torch.allclose(first, second, atol=2e-6, rtol=2e-6)


def test_cerc_three_evidence_consensus_is_permutation_invariant():
    module = _small_cerc().eval()
    values = [torch.randn(1, 16, 9, 7) for _ in range(3)]
    reference = module(values)
    permuted = module([values[2], values[0], values[1]])
    assert torch.allclose(reference, permuted, atol=3e-6, rtol=3e-6)


def test_cerc_matching_evidence_has_higher_cross_support_than_independent():
    module = _small_cerc().eval()
    torch.manual_seed(11)
    x = torch.randn(3, 16, 16, 16)
    _, same = module.forward_with_diagnostics((x, x.clone()))
    _, independent = module.forward_with_diagnostics((x, torch.randn_like(x)))
    g = module.groups
    same_support = torch.stack([same["support"][:, index, g + index] for index in range(g)]).mean()
    independent_support = torch.stack(
        [independent["support"][:, index, g + index] for index in range(g)]
    ).mean()
    assert same_support > independent_support


def test_cerc_support_is_nearly_scale_invariant():
    module = _small_cerc().eval()
    torch.manual_seed(12)
    a = torch.randn(2, 16, 11, 13)
    b = torch.randn_like(a)
    _, first = module.forward_with_diagnostics((a, b))
    _, second = module.forward_with_diagnostics((7.0 * a, 0.25 * b))
    assert torch.allclose(first["support"], second["support"], atol=2e-4, rtol=2e-4)


def test_cerc_zero_constant_rank_deficient_and_extreme_inputs_are_finite():
    module = _small_cerc().eval()
    base = torch.randn(2, 1, 12, 12).repeat(1, 16, 1, 1)
    cases = [
        torch.zeros(2, 16, 12, 12),
        torch.ones(2, 16, 12, 12) * 4.5,
        base,
        torch.randn(2, 16, 12, 12) * 1e8,
        torch.randn(2, 16, 12, 12) * 1e-8,
    ]
    for value in cases:
        output, diagnostics = module.forward_with_diagnostics(value)
        assert torch.isfinite(output).all()
        assert torch.isfinite(diagnostics["support"]).all()
        assert diagnostics["finite"]


def test_cerc_trust_projection_holds_after_nonzero_convolution():
    module = _small_cerc().eval()
    torch.manual_seed(13)
    with torch.no_grad():
        module.transport.weight.normal_(0.0, 0.2)
    values = (torch.randn(3, 16, 14, 10), torch.randn(3, 16, 14, 10))
    _, diagnostics = module.forward_with_diagnostics(values)
    applied = diagnostics["trust_scale"] * diagnostics["correction_norm"]
    radius = diagnostics["trust_radius_norm"]
    assert torch.all(applied <= radius + 2e-5)


def test_cerc_convolution_receives_nonzero_first_step_gradient():
    module = _small_cerc().train()
    x = torch.randn(2, 16, 10, 10)
    target = torch.randn_like(x)
    output = module(x)
    loss = F.mse_loss(output, target)
    loss.backward()
    grad = module.transport.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_cerc_innovation_parameter_count_is_independent_of_evidence_count():
    module = _small_cerc()
    expected = 4 * 4 * 3 * 3
    assert module.innovation_parameters == expected
    one = module(torch.randn(1, 16, 8, 8))
    three = module([torch.randn(1, 16, 8, 8) for _ in range(3)])
    assert one.shape == three.shape == (1, 16, 8, 8)
    assert module.innovation_parameters == expected


def test_unified_evidence_adapter_supports_shared_single_and_shared_multi_inputs():
    adapter = UnifiedEvidenceAdapter(input_channels=1, out_channels=8).eval()
    single = adapter(torch.randn(2, 1, 32, 32))
    multi = adapter([torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)])
    assert len(single) == 1 and single[0].shape == (2, 8, 16, 16)
    assert len(multi) == 2 and multi[0].shape == multi[1].shape == (2, 8, 16, 16)


def test_unified_evidence_adapter_supports_named_missing_modalities():
    adapter = UnifiedEvidenceAdapter(
        input_channels={"t1": 1, "t2": 1, "flair": 1, "adc": 1},
        out_channels=8,
    ).eval()
    subset = adapter({"t1": torch.randn(1, 1, 24, 20), "flair": torch.randn(1, 1, 24, 20)})
    full = adapter(
        {
            "t1": torch.randn(1, 1, 24, 20),
            "t2": torch.randn(1, 1, 24, 20),
            "flair": torch.randn(1, 1, 24, 20),
            "adc": torch.randn(1, 1, 24, 20),
        }
    )
    assert len(subset) == 2
    assert len(full) == 4
    assert all(value.shape == (1, 8, 12, 10) for value in subset)


def _backbone_kwargs(channels=3):
    return {
        "input_channels": channels,
        "widths": (16, 24, 32, 40),
        "group_width": 4,
        "stat_grid": 4,
        "trust_radius": 0.05,
        "relation_kernel": 3,
    }


def test_cerc_backbone_single_rgb_and_medical_grayscale_pyramids():
    rgb = CERCBackbone(**_backbone_kwargs(3)).eval()
    gray = CERCBackbone(**_backbone_kwargs(1)).eval()
    rgb_pyramid = rgb(torch.randn(1, 3, 64, 80))
    gray_pyramid = gray(torch.randn(1, 1, 64, 80))
    assert tuple(rgb_pyramid) == ("P2", "P3", "P4", "P5")
    assert tuple(gray_pyramid) == ("P2", "P3", "P4", "P5")
    assert rgb_pyramid["P2"].shape == gray_pyramid["P2"].shape


def test_cerc_medmnist_like_classification_28x28_grayscale():
    model = CERCModel(
        "classify", num_classes=4, backbone_kwargs=_backbone_kwargs(1)
    ).eval()
    output = model(torch.randn(3, 1, 28, 28))
    assert output.shape == (3, 4)
    assert torch.isfinite(output).all()


def test_cerc_isic_kvasir_like_rgb_segmentation():
    model = CERCModel(
        "segment", num_classes=2, backbone_kwargs=_backbone_kwargs(3), head_channels=16
    ).eval()
    x = torch.randn(2, 3, 96, 80)
    output = model(x)
    assert output.shape == (2, 2, 96, 80)
    assert torch.isfinite(output).all()


def test_cerc_ultrasound_like_grayscale_segmentation():
    model = CERCModel(
        "segment", num_classes=2, backbone_kwargs=_backbone_kwargs(1), head_channels=16
    ).eval()
    x = torch.randn(2, 1, 80, 72)
    output = model(x)
    assert output.shape == (2, 2, 80, 72)


def test_cerc_named_four_sequence_medical_evidence_uses_same_backbone():
    kwargs = _backbone_kwargs({"t1": 1, "t2": 1, "flair": 1, "adc": 1})
    model = CERCModel("segment", num_classes=3, backbone_kwargs=kwargs, head_channels=16).eval()
    data = {
        "t1": torch.randn(1, 1, 64, 64),
        "t2": torch.randn(1, 1, 64, 64),
        "flair": torch.randn(1, 1, 64, 64),
        "adc": torch.randn(1, 1, 64, 64),
    }
    output = model(data)
    assert output.shape == (1, 3, 64, 64)


def test_cerc_detection_and_anomaly_heads_are_task_independent():
    detect = CERCModel(
        "detect", num_classes=5, backbone_kwargs=_backbone_kwargs(3), head_channels=16
    ).eval()
    anomaly = CERCModel(
        "anomaly", backbone_kwargs=_backbone_kwargs(3), anomaly_embedding_channels=8
    ).eval()
    x = torch.randn(1, 3, 64, 64)
    det = detect(x)
    emb = anomaly(x)
    assert tuple(det) == ("P2", "P3", "P4", "P5")
    assert det["P2"]["class_logits"].shape[1] == 5
    assert emb.ndim == 4 and emb.shape[1] == 8 * 4


def test_cerc_checkpoint_round_trip_is_exact():
    model = CERCModel(
        "classify", num_classes=3, backbone_kwargs=_backbone_kwargs(1)
    ).eval()
    clone = copy.deepcopy(model).eval()
    x = torch.randn(2, 1, 28, 28)
    first = model(x)
    clone.load_state_dict(model.state_dict(), strict=True)
    second = clone(x)
    assert torch.equal(first, second)


def test_cerc_vectorized_message_matches_explicit_pairwise_equation():
    module = _small_cerc().eval()
    torch.manual_seed(77)
    values = (torch.randn(1, 16, 7, 6), torch.randn(1, 16, 7, 6))
    atoms, _, _, _, _ = module._stack_atoms(values)
    mean, rms, factor, z_stats, denom = module._statistics(atoms)
    full_z = module._full_whitened(atoms, mean, rms, factor)
    cross, reliability, support = module._relations(z_stats, denom)
    fast = module._relational_innovation(full_z, cross, reliability, support)

    explicit = torch.zeros_like(fast)
    k = full_z.shape[1]
    for a in range(k):
        numerator = torch.zeros_like(full_z[:, a])
        weight_sum = torch.zeros(full_z.shape[0], 1, 1)
        for b in range(k):
            cab = cross[:, a, b]
            aab = reliability[:, a, b]
            prediction = torch.matmul(full_z[:, b], cab.transpose(-1, -2))
            message = torch.matmul(prediction - full_z[:, a], aab)
            weight = support[:, a, b].view(-1, 1, 1)
            numerator = numerator + weight * message
            weight_sum = weight_sum + weight
        explicit[:, a] = numerator / (module.eps + weight_sum)

    assert torch.allclose(fast, explicit, atol=2e-5, rtol=2e-5)


def test_cerc_self_support_is_exactly_zero_without_case_branch():
    module = _small_cerc().eval()
    _, diagnostics = module.forward_with_diagnostics(
        (torch.randn(2, 16, 9, 9), torch.randn(2, 16, 9, 9))
    )
    diagonal = diagnostics["support"].diagonal(dim1=1, dim2=2)
    assert torch.count_nonzero(diagonal).item() == 0
