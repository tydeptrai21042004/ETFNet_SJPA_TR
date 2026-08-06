from __future__ import annotations

import torch

from ultralytics.nn.modules.block import DCSPF


def test_identity_initialized_expert_adapters():
    module = DCSPF(8, groups=2, anchors=4, max_shift=0).eval()
    x = torch.randn(3, 16, 12, 10)
    assert torch.equal(module.raw_adapter(x), x)
    assert torch.equal(module.canonical_adapter(x), x)


def test_normalized_coherence_is_bounded_and_identical_pair_is_high():
    module = DCSPF(8, groups=2, anchors=4, max_shift=0).eval()
    x = torch.randn(5, 8, 20, 20)
    coherence = module._normalized_coherence(x, x.clone())
    assert torch.isfinite(coherence).all()
    assert (coherence >= 0).all() and (coherence <= 1).all()
    assert torch.all(coherence > 0.95)


def test_guard_routes_coherent_pair_and_clear_single_modality():
    module = DCSPF(8, groups=2, anchors=4, max_shift=0).eval()
    coherent = 4.0 * torch.randn(4, 8, 16, 16)
    gate, coherence, typicality, dominance, *_ = module._guard_terms(coherent, coherent.clone())
    assert torch.all(gate == 1)
    assert torch.all(coherence > module.coherence_tau)

    present = 4.0 * torch.randn(4, 8, 16, 16)
    missing = torch.zeros_like(present)
    gate, _, _, dominance, *_ = module._guard_terms(present, missing)
    assert torch.all(gate == 1)
    assert torch.all(dominance >= module.dominance_tau)


def test_guard_falls_back_when_both_modalities_are_atypical_and_incoherent():
    module = DCSPF(8, groups=2, anchors=4, max_shift=0).eval()
    generator = torch.Generator().manual_seed(123)
    a = 10.0 * torch.randn(16, 8, 24, 24, generator=generator)
    b = 10.0 * torch.randn(16, 8, 24, 24, generator=generator)
    gate, coherence, typicality, dominance, *_ = module._guard_terms(a, b)
    assert torch.all(typicality > module.typicality_tau)
    assert torch.all(dominance < module.dominance_tau)
    assert torch.all(gate == 0)
    assert torch.isfinite(coherence).all()


def test_forward_backward_and_diagnostics_are_finite():
    module = DCSPF(8, groups=2, anchors=4, max_shift=1).train()
    rgb = torch.randn(2, 8, 24, 24, requires_grad=True)
    ir = torch.randn(2, 8, 24, 24, requires_grad=True)
    output, diagnostics = module.forward_with_diagnostics([rgb, ir])
    assert output.shape == (2, 16, 24, 24)
    assert set((0.0, 1.0)).issuperset(set(diagnostics['gate'].unique().tolist()))
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(rgb.grad).all() and torch.isfinite(ir.grad).all()
