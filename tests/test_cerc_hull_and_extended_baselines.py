from __future__ import annotations
import torch
from ultra_modeling.nn.modules.cerc import ConvolutionalCERC
from ultra_modeling.nn.modules.evidence_baselines import FUSION_BASELINES


def test_cerc_hull_contains_exact_mean_and_max_endpoints():
    torch.manual_seed(5)
    module=ConvolutionalCERC(16,group_width=4,stat_grid=4,kernel_size=3).eval()
    values=[torch.randn(2,16,11,9) for _ in range(3)]
    stack=torch.stack(values,1)
    with torch.no_grad():
        module.envelope_gain.zero_()
        mean_out=module(values)
        module.envelope_gain.fill_(1.0)
        max_out=module(values)
    assert torch.equal(mean_out,stack.mean(1))
    assert torch.allclose(max_out,stack.amax(1),atol=3e-7,rtol=3e-7)


def test_cerc_hull_single_evidence_is_identity_for_any_envelope_coordinate():
    module=ConvolutionalCERC(16,group_width=4,stat_grid=4,kernel_size=3).eval()
    x=torch.randn(2,16,10,12)
    with torch.no_grad():
        module.envelope_gain.fill_(0.73)
        assert torch.equal(module(x),x)


def test_envelope_coordinate_receives_gradient_at_mean_endpoint():
    module=ConvolutionalCERC(16,group_width=4,stat_grid=4,kernel_size=3).train()
    values=[torch.randn(2,16,9,9) for _ in range(3)]
    loss=module(values).square().mean(); loss.backward()
    assert module.envelope_gain.grad is not None
    assert torch.isfinite(module.envelope_gain.grad).all()
    assert float(module.envelope_gain.grad.abs().sum())>0


def test_extended_baseline_pack_has_eight_variable_cardinality_controls():
    assert set(FUSION_BASELINES) == {"mean","max","energy","gate","deepset","median","smoothmax","set_attention"}
    for name,cls in FUSION_BASELINES.items():
        module=cls(channels=8).eval()
        one=torch.randn(2,8,7,9)
        many=[torch.randn(2,8,7,9) for _ in range(4)]
        assert module(one).shape==(2,8,7,9)
        assert module(many).shape==(2,8,7,9)
        assert torch.isfinite(module(one)).all() and torch.isfinite(module(many)).all()
