from __future__ import annotations
import copy
import torch
import torch.nn.functional as F
from ultra_modeling.nn.modules.cerc import CERCModel, ConvolutionalCERC
from ultra_modeling.nn.modules.evidence_baselines import EvidenceFusionModel, FUSION_BASELINES


def kw(ch):
    return dict(input_channels=ch,widths=(8,12,16,20),group_width=4,stat_grid=4,trust_radius=.05,relation_kernel=3)


def copy_common(src,dst):
    ss=src.state_dict(); ds=dst.state_dict()
    with torch.no_grad():
        for k,v in ds.items():
            if k in ss and ss[k].shape==v.shape:
                v.copy_(ss[k])
    dst.load_state_dict(ds,strict=True)


def test_cerc_constructor_is_rng_neutral():
    torch.manual_seed(1234)
    before=torch.rand(9)
    torch.manual_seed(1234)
    _=ConvolutionalCERC(16,group_width=4,stat_grid=4,kernel_size=3)
    after=torch.rand(9)
    assert torch.equal(before,after)


def test_cerc_is_exact_mean_baseline_at_initialization_for_multiple_evidence():
    torch.manual_seed(11)
    base=EvidenceFusionModel('mean','segment',num_classes=2,backbone_kwargs=kw(1),head_channels=8).eval()
    torch.manual_seed(99)
    cerc=CERCModel('segment',num_classes=2,backbone_kwargs=kw(1),head_channels=8).eval()
    copy_common(base,cerc)
    x=[torch.randn(2,1,32,28),torch.randn(2,1,32,28),torch.randn(2,1,32,28)]
    with torch.no_grad():
        a=base(x); b=cerc(x)
    assert torch.equal(a,b)


def test_cerc_is_exact_single_stream_baseline_at_initialization():
    torch.manual_seed(12)
    base=EvidenceFusionModel('mean','classify',num_classes=3,backbone_kwargs=kw(1)).eval()
    torch.manual_seed(100)
    cerc=CERCModel('classify',num_classes=3,backbone_kwargs=kw(1)).eval()
    copy_common(base,cerc)
    x=torch.randn(3,1,28,28)
    with torch.no_grad(): assert torch.equal(base(x),cerc(x))


def test_local_and_global_quality_terms_receive_gradient_for_multiple_evidence():
    module=ConvolutionalCERC(16,group_width=4,stat_grid=4,kernel_size=3).train()
    x=[torch.randn(2,16,12,12),torch.randn(2,16,12,12),torch.randn(2,16,12,12)]
    target=torch.randn(2,16,12,12)
    loss=F.mse_loss(module(x),target); loss.backward()
    assert module.local_gate.weight.grad is not None and torch.isfinite(module.local_gate.weight.grad).all()
    assert float(module.local_gate.weight.grad.abs().sum())>0
    assert module.global_gate_w2.grad is not None and torch.isfinite(module.global_gate_w2.grad).all()
    assert float(module.global_gate_w2.grad.abs().sum())>0
    assert module.canonical_gain.grad is not None and torch.isfinite(module.canonical_gain.grad)


def test_all_added_baselines_are_variable_cardinality():
    for name in FUSION_BASELINES:
        model=EvidenceFusionModel(name,'classify',num_classes=2,backbone_kwargs=kw(3)).eval()
        one=model(torch.randn(1,3,32,32))
        four=model([torch.randn(1,3,32,32) for _ in range(4)])
        assert one.shape==four.shape==(1,2)
        assert torch.isfinite(one).all() and torch.isfinite(four).all()
