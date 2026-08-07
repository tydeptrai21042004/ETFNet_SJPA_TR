import torch
from ultra_modeling.nn.modules.evidence_baselines import FUSION_BASELINES, EvidenceFusionModel


def kwargs(ch=3):
    return dict(input_channels=ch,widths=(16,24,32,40),group_width=4,stat_grid=4,trust_radius=.05,relation_kernel=3)


def test_all_evidence_baselines_single_and_three_inputs():
    for name in FUSION_BASELINES:
        model=EvidenceFusionModel(name,'classify',num_classes=3,backbone_kwargs=kwargs(3)).eval()
        one=model(torch.randn(2,3,32,32)); three=model([torch.randn(2,3,32,32) for _ in range(3)])
        assert one.shape==three.shape==(2,3)
        assert torch.isfinite(one).all() and torch.isfinite(three).all()


def test_all_evidence_baselines_segment_backward():
    for name in FUSION_BASELINES:
        model=EvidenceFusionModel(name,'segment',num_classes=2,backbone_kwargs=kwargs(1),head_channels=12).train()
        x=torch.randn(2,1,32,28); out=model(x); loss=out.square().mean(); loss.backward()
        assert out.shape==(2,2,32,28)
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
