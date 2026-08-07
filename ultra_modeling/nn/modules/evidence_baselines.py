"""Task-neutral variable-cardinality evidence-fusion baselines for CERC studies.

These modules are intentionally simple, transparent comparators. They all accept
one or more aligned BCHW evidence tensors and return the same interface as
ConvolutionalCERC: updated evidence tuple, consensus tensor, diagnostics.
No baseline depends on a dataset, task, or fixed evidence count.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch
import torch.nn as nn

from .cerc import CERCBackbone, CERCModel, _as_tensor_tuple
from .generic_ccprf import ClassificationHead, SegmentationHead, DenseDetectionHead, AnomalyFeatureHead


class MeanEvidenceFusion(nn.Module):
    def __init__(self, channels: int, **_: Any) -> None:
        super().__init__(); self.channels=int(channels)
    def forward_set_with_diagnostics(self, evidence: Any):
        values=_as_tensor_tuple(evidence)
        if any(v.shape[1] != self.channels for v in values): raise ValueError("channel mismatch")
        consensus=torch.stack(values,1).mean(1)
        w=torch.full((values[0].shape[0], len(values), 1), 1.0/len(values), device=consensus.device, dtype=consensus.dtype)
        return values, consensus, {"evidence_count":len(values),"consensus_weights":w,"finite":bool(torch.isfinite(consensus).all())}
    def forward(self,evidence:Any): return self.forward_set_with_diagnostics(evidence)[1]


class MaxEvidenceFusion(nn.Module):
    def __init__(self, channels:int, **_:Any)->None:
        super().__init__(); self.channels=int(channels)
    def forward_set_with_diagnostics(self,evidence:Any):
        values=_as_tensor_tuple(evidence)
        if any(v.shape[1] != self.channels for v in values): raise ValueError("channel mismatch")
        consensus=torch.stack(values,1).amax(1)
        return values, consensus, {"evidence_count":len(values),"finite":bool(torch.isfinite(consensus).all())}
    def forward(self,evidence:Any): return self.forward_set_with_diagnostics(evidence)[1]


class EnergySoftmaxFusion(nn.Module):
    """Parameter-free softmax weighting by per-evidence RMS feature energy."""
    def __init__(self, channels:int, temperature:float=1.0, **_:Any)->None:
        super().__init__(); self.channels=int(channels); self.temperature=float(temperature)
    def forward_set_with_diagnostics(self,evidence:Any):
        values=_as_tensor_tuple(evidence); stack=torch.stack(values,1)
        score=stack.float().pow(2).mean(dim=(2,3,4)).sqrt()
        weights=torch.softmax(score/self.temperature, dim=1).to(stack.dtype)
        consensus=(stack*weights[:,:,None,None,None]).sum(1)
        return values, consensus, {"evidence_count":len(values),"consensus_weights":weights,"finite":bool(torch.isfinite(consensus).all())}
    def forward(self,evidence:Any): return self.forward_set_with_diagnostics(evidence)[1]


class LearnedGateFusion(nn.Module):
    """Shared evidence scorer followed by softmax pooling; parameter count independent of M."""
    def __init__(self, channels:int, **_:Any)->None:
        super().__init__(); self.channels=int(channels)
        hidden=max(4, channels//8)
        self.scorer=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(channels,hidden), nn.SiLU(), nn.Linear(hidden,1))
    def forward_set_with_diagnostics(self,evidence:Any):
        values=_as_tensor_tuple(evidence); stack=torch.stack(values,1)
        scores=torch.stack([self.scorer(v).squeeze(-1) for v in values],1)
        weights=torch.softmax(scores,1).to(stack.dtype)
        consensus=(stack*weights[:,:,None,None,None]).sum(1)
        return values, consensus, {"evidence_count":len(values),"consensus_weights":weights,"finite":bool(torch.isfinite(consensus).all())}
    def forward(self,evidence:Any): return self.forward_set_with_diagnostics(evidence)[1]


class DeepSetEvidenceFusion(nn.Module):
    """Shared residual transform + mean set pooling, a DeepSets-style comparator."""
    def __init__(self, channels:int, **_:Any)->None:
        super().__init__(); self.channels=int(channels)
        self.phi=nn.Sequential(nn.Conv2d(channels,channels,1,bias=False), nn.GroupNorm(1,channels), nn.SiLU(), nn.Conv2d(channels,channels,1,bias=False))
        nn.init.zeros_(self.phi[-1].weight)
    def forward_set_with_diagnostics(self,evidence:Any):
        values=_as_tensor_tuple(evidence)
        updated=tuple(v+self.phi(v) for v in values)
        consensus=torch.stack(updated,1).mean(1)
        return updated, consensus, {"evidence_count":len(values),"finite":bool(torch.isfinite(consensus).all())}
    def forward(self,evidence:Any): return self.forward_set_with_diagnostics(evidence)[1]


class MedianEvidenceFusion(nn.Module):
    """Parameter-free element-wise median over the available evidence set."""
    def __init__(self, channels:int, **_:Any)->None:
        super().__init__(); self.channels=int(channels)
    def forward_set_with_diagnostics(self,evidence:Any):
        values=_as_tensor_tuple(evidence); stack=torch.stack(values,1)
        if any(v.shape[1] != self.channels for v in values): raise ValueError("channel mismatch")
        consensus=stack.median(dim=1).values
        return values, consensus, {"evidence_count":len(values),"finite":bool(torch.isfinite(consensus).all())}
    def forward(self,evidence:Any): return self.forward_set_with_diagnostics(evidence)[1]


class SmoothMaxEvidenceFusion(nn.Module):
    """Parameter-free value-attention pooling, a differentiable max control."""
    def __init__(self, channels:int, beta:float=4.0, **_:Any)->None:
        super().__init__(); self.channels=int(channels); self.beta=float(beta)
    def forward_set_with_diagnostics(self,evidence:Any):
        values=_as_tensor_tuple(evidence); stack=torch.stack(values,1)
        if any(v.shape[1] != self.channels for v in values): raise ValueError("channel mismatch")
        centered=stack.float()-stack.float().mean(dim=1,keepdim=True)
        scale=torch.sqrt(centered.square().mean(dim=1,keepdim=True)+1e-8)
        weights=torch.softmax(self.beta*centered/scale,dim=1).to(stack.dtype)
        consensus=(stack*weights).sum(dim=1)
        return values, consensus, {"evidence_count":len(values),"consensus_weights":weights,"finite":bool(torch.isfinite(consensus).all())}
    def forward(self,evidence:Any): return self.forward_set_with_diagnostics(evidence)[1]


class SetAttentionEvidenceFusion(nn.Module):
    """Shared per-location self-attention over a variable evidence set.

    This is an engineering set-attention comparator, not a claim of reproducing
    the complete Set Transformer architecture.
    """
    def __init__(self, channels:int, heads:int=1, **_:Any)->None:
        super().__init__(); self.channels=int(channels)
        heads=max(1,min(int(heads),self.channels))
        while self.channels % heads: heads-=1
        self.norm=nn.LayerNorm(self.channels)
        self.attn=nn.MultiheadAttention(self.channels,heads,batch_first=True)
        self.out=nn.Linear(self.channels,1,bias=False)
    def forward_set_with_diagnostics(self,evidence:Any):
        values=_as_tensor_tuple(evidence); stack=torch.stack(values,1)
        if any(v.shape[1] != self.channels for v in values): raise ValueError("channel mismatch")
        b,m,c,h,w=stack.shape
        tokens=stack.permute(0,3,4,1,2).reshape(b*h*w,m,c)
        q=self.norm(tokens); attended,_=self.attn(q,q,q,need_weights=False)
        scores=self.out(attended).squeeze(-1)
        weights=torch.softmax(scores,dim=1)
        fused=(tokens*weights.unsqueeze(-1)).sum(dim=1).reshape(b,h,w,c).permute(0,3,1,2)
        return values, fused, {"evidence_count":m,"consensus_weights":weights.reshape(b,h,w,m).permute(0,3,1,2),"finite":bool(torch.isfinite(fused).all())}
    def forward(self,evidence:Any): return self.forward_set_with_diagnostics(evidence)[1]


FUSION_BASELINES={
    "mean": MeanEvidenceFusion,
    "max": MaxEvidenceFusion,
    "energy": EnergySoftmaxFusion,
    "gate": LearnedGateFusion,
    "deepset": DeepSetEvidenceFusion,
    "median": MedianEvidenceFusion,
    "smoothmax": SmoothMaxEvidenceFusion,
    "set_attention": SetAttentionEvidenceFusion,
}


class EvidenceFusionBackbone(CERCBackbone):
    """Same CERC backbone topology with a supplied evidence-fusion comparator."""
    def __init__(self, *args:Any, fusion_cls:Callable[...,nn.Module], **kwargs:Any)->None:
        super().__init__(*args, **kwargs)
        # Replace only relation operators. Stems/stages remain exactly identical.
        self.relations=nn.ModuleDict({name:fusion_cls(channels=width) for name,width in zip(self.stage_names,self.widths)})


class EvidenceFusionModel(nn.Module):
    def __init__(self, fusion:str, task:str, num_classes:int|None=None, backbone_kwargs:Mapping[str,Any]|None=None, head_channels:int=64, anomaly_embedding_channels:int=32)->None:
        super().__init__(); self.task=str(task).lower()
        if fusion not in FUSION_BASELINES: raise ValueError(f"unknown fusion baseline {fusion!r}")
        self.backbone=EvidenceFusionBackbone(fusion_cls=FUSION_BASELINES[fusion], **dict(backbone_kwargs or {}))
        fc=dict(zip(self.backbone.stage_names,self.backbone.widths))
        if self.task=="classify":
            if num_classes is None: raise ValueError("classification requires num_classes")
            self.head=ClassificationHead(self.backbone.widths[-1],num_classes)
        elif self.task=="segment":
            if num_classes is None: raise ValueError("segmentation requires num_classes")
            self.head=SegmentationHead(fc,num_classes,head_channels)
        elif self.task=="detect":
            if num_classes is None: raise ValueError("detection requires num_classes")
            self.head=DenseDetectionHead(fc,num_classes,head_channels)
        elif self.task=="anomaly": self.head=AnomalyFeatureHead(fc,anomaly_embedding_channels)
        else: raise ValueError("unsupported task")
    def forward_features(self,inputs:Any): return self.backbone(inputs)
    def forward(self,inputs:Any):
        features=self.forward_features(inputs)
        if self.task=="segment":
            values=_as_tensor_tuple(inputs); return self.head(features, tuple(values[0].shape[-2:]))
        return self.head(features)
