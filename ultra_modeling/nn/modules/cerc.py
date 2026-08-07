"""Convolutional Canonical Evidence Relational Consensus (CERC).

CERC is a task- and dataset-neutral relation operator over one or more aligned
2-D evidence fields. A single image is *not* treated as a special case: every
feature tensor is partitioned into latent channel atoms and the exact same
canonical-relation equations are applied for M >= 1 evidence fields.

The operator is permutation equivariant with respect to evidence-field order,
uses a shared zero-initialized spatial convolution for relational innovation,
and enforces a sample/atom-wise trust radius. The parameter count of the CERC
innovation convolution is independent of the number of evidence fields.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generic_ccprf import (
    AnomalyFeatureHead,
    ClassificationHead,
    DenseDetectionHead,
    SegmentationHead,
    UniversalConvNormAct,
    UniversalResidualDownBlock,
)


def _as_tensor_tuple(evidence: Any) -> tuple[torch.Tensor, ...]:
    """Normalize a tensor/container into a non-empty ordered evidence tuple."""
    if torch.is_tensor(evidence):
        values = (evidence,)
    elif isinstance(evidence, Mapping):
        if not evidence:
            raise ValueError("evidence mapping must not be empty")
        values = tuple(evidence[key] for key in evidence)
    elif isinstance(evidence, Sequence):
        if not evidence:
            raise ValueError("evidence sequence must not be empty")
        values = tuple(evidence)
    else:
        raise ValueError("evidence must be a BCHW tensor, sequence, or mapping")

    reference = values[0]
    if not torch.is_tensor(reference) or reference.ndim != 4:
        raise ValueError("every evidence field must be a BCHW tensor")
    for value in values[1:]:
        if not torch.is_tensor(value) or value.ndim != 4:
            raise ValueError("every evidence field must be a BCHW tensor")
        if value.shape[0] != reference.shape[0] or value.shape[-2:] != reference.shape[-2:]:
            raise ValueError("evidence fields must share batch and spatial dimensions")
    return values


class UnifiedEvidenceAdapter(nn.Module):
    """Project one or more raw evidence fields into a shared feature width.

    ``input_channels`` may be:
      * int: all evidence fields use the same shared stem;
      * sequence: fixed ordered fields use one stem per position;
      * mapping: named fields use one stem per name and any non-empty configured
        subset may be supplied at runtime.

    Container parsing is deliberately kept outside CERC mathematics. Once
    projected, all evidence fields enter the exact same CERC equations.
    """

    def __init__(
        self,
        input_channels: int | Sequence[int] | Mapping[str, int] = 3,
        out_channels: int = 32,
    ) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        self.out_channels = int(out_channels)
        self.shared_mode = isinstance(input_channels, int)
        self.mapping_mode = isinstance(input_channels, Mapping)

        if self.shared_mode:
            channels = int(input_channels)
            if channels <= 0:
                raise ValueError("input_channels must be positive")
            self.shared_channels = channels
            self.shared_stem = UniversalConvNormAct(channels, self.out_channels, 3, 2)
            self.names: tuple[str, ...] = ()
            self.channels: dict[str, int] = {}
            self.stems = nn.ModuleDict()
            return

        self.shared_channels = 0
        self.shared_stem = None
        if self.mapping_mode:
            items = [(str(name), int(channels)) for name, channels in input_channels.items()]
        else:
            items = [(str(index), int(channels)) for index, channels in enumerate(input_channels)]
        if not items or any(channels <= 0 for _, channels in items):
            raise ValueError("input channel specification must contain positive values")
        self.names = tuple(name for name, _ in items)
        self.channels = dict(items)
        self.stems = nn.ModuleDict(
            {
                name: UniversalConvNormAct(channels, self.out_channels, 3, 2)
                for name, channels in items
            }
        )

    @staticmethod
    def _check_geometry(values: Sequence[torch.Tensor]) -> None:
        _as_tensor_tuple(values)

    def forward(self, evidence: Any) -> tuple[torch.Tensor, ...]:
        if self.shared_mode:
            raw = _as_tensor_tuple(evidence)
            for value in raw:
                if value.shape[1] != self.shared_channels:
                    raise ValueError(
                        f"shared evidence stem expects {self.shared_channels} channels, "
                        f"got {value.shape[1]}"
                    )
            return tuple(self.shared_stem(value) for value in raw)

        if self.mapping_mode:
            if not isinstance(evidence, Mapping) or not evidence:
                raise ValueError("named evidence configuration expects a non-empty mapping")
            unknown = set(evidence).difference(self.stems)
            if unknown:
                raise ValueError(f"unknown evidence names: {sorted(unknown)}")
            names = tuple(name for name in self.names if name in evidence)
            raw = tuple(evidence[name] for name in names)
        else:
            if torch.is_tensor(evidence) or not isinstance(evidence, Sequence):
                raise ValueError("ordered evidence configuration expects a tensor sequence")
            if len(evidence) != len(self.names):
                raise ValueError(f"expected {len(self.names)} evidence fields, got {len(evidence)}")
            names = self.names
            raw = tuple(evidence)
        self._check_geometry(raw)
        projected = []
        for name, value in zip(names, raw):
            expected = self.channels[name]
            if value.shape[1] != expected:
                raise ValueError(f"evidence {name!r} expects {expected} channels, got {value.shape[1]}")
            projected.append(self.stems[name](value))
        return tuple(projected)


class ConvolutionalCERC(nn.Module):
    r"""Canonical Evidence Relational Consensus with shared spatial convolution.

    Let each aligned evidence field be split into G channel atoms. After
    scale-normalized whitening, for atoms a,b define

        C_ab = Z_a^T Z_b / (N - 1),
        A_ab = C_ab C_ab^T,
        s_ab = tr(A_ab)/d / (1 + tr(A_ab)/d).

    Self relations are removed algebraically by an off-diagonal mask, not by a
    single/multi-view branch. The canonical relational message can be written
    without materializing pairwise spatial predictions:

        E_a = [sum_b s_ab Z_b (C_ab^T A_ab)
               - Z_a sum_b s_ab A_ab] / [eps + sum_b s_ab].

    A *single shared* zero-initialized kxk convolution maps the recolored E_a
    into a local correction. Hence the module is exact identity on every atom
    at initialization for any M >= 1, while gradients reach the convolution on
    the first optimization step. A trust projection enforces

        ||Delta_a||_F <= rho ||X_a||_F.

    Consensus pooling uses the same relation supports. For each channel group g
    it softmax-normalizes relational centrality over available evidence fields;
    when M=1 the weight is exactly one by the same equation.
    """

    def __init__(
        self,
        channels: int,
        group_width: int = 8,
        stat_grid: int = 8,
        eps: float = 1e-5,
        ridge: float = 1e-3,
        trust_radius: float = 0.05,
        kernel_size: int = 3,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or group_width <= 0:
            raise ValueError("channels and group_width must be positive")
        if channels % group_width != 0:
            raise ValueError("group_width must divide channels")
        if stat_grid <= 1:
            raise ValueError("stat_grid must be > 1")
        if not (0.0 < trust_radius <= 1.0):
            raise ValueError("trust_radius must lie in (0, 1]")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.channels = int(channels)
        self.group_width = int(group_width)
        self.groups = self.channels // self.group_width
        self.stat_grid = int(stat_grid)
        self.eps = float(eps)
        self.ridge = float(ridge)
        self.trust_radius = float(trust_radius)
        self.temperature = float(temperature)

        # One convolution shared by every latent atom and every evidence field.
        # Zero initialization gives exact identity atom updates at initialization
        # while preserving nonzero first-step gradients for these weights.
        self.transport = nn.Conv2d(
            self.group_width,
            self.group_width,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        nn.init.zeros_(self.transport.weight)

    @property
    def innovation_parameters(self) -> int:
        return int(self.transport.weight.numel())

    @staticmethod
    def _sample_norm(x: torch.Tensor) -> torch.Tensor:
        return x.float().flatten(2).norm(dim=2, keepdim=True).view(x.shape[0], x.shape[1], 1, 1, 1)

    def _stack_atoms(self, evidence: Any) -> tuple[torch.Tensor, int, int, int, int]:
        values = _as_tensor_tuple(evidence)
        reference = values[0]
        for value in values:
            if value.shape != reference.shape:
                raise ValueError("CERC evidence features must have identical BCHW shapes")
            if value.shape[1] != self.channels:
                raise ValueError(f"expected {self.channels} feature channels, got {value.shape[1]}")
        stacked = torch.stack(values, dim=1)  # B,M,C,H,W
        b, m, _, h, w = stacked.shape
        atoms = stacked.reshape(b, m, self.groups, self.group_width, h, w)
        atoms = atoms.reshape(b, m * self.groups, self.group_width, h, w)
        return atoms, m, b, h, w

    def _cholesky(self, covariance: torch.Tensor) -> torch.Tensor:
        d = covariance.shape[-1]
        eye = torch.eye(d, device=covariance.device, dtype=covariance.dtype)
        symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
        trace_scale = symmetric.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(0.0)
        base = self.eps + self.ridge * (1.0 + trace_scale)

        factor = None
        info = None
        for multiplier in (1.0, 10.0, 100.0, 1000.0):
            matrix = symmetric + (base * multiplier)[..., None, None] * eye
            candidate, candidate_info = torch.linalg.cholesky_ex(matrix)
            if factor is None:
                factor, info = candidate, candidate_info
            else:
                failed = info.ne(0)
                factor = torch.where(failed[..., None, None], candidate, factor)
                info = torch.where(failed, candidate_info, info)
            if not torch.any(info.ne(0)):
                break

        if torch.any(info.ne(0)):
            matrix = symmetric + (base * 1000.0)[..., None, None] * eye
            diagonal = matrix.diagonal(dim1=-2, dim2=-1).clamp_min(self.eps).sqrt()
            fallback = torch.diag_embed(diagonal)
            factor = torch.where(info.ne(0)[..., None, None], fallback, factor)
        return factor

    def _statistics(self, atoms: torch.Tensor):
        b, k, d, h, w = atoms.shape
        work = atoms.float().reshape(b * k, d, h, w)
        gh = min(self.stat_grid, h)
        gw = min(self.stat_grid, w)
        pooled = F.adaptive_avg_pool2d(work, (gh, gw))
        tokens = pooled.flatten(2).transpose(1, 2).reshape(b, k, gh * gw, d)
        mean = tokens.mean(dim=2, keepdim=True)
        centered = tokens - mean
        n = centered.shape[2]
        rms = torch.linalg.vector_norm(centered, dim=(-2, -1), keepdim=True)
        rms = rms / math.sqrt(max(1, n * d))
        rms = rms.clamp_min(self.eps)
        normalized = centered / rms
        denominator = float(max(1, n - 1))
        covariance = torch.einsum("bknd,bkne->bkde", normalized, normalized) / denominator
        factor = self._cholesky(covariance)
        whitened = torch.linalg.solve_triangular(
            factor,
            normalized.transpose(-1, -2),
            upper=False,
        ).transpose(-1, -2)
        return mean, rms, factor, whitened, denominator

    def _full_whitened(
        self,
        atoms: torch.Tensor,
        mean: torch.Tensor,
        rms: torch.Tensor,
        factor: torch.Tensor,
    ) -> torch.Tensor:
        b, k, d, h, w = atoms.shape
        full = atoms.float().flatten(3).transpose(-1, -2)  # B,K,HW,d
        normalized = (full - mean) / rms
        return torch.linalg.solve_triangular(
            factor,
            normalized.transpose(-1, -2),
            upper=False,
        ).transpose(-1, -2)

    def _relations(self, whitened_stats: torch.Tensor, denominator: float):
        # C[a,b] = Z_a^T Z_b/(N-1), shape B,K,K,d,d.
        cross = torch.einsum(
            "bknd,blne->bklde", whitened_stats, whitened_stats
        ) / denominator
        reliability = cross @ cross.transpose(-1, -2)
        energy = reliability.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(0.0)
        support = energy / (1.0 + energy)

        k = support.shape[-1]
        offdiag = 1.0 - torch.eye(k, device=support.device, dtype=support.dtype)
        support = support * offdiag.unsqueeze(0)
        return cross, reliability, support

    def _relational_innovation(
        self,
        full_z: torch.Tensor,
        cross: torch.Tensor,
        reliability: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        # (C_ab^T A_ab), followed by weighted source transport.
        transport_matrix = cross.transpose(-1, -2) @ reliability
        weighted_transport = support[..., None, None] * transport_matrix
        weighted_self = (support[..., None, None] * reliability).sum(dim=2)

        source_term = torch.einsum(
            "blne,bkled->bknd", full_z, weighted_transport
        )
        self_term = torch.einsum("bkne,bked->bknd", full_z, weighted_self)
        denominator = self.eps + support.sum(dim=2, keepdim=True)[..., None]
        return (source_term - self_term) / denominator

    def forward_set_with_diagnostics(self, evidence: Any):
        atoms, m, b, h, w = self._stack_atoms(evidence)
        k = atoms.shape[1]
        mean, rms, factor, z_stats, stat_denominator = self._statistics(atoms)
        full_z = self._full_whitened(atoms, mean, rms, factor)
        cross, reliability, support = self._relations(z_stats, stat_denominator)
        innovation_z = self._relational_innovation(full_z, cross, reliability, support)

        # Recolor into each atom's native scale: X_centered ~= Z L^T * rms.
        recolored = torch.einsum("bkne,bkde->bknd", innovation_z, factor)
        recolored = recolored * rms
        recolored = recolored.transpose(-1, -2).reshape(
            b * k, self.group_width, h, w
        )
        correction = self.transport(recolored).reshape(
            b, k, self.group_width, h, w
        )

        raw_norm = self._sample_norm(atoms)
        correction_norm = self._sample_norm(correction)
        radius = self.trust_radius * raw_norm
        scale = torch.minimum(
            torch.ones_like(correction_norm),
            radius / (correction_norm + self.eps),
        ).to(correction.dtype)
        updated = atoms + scale * correction.to(atoms.dtype)

        updated_views = updated.reshape(
            b, m, self.groups, self.group_width, h, w
        )

        # Relation-centrality pooling. The exact same softmax equation gives
        # weight one when M=1 and re-normalizes over any available evidence set.
        centrality = support.sum(dim=2) / float(max(1, k - 1))
        centrality = centrality.reshape(b, m, self.groups)
        weights = torch.softmax(centrality / self.temperature, dim=1).to(updated.dtype)
        consensus_groups = (
            updated_views * weights[..., None, None, None]
        ).sum(dim=1)
        consensus = consensus_groups.reshape(b, self.channels, h, w)

        view_tuple = tuple(updated_views[:, index].reshape(b, self.channels, h, w) for index in range(m))
        diagnostics = {
            "evidence_count": m,
            "group_count": self.groups,
            "atom_count": k,
            "support": support,
            "centrality": centrality,
            "consensus_weights": weights,
            "correction_norm": correction_norm,
            "trust_radius_norm": radius,
            "trust_scale": scale,
            "innovation_parameter_count": self.innovation_parameters,
            "finite": bool(
                torch.isfinite(consensus).all()
                and torch.isfinite(support).all()
                and torch.isfinite(scale).all()
            ),
        }
        return view_tuple, consensus, diagnostics

    def forward_with_diagnostics(self, evidence: Any):
        _, consensus, diagnostics = self.forward_set_with_diagnostics(evidence)
        return consensus, diagnostics

    def forward(self, evidence: Any) -> torch.Tensor:
        return self.forward_with_diagnostics(evidence)[0]


class CERCBackbone(nn.Module):
    """Shared-stream P2--P5 backbone using the same CERC equation at every stage."""

    stage_names = ("P2", "P3", "P4", "P5")

    def __init__(
        self,
        input_channels: int | Sequence[int] | Mapping[str, int] = 3,
        widths: Sequence[int] = (64, 128, 256, 512),
        group_width: int = 8,
        stat_grid: int = 8,
        trust_radius: float = 0.05,
        relation_kernel: int = 3,
    ) -> None:
        super().__init__()
        if len(widths) != 4 or any(int(width) <= 0 for width in widths):
            raise ValueError("widths must contain four positive values")
        self.widths = tuple(int(width) for width in widths)
        if any(width % group_width != 0 for width in self.widths):
            raise ValueError("group_width must divide every backbone width")

        stem_width = max(8, self.widths[0] // 2)
        self.adapter = UnifiedEvidenceAdapter(input_channels, stem_width)
        channels_in = (stem_width, *self.widths[:-1])
        self.stages = nn.ModuleDict(
            {
                name: UniversalResidualDownBlock(cin, cout)
                for name, cin, cout in zip(self.stage_names, channels_in, self.widths)
            }
        )
        self.relations = nn.ModuleDict(
            {
                name: ConvolutionalCERC(
                    channels=width,
                    group_width=group_width,
                    stat_grid=stat_grid,
                    trust_radius=trust_radius,
                    kernel_size=relation_kernel,
                )
                for name, width in zip(self.stage_names, self.widths)
            }
        )

    def forward_with_diagnostics(self, inputs: Any):
        evidence = tuple(self.adapter(inputs))
        pyramid: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, Any] = {
            "input_evidence_count": len(evidence),
            "stages": {},
        }
        for name in self.stage_names:
            evidence = tuple(self.stages[name](value) for value in evidence)
            evidence, consensus, relation_diag = self.relations[name].forward_set_with_diagnostics(evidence)
            pyramid[name] = consensus
            diagnostics["stages"][name] = relation_diag
        return pyramid, diagnostics

    def forward(self, inputs: Any) -> dict[str, torch.Tensor]:
        return self.forward_with_diagnostics(inputs)[0]


class CERCModel(nn.Module):
    """CERC backbone with interchangeable 2-D vision heads.

    The task head is outside the CERC relation mathematics. Supported heads are
    classification, segmentation, dense detection, and anomaly embeddings.
    """

    def __init__(
        self,
        task: str,
        num_classes: int | None = None,
        backbone_kwargs: Mapping[str, Any] | None = None,
        head_channels: int = 64,
        anomaly_embedding_channels: int = 32,
    ) -> None:
        super().__init__()
        self.task = str(task).lower()
        self.backbone = CERCBackbone(**dict(backbone_kwargs or {}))
        feature_channels = dict(zip(self.backbone.stage_names, self.backbone.widths))
        if self.task == "classify":
            if num_classes is None:
                raise ValueError("classification requires num_classes")
            self.head = ClassificationHead(self.backbone.widths[-1], num_classes)
        elif self.task == "segment":
            if num_classes is None:
                raise ValueError("segmentation requires num_classes")
            self.head = SegmentationHead(feature_channels, num_classes, head_channels)
        elif self.task == "detect":
            if num_classes is None:
                raise ValueError("detection requires num_classes")
            self.head = DenseDetectionHead(feature_channels, num_classes, head_channels)
        elif self.task == "anomaly":
            self.head = AnomalyFeatureHead(feature_channels, anomaly_embedding_channels)
        else:
            raise ValueError("task must be classify, segment, detect, or anomaly")

    @staticmethod
    def _spatial_size(inputs: Any) -> tuple[int, int]:
        values = _as_tensor_tuple(inputs)
        return tuple(int(value) for value in values[0].shape[-2:])

    def forward_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        return self.backbone(inputs)

    def forward(self, inputs: Any):
        output_size = self._spatial_size(inputs)
        features = self.forward_features(inputs)
        if self.task == "segment":
            return self.head(features, output_size)
        return self.head(features)
