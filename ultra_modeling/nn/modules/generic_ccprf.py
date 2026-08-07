"""Task-agnostic universal CCPRF components.

This module deliberately contains no dataset names, class mappings, oriented-box
assumptions, or application-specific geometry. It provides:

* adapters for direct single images, self-complementary decompositions, or one-to-many physical views;
* a generic view-named wrapper around CCPRF;
* optional local-global statistics using shared CCPRF parameters;
* backward-compatible dual-stream and universal one-to-many P2--P5 backbones; and
* interchangeable classification, segmentation, dense-detection, and
  anomaly-feature heads.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import CCPRF


class ConvNormAct(nn.Module):
    """Small convolution block used by the generic backbone and heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("groups must divide both input and output channels")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualDownBlock(nn.Module):
    """Stride-two stage with a residual refinement."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = ConvNormAct(in_channels, out_channels, 3, 2)
        self.refine1 = ConvNormAct(out_channels, out_channels, 3, 1)
        self.refine2 = ConvNormAct(out_channels, out_channels, 3, 1, activation=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        return self.act(y + self.refine2(self.refine1(y)))


class AppearanceTextureAdapter(nn.Module):
    """Create complementary appearance and high-frequency texture views.

    The low-pass kernel is constrained to be non-negative and to sum to one by
    a softmax parameterization. The texture view is the exact residual
    ``image - lowpass(image)``, so no image information is discarded by the
    decomposition itself.
    """

    def __init__(
        self,
        input_channels: int = 3,
        out_channels: int = 32,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or out_channels <= 0:
            raise ValueError("input_channels and out_channels must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.raw_kernel = nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size))
        self.appearance_stem = ConvNormAct(input_channels, out_channels, 3, 2)
        self.texture_stem = ConvNormAct(input_channels, out_channels, 3, 2)

    def normalized_kernel(self) -> torch.Tensor:
        kernel = torch.softmax(self.raw_kernel.flatten(), dim=0)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def decompose(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != self.input_channels:
            raise ValueError(
                f"expected BCHW input with {self.input_channels} channels, got {tuple(image.shape)}"
            )
        pad = self.kernel_size // 2
        mode = "reflect" if min(image.shape[-2:]) > pad else "replicate"
        padded = F.pad(image, (pad, pad, pad, pad), mode=mode)
        kernel = self.normalized_kernel().to(device=image.device, dtype=image.dtype)
        kernel = kernel.expand(self.input_channels, 1, -1, -1)
        low = F.conv2d(padded, kernel, groups=self.input_channels)
        high = image - low
        return low, high

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, texture = self.decompose(image)
        return self.appearance_stem(image), self.texture_stem(texture)


class PairedViewAdapter(nn.Module):
    """Project two physical input views to a common feature width."""

    def __init__(
        self,
        view_a_channels: int,
        view_b_channels: int,
        out_channels: int = 32,
    ) -> None:
        super().__init__()
        self.view_a_channels = int(view_a_channels)
        self.view_b_channels = int(view_b_channels)
        if self.view_a_channels <= 0 or self.view_b_channels <= 0:
            raise ValueError("view channel counts must be positive")
        self.stem_a = ConvNormAct(self.view_a_channels, out_channels, 3, 2)
        self.stem_b = ConvNormAct(self.view_b_channels, out_channels, 3, 2)

    def forward(
        self, views: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(views, Sequence) or len(views) != 2:
            raise ValueError("PairedViewAdapter expects exactly two tensors")
        view_a, view_b = views
        if view_a.ndim != 4 or view_b.ndim != 4:
            raise ValueError("both views must be BCHW tensors")
        if view_a.shape[0] != view_b.shape[0] or view_a.shape[-2:] != view_b.shape[-2:]:
            raise ValueError("paired views must share batch and spatial dimensions")
        if view_a.shape[1] != self.view_a_channels or view_b.shape[1] != self.view_b_channels:
            raise ValueError(
                "paired view channel counts do not match adapter configuration"
            )
        return self.stem_a(view_a), self.stem_b(view_b)


class GenericCCPRF(CCPRF):
    """View-named, task-agnostic interface for CCPRF v10.3.

    The inherited mathematics is independent of modality semantics. This class
    exposes neutral diagnostic names while retaining compatibility aliases for
    existing checkpoints and tests.
    """

    def _forward_impl(self, x: Sequence[torch.Tensor]):
        if not isinstance(x, Sequence) or len(x) != 2:
            raise ValueError("GenericCCPRF expects exactly two aligned feature views")
        view_a, view_b = x
        if view_a.shape != view_b.shape:
            raise ValueError(
                f"GenericCCPRF expects matching shapes, got {view_a.shape} and {view_b.shape}"
            )
        output, diagnostics = super()._forward_impl((view_a, view_b))
        generic = {
            "raw": diagnostics["raw"],
            "view_a_basis": diagnostics["rgb_basis"],
            "view_b_basis": diagnostics["ir_basis"],
            "grouped_basis": diagnostics["grouped_basis"],
            "cross_prediction": diagnostics["cross_prediction"],
            "coherence": diagnostics["coherence"],
            "view_a_reliability": diagnostics["rgb_reliability"],
            "view_b_reliability": diagnostics["ir_reliability"],
            "correction": diagnostics["correction"],
            "correction_norm": diagnostics["correction_norm"],
            "trust_radius_norm": diagnostics["trust_radius_norm"],
            "trust_scale": diagnostics["trust_scale"],
        }
        return output, generic


class LocalGlobalCCPRF(nn.Module):
    """Blend global and non-overlapping local CCPRF corrections.

    Both paths share the same CCPRF parameters. The final correction is
    projected once more onto the samplewise trust ball, so the original CCPRF
    norm guarantee remains valid after local/global blending.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 16,
        stat_grid: int = 16,
        eps: float = 1e-3,
        trust_radius: float = 0.05,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        if window_size < 0:
            raise ValueError("window_size must be non-negative")
        self.window_size = int(window_size)
        self.fusion = GenericCCPRF(
            channels=channels,
            groups=groups,
            stat_grid=stat_grid,
            eps=eps,
            trust_radius=trust_radius,
        )
        self.local_logit = nn.Parameter(torch.zeros(()))

    @property
    def channels(self) -> int:
        return self.fusion.channels

    @property
    def trust_radius(self) -> float:
        return self.fusion.trust_radius

    @staticmethod
    def _sample_norm(x: torch.Tensor) -> torch.Tensor:
        return x.float().flatten(1).norm(dim=1).view(-1, 1, 1, 1)

    def _to_windows(
        self, x: torch.Tensor, window_size: int
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        b, c, h, w = x.shape
        hp = math.ceil(h / window_size) * window_size
        wp = math.ceil(w / window_size) * window_size
        padded = F.pad(x, (0, wp - w, 0, hp - h), mode="replicate")
        nh, nw = hp // window_size, wp // window_size
        windows = (
            padded.unfold(2, window_size, window_size)
            .unfold(3, window_size, window_size)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(b * nh * nw, c, window_size, window_size)
        )
        return windows, (h, w, nh, nw)

    @staticmethod
    def _from_windows(
        windows: torch.Tensor,
        metadata: tuple[int, int, int, int],
        batch_size: int,
        window_size: int,
    ) -> torch.Tensor:
        h, w, nh, nw = metadata
        channels = windows.shape[1]
        merged = (
            windows.reshape(batch_size, nh, nw, channels, window_size, window_size)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch_size, channels, nh * window_size, nw * window_size)
        )
        return merged[..., :h, :w]

    def _forward_impl(self, views: Sequence[torch.Tensor]):
        if not isinstance(views, Sequence) or len(views) != 2:
            raise ValueError("LocalGlobalCCPRF expects exactly two feature views")
        view_a, view_b = views
        if view_a.shape != view_b.shape:
            raise ValueError("local-global fusion requires matching feature shapes")
        raw = torch.cat((view_a, view_b), dim=1)
        global_output, global_diag = self.fusion.forward_with_diagnostics((view_a, view_b))
        global_correction = global_output - raw

        if self.window_size <= 1:
            diagnostics = dict(global_diag)
            diagnostics.update(
                {
                    "local_weight": torch.zeros((), device=raw.device, dtype=raw.dtype),
                    "global_weight": torch.ones((), device=raw.device, dtype=raw.dtype),
                    "local_correction": torch.zeros_like(raw),
                }
            )
            return global_output, diagnostics

        window = min(self.window_size, view_a.shape[-2], view_a.shape[-1])
        if window <= 1:
            return global_output, global_diag
        a_windows, metadata = self._to_windows(view_a, window)
        b_windows, metadata_b = self._to_windows(view_b, window)
        if metadata != metadata_b:
            raise RuntimeError("internal local-window metadata mismatch")
        local_output, local_diag = self.fusion.forward_with_diagnostics(
            (a_windows, b_windows)
        )
        local_raw = torch.cat((a_windows, b_windows), dim=1)
        local_correction = self._from_windows(
            local_output - local_raw,
            metadata,
            view_a.shape[0],
            window,
        )

        alpha = torch.sigmoid(self.local_logit).to(raw.dtype)
        correction = alpha * local_correction + (1.0 - alpha) * global_correction
        raw_norm = self._sample_norm(raw)
        correction_norm = self._sample_norm(correction)
        radius = self.trust_radius * raw_norm
        scale = torch.minimum(
            torch.ones_like(correction_norm),
            radius / (correction_norm + self.fusion.eps),
        ).to(correction.dtype)
        output = raw + scale * correction
        diagnostics = {
            "raw": raw,
            "local_weight": alpha,
            "global_weight": 1.0 - alpha,
            "local_correction": local_correction,
            "global_correction": global_correction,
            "correction": correction,
            "correction_norm": correction_norm,
            "trust_radius_norm": radius,
            "trust_scale": scale,
            "global": global_diag,
            "local": local_diag,
        }
        return output, diagnostics

    def forward_with_diagnostics(self, views: Sequence[torch.Tensor]):
        return self._forward_impl(views)

    def forward(self, views: Sequence[torch.Tensor]) -> torch.Tensor:
        return self._forward_impl(views)[0]


class MultiViewCCPRFBackbone(nn.Module):
    """Generic dual-view P2--P5 feature-pyramid backbone."""

    stage_names = ("P2", "P3", "P4", "P5")

    def __init__(
        self,
        input_mode: str = "appearance_texture",
        input_channels: int = 3,
        view_a_channels: int | None = None,
        view_b_channels: int | None = None,
        widths: Sequence[int] = (64, 128, 256, 512),
        fusion_stages: Sequence[str] = ("P2", "P3"),
        group_width: int = 8,
        stat_grid: int = 16,
        trust_radius: Mapping[str, float] | float = 0.05,
        local_windows: Mapping[str, int] | None = None,
        lowpass_kernel: int = 5,
    ) -> None:
        super().__init__()
        if len(widths) != 4 or any(int(width) <= 0 for width in widths):
            raise ValueError("widths must contain four positive channel counts")
        self.widths = tuple(int(width) for width in widths)
        self.input_mode = str(input_mode)
        stem_width = max(8, self.widths[0] // 2)
        if self.input_mode == "appearance_texture":
            self.adapter = AppearanceTextureAdapter(
                input_channels=input_channels,
                out_channels=stem_width,
                kernel_size=lowpass_kernel,
            )
        elif self.input_mode == "paired":
            if view_a_channels is None or view_b_channels is None:
                raise ValueError("paired mode requires view_a_channels and view_b_channels")
            self.adapter = PairedViewAdapter(
                view_a_channels=view_a_channels,
                view_b_channels=view_b_channels,
                out_channels=stem_width,
            )
        else:
            raise ValueError("input_mode must be 'appearance_texture' or 'paired'")

        channels_in = (stem_width, *self.widths[:-1])
        self.stages_a = nn.ModuleDict()
        self.stages_b = nn.ModuleDict()
        for name, cin, cout in zip(self.stage_names, channels_in, self.widths):
            self.stages_a[name] = ResidualDownBlock(cin, cout)
            self.stages_b[name] = ResidualDownBlock(cin, cout)

        selected = {str(stage) for stage in fusion_stages}
        unknown = selected.difference(self.stage_names)
        if unknown:
            raise ValueError(f"unknown fusion stages: {sorted(unknown)}")
        self.fusion_stages = selected
        local_windows = dict(local_windows or {})
        self.fusions = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for name, channels in zip(self.stage_names, self.widths):
            self.projections[name] = ConvNormAct(2 * channels, channels, 1, 1)
            if name in self.fusion_stages:
                groups = max(1, channels // max(1, int(group_width)))
                while channels % groups != 0:
                    groups -= 1
                radius = (
                    float(trust_radius[name])
                    if isinstance(trust_radius, Mapping)
                    else float(trust_radius)
                )
                self.fusions[name] = LocalGlobalCCPRF(
                    channels=channels,
                    groups=groups,
                    stat_grid=stat_grid,
                    trust_radius=radius,
                    window_size=int(local_windows.get(name, 0)),
                )

    def forward(self, inputs: Any) -> dict[str, torch.Tensor]:
        view_a, view_b = self.adapter(inputs)
        pyramid: dict[str, torch.Tensor] = {}
        for name in self.stage_names:
            view_a = self.stages_a[name](view_a)
            view_b = self.stages_b[name](view_b)
            if name in self.fusions:
                paired = self.fusions[name]((view_a, view_b))
                view_a, view_b = paired.chunk(2, dim=1)
            else:
                paired = torch.cat((view_a, view_b), dim=1)
            pyramid[name] = self.projections[name](paired)
        return pyramid


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(features["P5"], 1).flatten(1)
        return self.classifier(self.dropout(x))


class SegmentationHead(nn.Module):
    def __init__(
        self,
        feature_channels: Mapping[str, int],
        num_classes: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.levels = tuple(feature_channels.keys())
        self.lateral = nn.ModuleDict(
            {
                name: ConvNormAct(channels, hidden_channels, 1, 1)
                for name, channels in feature_channels.items()
            }
        )
        self.refine = nn.Sequential(
            ConvNormAct(hidden_channels, hidden_channels, 3, 1),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        target = features["P2"].shape[-2:]
        fused = None
        for name in self.levels:
            value = self.lateral[name](features[name])
            if value.shape[-2:] != target:
                value = F.interpolate(value, target, mode="bilinear", align_corners=False)
            fused = value if fused is None else fused + value
        logits = self.refine(fused)
        return F.interpolate(logits, output_size, mode="bilinear", align_corners=False)


class DenseDetectionHead(nn.Module):
    """Task-neutral dense axis-aligned detection predictions.

    It returns per-level classification logits, positive box distances, and an
    object-quality logit. Decoding and assignment remain training-pipeline
    responsibilities rather than backbone assumptions.
    """

    def __init__(
        self,
        feature_channels: Mapping[str, int],
        num_classes: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.levels = tuple(feature_channels.keys())
        self.stems = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        self.regressors = nn.ModuleDict()
        self.quality = nn.ModuleDict()
        for name, channels in feature_channels.items():
            self.stems[name] = ConvNormAct(channels, hidden_channels, 3, 1)
            self.classifiers[name] = nn.Conv2d(hidden_channels, num_classes, 1)
            self.regressors[name] = nn.Conv2d(hidden_channels, 4, 1)
            self.quality[name] = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, features: Mapping[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for name in self.levels:
            x = self.stems[name](features[name])
            outputs[name] = {
                "class_logits": self.classifiers[name](x),
                "box_distances": F.softplus(self.regressors[name](x)),
                "quality_logits": self.quality[name](x),
            }
        return outputs


class AnomalyFeatureHead(nn.Module):
    """Return a normalized multi-scale patch embedding map."""

    def __init__(
        self,
        feature_channels: Mapping[str, int],
        embedding_channels: int = 32,
    ) -> None:
        super().__init__()
        if embedding_channels <= 0:
            raise ValueError("embedding_channels must be positive")
        self.levels = tuple(feature_channels.keys())
        self.projections = nn.ModuleDict(
            {
                name: nn.Conv2d(channels, embedding_channels, 1, bias=False)
                for name, channels in feature_channels.items()
            }
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        target = features["P2"].shape[-2:]
        embeddings = []
        for name in self.levels:
            value = self.projections[name](features[name])
            if value.shape[-2:] != target:
                value = F.interpolate(value, target, mode="bilinear", align_corners=False)
            embeddings.append(F.normalize(value, dim=1, eps=1e-6))
        return torch.cat(embeddings, dim=1)


class TaskAgnosticCCPRFModel(nn.Module):
    """Backbone plus an interchangeable task head."""

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
        self.backbone = MultiViewCCPRFBackbone(**dict(backbone_kwargs or {}))
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
        tensor = inputs[0] if isinstance(inputs, Sequence) and not torch.is_tensor(inputs) else inputs
        if not torch.is_tensor(tensor) or tensor.ndim != 4:
            raise ValueError("model inputs must contain BCHW tensors")
        return tuple(int(v) for v in tensor.shape[-2:])

    def forward(self, inputs: Any):
        output_size = self._spatial_size(inputs)
        features = self.backbone(inputs)
        if self.task == "segment":
            return self.head(features, output_size)
        return self.head(features)

class UniversalConvNormAct(nn.Module):
    """View-order-stable convolution block using GroupNorm.

    GroupNorm has no running statistics, so shared stream weights behave the
    same for one view, several views, batch size one, or different view orders.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        groups = math.gcd(out_channels, min(8, out_channels))
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UniversalResidualDownBlock(nn.Module):
    """Shared-stream residual stage without batch-dependent state."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = UniversalConvNormAct(in_channels, out_channels, 3, 2)
        self.refine1 = UniversalConvNormAct(out_channels, out_channels, 3, 1)
        self.refine2 = UniversalConvNormAct(
            out_channels, out_channels, 3, 1, activation=False
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        return self.act(y + self.refine2(self.refine1(y)))


class UniversalAppearanceTextureAdapter(nn.Module):
    """Single-image appearance/texture adapter with view-stable normalization."""

    def __init__(
        self,
        input_channels: int = 3,
        out_channels: int = 32,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or out_channels <= 0:
            raise ValueError("input_channels and out_channels must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.raw_kernel = nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size))
        self.appearance_stem = UniversalConvNormAct(input_channels, out_channels, 3, 2)
        self.texture_stem = UniversalConvNormAct(input_channels, out_channels, 3, 2)

    def normalized_kernel(self) -> torch.Tensor:
        return torch.softmax(self.raw_kernel.flatten(), dim=0).view(
            1, 1, self.kernel_size, self.kernel_size
        )

    def decompose(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(image) or image.ndim != 4:
            raise ValueError("expected one BCHW tensor")
        if image.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} channels, got {image.shape[1]}"
            )
        pad = self.kernel_size // 2
        mode = "reflect" if min(image.shape[-2:]) > pad else "replicate"
        padded = F.pad(image, (pad, pad, pad, pad), mode=mode)
        kernel = self.normalized_kernel().to(device=image.device, dtype=image.dtype)
        kernel = kernel.expand(self.input_channels, 1, -1, -1)
        low = F.conv2d(padded, kernel, groups=self.input_channels)
        return low, image - low

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, texture = self.decompose(image)
        return self.appearance_stem(image), self.texture_stem(texture)


class SingleViewAdapter(nn.Module):
    """Project one image or feature view without manufacturing a second view."""

    def __init__(self, input_channels: int = 3, out_channels: int = 32) -> None:
        super().__init__()
        if input_channels <= 0 or out_channels <= 0:
            raise ValueError("input_channels and out_channels must be positive")
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.stem = UniversalConvNormAct(self.input_channels, self.out_channels, 3, 2)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor]:
        if not torch.is_tensor(image) or image.ndim != 4:
            raise ValueError("SingleViewAdapter expects one BCHW tensor")
        if image.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} channels, got {image.shape[1]}"
            )
        return (self.stem(image),)


class MultiInputAdapter(nn.Module):
    """Project one or more named/ordered physical views to a common width.

    ``view_channels`` may be a sequence, in which case the input must be an
    equally long sequence, or a mapping, in which case any non-empty configured
    subset can be supplied. Mapping mode is useful for optional or missing
    modalities because the same model can consume, for example, RGB only,
    RGB+depth, or RGB+depth+thermal without changing the backbone mathematics.
    """

    def __init__(
        self,
        view_channels: Mapping[str, int] | Sequence[int],
        out_channels: int = 32,
    ) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        self.out_channels = int(out_channels)
        self.mapping_mode = isinstance(view_channels, Mapping)
        if self.mapping_mode:
            items = [(str(name), int(channels)) for name, channels in view_channels.items()]
            if not items:
                raise ValueError("view_channels mapping must not be empty")
            if len({name for name, _ in items}) != len(items):
                raise ValueError("view names must be unique")
            self.view_names = tuple(name for name, _ in items)
            channel_values = tuple(channels for _, channels in items)
        else:
            channel_values = tuple(int(channels) for channels in view_channels)
            if not channel_values:
                raise ValueError("view_channels sequence must not be empty")
            self.view_names = tuple(str(index) for index in range(len(channel_values)))
        if any(channels <= 0 for channels in channel_values):
            raise ValueError("all view channel counts must be positive")
        self.view_channels = dict(zip(self.view_names, channel_values))
        self.stems = nn.ModuleDict(
            {
                name: UniversalConvNormAct(channels, self.out_channels, 3, 2)
                for name, channels in self.view_channels.items()
            }
        )

    @staticmethod
    def _validate_common_geometry(views: Sequence[torch.Tensor]) -> None:
        if not views:
            raise ValueError("at least one view is required")
        reference = views[0]
        if not torch.is_tensor(reference) or reference.ndim != 4:
            raise ValueError("every view must be a BCHW tensor")
        for view in views[1:]:
            if not torch.is_tensor(view) or view.ndim != 4:
                raise ValueError("every view must be a BCHW tensor")
            if view.shape[0] != reference.shape[0] or view.shape[-2:] != reference.shape[-2:]:
                raise ValueError("all views must share batch and spatial dimensions")

    def forward(self, inputs: Mapping[str, torch.Tensor] | Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if self.mapping_mode:
            if not isinstance(inputs, Mapping) or not inputs:
                raise ValueError("named multi-input mode expects a non-empty mapping")
            unknown = set(inputs).difference(self.stems)
            if unknown:
                raise ValueError(f"unknown view names: {sorted(unknown)}")
            names = tuple(name for name in self.view_names if name in inputs)
            raw_views = tuple(inputs[name] for name in names)
        else:
            if torch.is_tensor(inputs) or not isinstance(inputs, Sequence):
                raise ValueError("ordered multi-input mode expects a tensor sequence")
            if len(inputs) != len(self.view_names):
                raise ValueError(
                    f"expected {len(self.view_names)} views, got {len(inputs)}"
                )
            names = self.view_names
            raw_views = tuple(inputs)
        self._validate_common_geometry(raw_views)
        projected = []
        for name, view in zip(names, raw_views):
            expected = self.view_channels[name]
            if view.shape[1] != expected:
                raise ValueError(
                    f"view {name!r} expects {expected} channels, got {view.shape[1]}"
                )
            projected.append(self.stems[name](view))
        return tuple(projected)


class UniversalInputAdapter(nn.Module):
    """Runtime input adapter for direct, decomposed, paired, and N-view cases.

    Modes:
      * ``single``: one tensor, one stream, no artificial second view;
      * ``decomposed``: one tensor -> appearance/texture streams;
      * ``multi``: one or more configured physical views;
      * ``auto``: tensor inputs use ``single_strategy`` while mappings/sequences
        use the configured multi-view adapter.
    """

    valid_modes = {"single", "decomposed", "multi", "auto"}

    def __init__(
        self,
        mode: str = "auto",
        input_channels: int = 3,
        out_channels: int = 32,
        single_strategy: str = "direct",
        view_channels: Mapping[str, int] | Sequence[int] | None = None,
        lowpass_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        self.single_strategy = str(single_strategy).lower()
        if self.mode not in self.valid_modes:
            raise ValueError(f"mode must be one of {sorted(self.valid_modes)}")
        if self.single_strategy not in {"direct", "appearance_texture"}:
            raise ValueError("single_strategy must be direct or appearance_texture")
        needs_direct = self.mode == "single" or (
            self.mode == "auto" and self.single_strategy == "direct"
        )
        needs_decomposed = self.mode == "decomposed" or (
            self.mode == "auto" and self.single_strategy == "appearance_texture"
        )
        self.single = (
            SingleViewAdapter(input_channels, out_channels) if needs_direct else None
        )
        self.decomposed = (
            UniversalAppearanceTextureAdapter(
                input_channels=input_channels,
                out_channels=out_channels,
                kernel_size=lowpass_kernel,
            )
            if needs_decomposed
            else None
        )
        self.multi = (
            MultiInputAdapter(view_channels, out_channels)
            if view_channels is not None
            else None
        )

    @staticmethod
    def _is_multi_input(inputs: Any) -> bool:
        return isinstance(inputs, Mapping) or (
            isinstance(inputs, Sequence) and not torch.is_tensor(inputs)
        )

    def _single_forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.single_strategy == "appearance_texture":
            if self.decomposed is None:
                raise RuntimeError("decomposed adapter is not initialized")
            return tuple(self.decomposed(image))
        if self.single is None:
            raise RuntimeError("single adapter is not initialized")
        return self.single(image)

    def forward(self, inputs: Any) -> tuple[torch.Tensor, ...]:
        if self.mode == "single":
            if self.single is None:
                raise RuntimeError("single adapter is not initialized")
            return self.single(inputs)
        if self.mode == "decomposed":
            if self.decomposed is None:
                raise RuntimeError("decomposed adapter is not initialized")
            return tuple(self.decomposed(inputs))
        if self.mode == "multi":
            if self.multi is None:
                raise ValueError("multi mode requires view_channels")
            return self.multi(inputs)

        # auto mode
        if torch.is_tensor(inputs):
            return self._single_forward(inputs)
        if self._is_multi_input(inputs):
            if isinstance(inputs, Sequence) and not isinstance(inputs, Mapping) and len(inputs) == 1:
                return self._single_forward(inputs[0])
            if self.multi is None:
                raise ValueError("auto mode requires view_channels for multi-view inputs")
            return self.multi(inputs)
        raise ValueError("unsupported input container")


class UniversalCCPRFSetFusion(nn.Module):
    """Apply one shared CCPRF operator to one, two, or many aligned views.

    One view is returned unchanged. Two views use the original pairwise CCPRF.
    For ``M > 2``, each view is paired with the leave-one-out consensus

        c_m = (sum_j F_j - F_m) / (M - 1),

    and only the corrected first branch is retained. The construction is
    permutation equivariant and keeps the exact identity initialization.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 16,
        stat_grid: int = 16,
        eps: float = 1e-3,
        trust_radius: float = 0.05,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.pair_fusion = LocalGlobalCCPRF(
            channels=channels,
            groups=groups,
            stat_grid=stat_grid,
            eps=eps,
            trust_radius=trust_radius,
            window_size=window_size,
        )

    @staticmethod
    def _validate(views: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if torch.is_tensor(views) or not isinstance(views, Sequence) or len(views) == 0:
            raise ValueError("UniversalCCPRFSetFusion expects one or more tensors")
        values = tuple(views)
        reference = values[0]
        if not torch.is_tensor(reference) or reference.ndim != 4:
            raise ValueError("every feature view must be a BCHW tensor")
        for view in values[1:]:
            if not torch.is_tensor(view) or view.shape != reference.shape:
                raise ValueError("all feature views must have identical BCHW shapes")
        return values

    def _forward_impl(self, views: Sequence[torch.Tensor]):
        values = self._validate(views)
        count = len(values)
        if count == 1:
            return values, {
                "view_count": 1,
                "mode": "identity",
                "per_view": (),
            }
        if count == 2:
            output, diagnostics = self.pair_fusion.forward_with_diagnostics(values)
            corrected = tuple(output.chunk(2, dim=1))
            return corrected, {
                "view_count": 2,
                "mode": "pair",
                "per_view": (diagnostics,),
            }

        total = torch.stack(values, dim=0).sum(dim=0)
        corrected = []
        diagnostics = []
        denominator = float(count - 1)
        for view in values:
            consensus = (total - view) / denominator
            pair_output, pair_diag = self.pair_fusion.forward_with_diagnostics(
                (view, consensus)
            )
            corrected.append(pair_output[:, : self.channels])
            diagnostics.append(pair_diag)
        return tuple(corrected), {
            "view_count": count,
            "mode": "leave_one_out_consensus",
            "per_view": tuple(diagnostics),
        }

    def forward_with_diagnostics(self, views: Sequence[torch.Tensor]):
        return self._forward_impl(views)

    def forward(self, views: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return self._forward_impl(views)[0]


class ViewSetAggregator(nn.Module):
    """Permutation-invariant attention pooling over one or more feature views.

    The score map is zero initialized. Therefore the initial aggregation is an
    arithmetic mean, and a single view is returned exactly unchanged.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.score_weight = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        self.score_bias = nn.Parameter(torch.zeros(1))

    def forward_with_weights(
        self, views: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = UniversalCCPRFSetFusion._validate(views)
        if len(values) == 1:
            weights = torch.ones(
                values[0].shape[0], 1, 1, values[0].shape[-2], values[0].shape[-1],
                device=values[0].device,
                dtype=values[0].dtype,
            )
            return values[0], weights
        stacked = torch.stack(values, dim=1)  # B, M, C, H, W
        scores = (stacked * self.score_weight.unsqueeze(1)).sum(dim=2, keepdim=True)
        scores = scores + self.score_bias.view(1, 1, 1, 1, 1)
        weights = torch.softmax(scores.float(), dim=1).to(stacked.dtype)
        return (weights * stacked).sum(dim=1), weights

    def forward(self, views: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.forward_with_weights(views)[0]


class UniversalCCPRFBackbone(nn.Module):
    """One-to-many, input- and task-neutral P2--P5 CCPRF backbone.

    Unlike ``MultiViewCCPRFBackbone``, this backbone does not require two
    streams. Direct single-view inputs bypass cross-view fusion; decomposed
    single images use two complementary streams; and physical inputs may
    contain two or more views. Stage weights are shared across streams, so the
    number of views does not change the feature-pyramid interface.
    """

    stage_names = ("P2", "P3", "P4", "P5")

    def __init__(
        self,
        input_mode: str = "auto",
        input_channels: int = 3,
        single_strategy: str = "direct",
        view_channels: Mapping[str, int] | Sequence[int] | None = None,
        widths: Sequence[int] = (64, 128, 256, 512),
        fusion_stages: Sequence[str] = ("P2", "P3"),
        group_width: int = 8,
        stat_grid: int = 16,
        trust_radius: Mapping[str, float] | float = 0.05,
        local_windows: Mapping[str, int] | None = None,
        lowpass_kernel: int = 5,
    ) -> None:
        super().__init__()
        if len(widths) != 4 or any(int(width) <= 0 for width in widths):
            raise ValueError("widths must contain four positive channel counts")
        self.widths = tuple(int(width) for width in widths)
        stem_width = max(8, self.widths[0] // 2)
        self.adapter = UniversalInputAdapter(
            mode=input_mode,
            input_channels=input_channels,
            out_channels=stem_width,
            single_strategy=single_strategy,
            view_channels=view_channels,
            lowpass_kernel=lowpass_kernel,
        )
        channels_in = (stem_width, *self.widths[:-1])
        self.stages = nn.ModuleDict(
            {
                name: UniversalResidualDownBlock(cin, cout)
                for name, cin, cout in zip(self.stage_names, channels_in, self.widths)
            }
        )
        selected = {str(stage) for stage in fusion_stages}
        # A strictly direct single-stream model has no cross-view operation and
        # therefore allocates no unused CCPRF parameters. Auto mode retains the
        # modules because the same instance may later receive several views.
        if str(input_mode).lower() == "single":
            selected = set()
        unknown = selected.difference(self.stage_names)
        if unknown:
            raise ValueError(f"unknown fusion stages: {sorted(unknown)}")
        self.fusion_stages = selected
        local_windows = dict(local_windows or {})
        self.fusions = nn.ModuleDict()
        self.aggregators = nn.ModuleDict()
        for name, channels in zip(self.stage_names, self.widths):
            self.aggregators[name] = ViewSetAggregator(channels)
            if name in selected:
                groups = max(1, channels // max(1, int(group_width)))
                while channels % groups != 0:
                    groups -= 1
                radius = (
                    float(trust_radius[name])
                    if isinstance(trust_radius, Mapping)
                    else float(trust_radius)
                )
                self.fusions[name] = UniversalCCPRFSetFusion(
                    channels=channels,
                    groups=groups,
                    stat_grid=stat_grid,
                    trust_radius=radius,
                    window_size=int(local_windows.get(name, 0)),
                )

    def forward_with_diagnostics(self, inputs: Any):
        views = tuple(self.adapter(inputs))
        pyramid: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, Any] = {"input_view_count": len(views), "stages": {}}
        for name in self.stage_names:
            views = tuple(self.stages[name](view) for view in views)
            fusion_diag = None
            if name in self.fusions:
                views, fusion_diag = self.fusions[name].forward_with_diagnostics(views)
            aggregate, weights = self.aggregators[name].forward_with_weights(views)
            pyramid[name] = aggregate
            diagnostics["stages"][name] = {
                "view_count": len(views),
                "aggregation_weights": weights,
                "fusion": fusion_diag,
            }
        return pyramid, diagnostics

    def forward(self, inputs: Any) -> dict[str, torch.Tensor]:
        return self.forward_with_diagnostics(inputs)[0]


class UniversalCCPRFModel(nn.Module):
    """Universal CCPRF backbone with interchangeable task heads."""

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
        self.backbone = UniversalCCPRFBackbone(**dict(backbone_kwargs or {}))
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
        if torch.is_tensor(inputs):
            tensor = inputs
        elif isinstance(inputs, Mapping):
            if not inputs:
                raise ValueError("input mapping must not be empty")
            tensor = next(iter(inputs.values()))
        elif isinstance(inputs, Sequence) and inputs:
            tensor = inputs[0]
        else:
            raise ValueError("model inputs must contain at least one BCHW tensor")
        if not torch.is_tensor(tensor) or tensor.ndim != 4:
            raise ValueError("model inputs must contain BCHW tensors")
        return tuple(int(value) for value in tensor.shape[-2:])

    def forward_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        return self.backbone(inputs)

    def forward(self, inputs: Any):
        output_size = self._spatial_size(inputs)
        features = self.forward_features(inputs)
        if self.task == "segment":
            return self.head(features, output_size)
        return self.head(features)

