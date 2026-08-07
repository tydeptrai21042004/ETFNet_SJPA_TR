"""Validation utility for Universal CCPRF v10.5."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ultralytics
from ultralytics.nn.modules import (
    UniversalCCPRFBackbone,
    UniversalCCPRFModel,
    UniversalCCPRFSetFusion,
)


def shape_tree(value):
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: shape_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [shape_tree(item) for item in value]
    return str(type(value).__name__)


def small_kwargs(**overrides):
    config = {
        "input_mode": "auto",
        "input_channels": 3,
        "single_strategy": "direct",
        "view_channels": {"image": 3, "depth": 1, "thermal": 1},
        "widths": (16, 32, 48, 64),
        "fusion_stages": ("P2", "P3"),
        "group_width": 8,
        "stat_grid": 6,
        "trust_radius": {"P2": 0.03, "P3": 0.05},
        "local_windows": {"P2": 4, "P3": 2},
    }
    config.update(overrides)
    return config


def fixture(batch=6, size=48):
    y = torch.linspace(0, 2 * torch.pi, size).view(1, 1, size, 1)
    x = torch.linspace(0, 2 * torch.pi, size).view(1, 1, 1, size)
    base = 0.45 + 0.10 * torch.sin(10 * x) + 0.08 * torch.sin(12 * y)
    images = base.repeat(batch, 3, 1, 1)
    masks = torch.zeros(batch, 1, size, size)
    for index in range(batch):
        top = 6 + (index * 5) % 25
        left = 7 + (index * 7) % 24
        images[index, :, top : top + 6, left : left + 8] += 0.35
        masks[index, :, top : top + 6, left : left + 8] = 1
    return images.clamp(0, 1), masks


def train_smoke(single_strategy: str):
    torch.manual_seed(91)
    model = UniversalCCPRFModel(
        task="segment",
        num_classes=1,
        backbone_kwargs=small_kwargs(
            input_mode="auto",
            single_strategy=single_strategy,
            view_channels=None,
            widths=(8, 16, 24, 32),
            group_width=4,
            local_windows={"P2": 3, "P3": 2},
        ),
        head_channels=8,
    )
    images, masks = fixture()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    with torch.no_grad():
        initial = F.binary_cross_entropy_with_logits(model(images), masks).item()
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(images), masks)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = F.binary_cross_entropy_with_logits(model(images), masks).item()
    return {
        "initial_loss": initial,
        "final_loss": final,
        "relative_reduction": (initial - final) / initial,
    }


def main():
    torch.manual_seed(90)
    image = torch.randn(2, 3, 64, 80)
    depth = torch.randn(2, 1, 64, 80)
    thermal = torch.randn(2, 1, 64, 80)

    backbone = UniversalCCPRFBackbone(**small_kwargs()).eval()
    with torch.no_grad():
        one, one_diag = backbone.forward_with_diagnostics(image)
        two, two_diag = backbone.forward_with_diagnostics(
            {"image": image, "depth": depth}
        )
        three, three_diag = backbone.forward_with_diagnostics(
            {"image": image, "depth": depth, "thermal": thermal}
        )
        subset, subset_diag = backbone.forward_with_diagnostics(
            {"image": image, "thermal": thermal}
        )

    tasks = {}
    task_cases = {
        "classify_single": (
            "classify",
            4,
            small_kwargs(input_mode="single", view_channels=None),
            image,
        ),
        "segment_decomposed": (
            "segment",
            1,
            small_kwargs(input_mode="decomposed", view_channels=None),
            image,
        ),
        "detect_three_view": (
            "detect",
            3,
            small_kwargs(input_mode="multi"),
            {"image": image, "depth": depth, "thermal": thermal},
        ),
        "anomaly_optional_subset": (
            "anomaly",
            None,
            small_kwargs(),
            {"image": image, "thermal": thermal},
        ),
    }
    for label, (task, classes, kwargs, inputs) in task_cases.items():
        model = UniversalCCPRFModel(
            task=task,
            num_classes=classes,
            backbone_kwargs=kwargs,
            head_channels=16,
            anomaly_embedding_channels=8,
        ).eval()
        with torch.no_grad():
            output = model(inputs)
        tasks[label] = {
            "output": shape_tree(output),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "finite": all(
                torch.isfinite(tensor).all().item()
                for tensor in _flatten_tensors(output)
            ),
        }

    fusion = UniversalCCPRFSetFusion(
        channels=16, groups=2, stat_grid=6, trust_radius=0.04
    ).eval()
    with torch.no_grad():
        fusion.pair_fusion.fusion.innovation_weight.normal_(0, 0.2)
    views = tuple(torch.randn(2, 16, 12, 10) for _ in range(3))
    permutation = (2, 0, 1)
    with torch.no_grad():
        first = fusion(views)
        second = fusion(tuple(views[index] for index in permutation))
    permutation_error = max(
        float((second[new] - first[old]).abs().max())
        for new, old in enumerate(permutation)
    )

    report = {
        "version": ultralytics.__version__,
        "cases": {
            "single_direct": {
                "input_view_count": one_diag["input_view_count"],
                "pyramid": shape_tree(one),
                "p2_fusion_mode": "not_allocated" if one_diag["stages"]["P2"]["fusion"] is None else one_diag["stages"]["P2"]["fusion"]["mode"],
            },
            "paired_physical": {
                "input_view_count": two_diag["input_view_count"],
                "pyramid": shape_tree(two),
            },
            "three_physical": {
                "input_view_count": three_diag["input_view_count"],
                "pyramid": shape_tree(three),
            },
            "optional_subset": {
                "input_view_count": subset_diag["input_view_count"],
                "pyramid": shape_tree(subset),
            },
        },
        "task_heads": tasks,
        "set_fusion": {
            "three_view_permutation_max_error": permutation_error,
            "identity_initialization_tested_for_view_counts": [1, 2, 3, 4],
        },
        "single_image_training_smoke": {
            "direct": train_smoke("direct"),
            "appearance_texture": train_smoke("appearance_texture"),
        },
        "design_guarantees": [
            "Direct single-view inputs do not manufacture a second view.",
            "One-view set fusion is an exact identity.",
            "Two-view inputs use pairwise bounded CCPRF.",
            "More than two views use a shared leave-one-out consensus operator.",
            "Named view mappings accept any non-empty configured subset.",
            "Shared stream stages use GroupNorm and have no running statistics.",
            "Feature output is always P2-P5 regardless of input view count.",
        ],
        "limitations": [
            "Software validation does not establish accuracy on an unseen real dataset.",
            "Physical modalities must be spatially aligned before cross-view fusion.",
            "Dense detection outputs require an external assignment, loss, and decoding pipeline.",
        ],
    }
    destination = ROOT / "VALIDATION" / "universal_ccprf_v10_5_validation_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _flatten_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_flatten_tensors(item))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_flatten_tensors(item))
        return result
    return []


if __name__ == "__main__":
    main()
