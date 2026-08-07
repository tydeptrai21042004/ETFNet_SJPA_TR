from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ultra_modeling.nn.modules.generic_ccprf import (
    AppearanceTextureAdapter,
    GenericCCPRF,
    LocalGlobalCCPRF,
    MultiViewCCPRFBackbone,
    TaskAgnosticCCPRFModel,
)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def fabric_batch(batch=6, size=48):
    y = torch.linspace(0, 2 * torch.pi, size).view(1, 1, size, 1)
    x = torch.linspace(0, 2 * torch.pi, size).view(1, 1, 1, size)
    base = 0.45 + 0.12 * torch.sin(10 * x) + 0.10 * torch.sin(12 * y)
    images = base.repeat(batch, 3, 1, 1)
    masks = torch.zeros(batch, 1, size, size)
    for index in range(batch):
        top = 8 + (index * 5) % 22
        left = 7 + (index * 7) % 24
        height = 5 + index % 3
        width = 7 + (index + 1) % 4
        images[index, :, top : top + height, left : left + width] += 0.35
        masks[index, :, top : top + height, left : left + width] = 1.0
    return images.clamp(0, 1), masks


def main():
    torch.manual_seed(20260807)
    report = {}

    adapter = AppearanceTextureAdapter(3, 8, 5)
    image = torch.randn(2, 3, 31, 29)
    low, high = adapter.decompose(image)
    report["appearance_texture"] = {
        "max_reconstruction_error": float((low + high - image).abs().max().detach()),
        "kernel_sum": float(adapter.normalized_kernel().sum().detach()),
        "kernel_min": float(adapter.normalized_kernel().min().detach()),
    }

    fusion = GenericCCPRF(16, groups=2, stat_grid=8, trust_radius=0.05).eval()
    a = torch.randn(2, 16, 17, 19)
    b = torch.randn_like(a)
    with torch.no_grad():
        identity = fusion((a, b))
        fusion.innovation_weight.normal_(0, 0.5)
        changed = fusion((a, b))
    raw = torch.cat((a, b), dim=1)
    delta = (changed - raw).flatten(1).float().norm(dim=1)
    radius = 0.05 * raw.flatten(1).float().norm(dim=1)
    report["generic_fusion"] = {
        "initial_max_difference": float((identity - raw).abs().max()),
        "max_trust_ratio": float((delta / radius.clamp_min(1e-12)).max()),
        "finite": bool(torch.isfinite(changed).all()),
    }

    local_global = LocalGlobalCCPRF(
        16, groups=2, stat_grid=8, trust_radius=0.04, window_size=7
    ).eval()
    with torch.no_grad():
        local_global.fusion.innovation_weight.normal_(0, 0.4)
        lg_out = local_global((a, b))
    lg_delta = (lg_out - raw).flatten(1).float().norm(dim=1)
    lg_radius = 0.04 * raw.flatten(1).float().norm(dim=1)
    report["local_global"] = {
        "output_shape": list(lg_out.shape),
        "max_trust_ratio": float((lg_delta / lg_radius.clamp_min(1e-12)).max()),
        "finite": bool(torch.isfinite(lg_out).all()),
    }

    backbone_kwargs = {
        "input_mode": "appearance_texture",
        "widths": (16, 32, 64, 96),
        "fusion_stages": ("P2", "P3"),
        "group_width": 8,
        "stat_grid": 8,
        "trust_radius": {"P2": 0.03, "P3": 0.05},
        "local_windows": {"P2": 4, "P3": 2},
    }
    backbone = MultiViewCCPRFBackbone(**backbone_kwargs).eval()
    input_image = torch.randn(2, 3, 64, 80)
    with torch.no_grad():
        pyramid = backbone(input_image)
    report["pyramid"] = {name: list(value.shape) for name, value in pyramid.items()}

    task_outputs = {}
    for task in ("classify", "segment", "detect", "anomaly"):
        model = TaskAgnosticCCPRFModel(
            task=task,
            num_classes=None if task == "anomaly" else 3,
            backbone_kwargs=backbone_kwargs,
            head_channels=16,
            anomaly_embedding_channels=8,
        ).eval()
        with torch.no_grad():
            output = model(input_image)
        if torch.is_tensor(output):
            shape = list(output.shape)
        else:
            shape = {
                level: {name: list(tensor.shape) for name, tensor in values.items()}
                for level, values in output.items()
            }
        task_outputs[task] = {
            "parameters": count_parameters(model),
            "output": shape,
        }
    report["tasks"] = task_outputs

    torch.manual_seed(6)
    segmenter = TaskAgnosticCCPRFModel(
        task="segment",
        num_classes=1,
        backbone_kwargs={
            "input_mode": "appearance_texture",
            "widths": (8, 16, 24, 32),
            "fusion_stages": ("P2", "P3"),
            "group_width": 4,
            "stat_grid": 6,
            "trust_radius": {"P2": 0.03, "P3": 0.05},
            "local_windows": {"P2": 3, "P3": 2},
        },
        head_channels=8,
    )
    images, masks = fabric_batch()
    optimizer = torch.optim.Adam(segmenter.parameters(), lr=3e-3)
    segmenter.train()
    with torch.no_grad():
        initial_loss = F.binary_cross_entropy_with_logits(segmenter(images), masks).item()
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(segmenter(images), masks)
        loss.backward()
        optimizer.step()
    segmenter.eval()
    with torch.no_grad():
        final_loss = F.binary_cross_entropy_with_logits(segmenter(images), masks).item()
    report["synthetic_fabric_learnability"] = {
        "initial_bce": initial_loss,
        "final_bce": final_loss,
        "relative_reduction": (initial_loss - final_loss) / initial_loss,
        "steps": 12,
        "note": "Software learnability smoke test; not a real-dataset benchmark.",
    }

    output = Path(__file__).with_name("generic_ccprf_validation_report.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(output)


if __name__ == "__main__":
    main()
