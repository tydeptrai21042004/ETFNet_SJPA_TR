from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ultra_modeling.nn.modules.cerc import CERCModel, ConvolutionalCERC


torch.set_num_threads(2)
torch.manual_seed(20260807)


def kwargs(channels):
    return {
        "input_channels": channels,
        "widths": (16, 24, 32, 40),
        "group_width": 4,
        "stat_grid": 4,
        "trust_radius": 0.05,
        "relation_kernel": 3,
    }


def train_steps(model, batch_fn, loss_fn, steps=12, lr=4e-3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        x, target = batch_fn()
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def fabric_batch(batch=6, size=48):
    yy, xx = torch.meshgrid(
        torch.linspace(-math.pi, math.pi, size),
        torch.linspace(-math.pi, math.pi, size), indexing="ij"
    )
    base = 0.5 + 0.15 * torch.sin(10 * xx) + 0.12 * torch.cos(12 * yy)
    images = []
    masks = []
    for index in range(batch):
        image = base.clone()
        mask = torch.zeros(size, size)
        h = 5 + (index % 3)
        w = 7 + (index % 4)
        y0 = 6 + ((index * 7) % (size - h - 8))
        x0 = 5 + ((index * 9) % (size - w - 7))
        image[y0:y0+h, x0:x0+w] += 0.55
        mask[y0:y0+h, x0:x0+w] = 1.0
        rgb = torch.stack((image, image * 0.93, image * 1.05)).clamp(0, 1)
        rgb += 0.02 * torch.randn_like(rgb)
        images.append(rgb)
        masks.append(mask)
    return torch.stack(images), torch.stack(masks).long()


def medical_classification_batch(batch=12, size=28):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    images = []
    labels = []
    for i in range(batch):
        label = i % 2
        radius = 0.25 if label == 0 else 0.50
        blob = torch.exp(-((xx ** 2 + yy ** 2) / (2 * radius ** 2)))
        if label == 1:
            blob = blob - 0.45 * torch.exp(-((xx ** 2 + yy ** 2) / (2 * 0.16 ** 2)))
        image = blob + 0.04 * torch.randn_like(blob)
        images.append(image.unsqueeze(0))
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def ultrasound_segmentation_batch(batch=6, h=48, w=44):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    images = []
    masks = []
    for i in range(batch):
        cx = -0.25 + 0.1 * (i % 5)
        cy = -0.15 + 0.08 * (i % 4)
        rx = 0.28 + 0.02 * (i % 3)
        ry = 0.20 + 0.02 * ((i + 1) % 3)
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1).float()
        background = 0.35 + 0.10 * torch.randn(h, w)
        speckle = 0.10 * background * torch.randn(h, w)
        image = background + speckle + 0.35 * mask
        images.append(image.unsqueeze(0))
        masks.append(mask.long())
    return torch.stack(images), torch.stack(masks)


def multimodal_batch(batch=4, size=40):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    evidence = {name: [] for name in ("t1", "t2", "flair", "adc")}
    masks = []
    for i in range(batch):
        cx = -0.2 + 0.12 * i
        cy = 0.15 - 0.08 * i
        mask = (((xx - cx) ** 2 + (yy - cy) ** 2) < 0.18 ** 2).float()
        common = torch.exp(-((xx-cx)**2 + (yy-cy)**2) / 0.08)
        evidence["t1"].append((0.3 + 0.35 * common + 0.05 * torch.randn_like(common)).unsqueeze(0))
        evidence["t2"].append((0.4 + 0.25 * common + 0.05 * torch.randn_like(common)).unsqueeze(0))
        evidence["flair"].append((0.2 + 0.55 * common + 0.05 * torch.randn_like(common)).unsqueeze(0))
        evidence["adc"].append((0.6 - 0.25 * common + 0.05 * torch.randn_like(common)).unsqueeze(0))
        masks.append(mask.long())
    return {name: torch.stack(values) for name, values in evidence.items()}, torch.stack(masks)


def loss_seg(logits, target):
    return F.cross_entropy(logits, target)


results = {}

# Mathematical diagnostics on actual ConvolutionalCERC.
relation = ConvolutionalCERC(16, group_width=4, stat_grid=4, trust_radius=0.05)
x = torch.randn(3, 16, 16, 16)
_, single_diag = relation.forward_with_diagnostics(x)
_, paired_diag = relation.forward_with_diagnostics((x, x + 0.1 * torch.randn_like(x)))
with torch.no_grad():
    relation.transport.weight.normal_(0, 0.15)
_, bounded_diag = relation.forward_with_diagnostics((x, torch.randn_like(x)))
applied = bounded_diag["trust_scale"] * bounded_diag["correction_norm"]
trust_ratio = float((applied / (bounded_diag["trust_radius_norm"] + 1e-8)).max())
results["mathematics"] = {
    "single_evidence_weight_min": float(single_diag["consensus_weights"].min()),
    "single_evidence_weight_max": float(single_diag["consensus_weights"].max()),
    "paired_support_mean": float(paired_diag["support"].mean()),
    "max_trust_ratio": trust_ratio,
    "innovation_parameters_per_stage_d4_k3": relation.innovation_parameters,
    "finite": bool(single_diag["finite"] and paired_diag["finite"] and bounded_diag["finite"]),
}

# Fabric-like RGB segmentation.
torch.manual_seed(100)
fabric_model = CERCModel("segment", num_classes=2, backbone_kwargs=kwargs(3), head_channels=12)
fabric_losses = train_steps(fabric_model, fabric_batch, loss_seg, steps=12, lr=5e-3)
results["fabric_rgb_segmentation_fixture"] = {
    "initial_loss": fabric_losses[0],
    "final_loss": fabric_losses[-1],
    "relative_reduction": (fabric_losses[0] - fabric_losses[-1]) / fabric_losses[0],
}

# MedMNIST-like small grayscale classification.
torch.manual_seed(101)
medical_cls = CERCModel("classify", num_classes=2, backbone_kwargs=kwargs(1))
cls_losses = train_steps(
    medical_cls,
    medical_classification_batch,
    lambda logits, labels: F.cross_entropy(logits, labels),
    steps=14,
    lr=5e-3,
)
results["medical_28x28_grayscale_classification_fixture"] = {
    "initial_loss": cls_losses[0],
    "final_loss": cls_losses[-1],
    "relative_reduction": (cls_losses[0] - cls_losses[-1]) / cls_losses[0],
}

# Ultrasound-like grayscale segmentation.
torch.manual_seed(102)
ultrasound = CERCModel("segment", num_classes=2, backbone_kwargs=kwargs(1), head_channels=12)
us_losses = train_steps(ultrasound, ultrasound_segmentation_batch, loss_seg, steps=12, lr=4e-3)
results["medical_ultrasound_segmentation_fixture"] = {
    "initial_loss": us_losses[0],
    "final_loss": us_losses[-1],
    "relative_reduction": (us_losses[0] - us_losses[-1]) / us_losses[0],
}

# Four-evidence medical-like segmentation using the same relation equations.
torch.manual_seed(103)
mm_kwargs = kwargs({"t1": 1, "t2": 1, "flair": 1, "adc": 1})
multimodal = CERCModel("segment", num_classes=2, backbone_kwargs=mm_kwargs, head_channels=12)
mm_losses = train_steps(multimodal, multimodal_batch, loss_seg, steps=10, lr=4e-3)
results["medical_four_evidence_segmentation_fixture"] = {
    "initial_loss": mm_losses[0],
    "final_loss": mm_losses[-1],
    "relative_reduction": (mm_losses[0] - mm_losses[-1]) / mm_losses[0],
}

# Checkpoint and missing-modality path after optimization.
multimodal.eval()
full_inputs, _ = multimodal_batch(batch=1)
subset = {"t1": full_inputs["t1"], "flair": full_inputs["flair"]}
with torch.no_grad():
    full_output = multimodal(full_inputs)
    subset_output = multimodal(subset)
clone = copy.deepcopy(multimodal).eval()
clone.load_state_dict(multimodal.state_dict(), strict=True)
with torch.no_grad():
    reloaded = clone(subset)
results["runtime_generalization"] = {
    "full_four_evidence_output_shape": list(full_output.shape),
    "missing_modality_two_evidence_output_shape": list(subset_output.shape),
    "checkpoint_reload_max_difference": float((subset_output - reloaded).abs().max()),
    "finite_full": bool(torch.isfinite(full_output).all()),
    "finite_subset": bool(torch.isfinite(subset_output).all()),
}

for name, payload in results.items():
    if "initial_loss" in payload and not payload["final_loss"] < payload["initial_loss"]:
        raise RuntimeError(f"fixture did not learn: {name}: {payload}")
if results["mathematics"]["max_trust_ratio"] > 1.0001:
    raise RuntimeError(f"trust bound failed: {results['mathematics']}")
if results["runtime_generalization"]["checkpoint_reload_max_difference"] != 0.0:
    raise RuntimeError("checkpoint reload was not exact")

output = Path(__file__).with_name("cerc_v10_6_validation_report.json")
output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(json.dumps(results, indent=2))
print("saved", output)
