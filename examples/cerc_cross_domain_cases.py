"""Small forward examples for CERC v10.6 across unrelated 2-D domains."""

import torch

from ultra_modeling.nn.modules import CERCModel


def small_backbone(input_channels):
    return {
        "input_channels": input_channels,
        "widths": (16, 24, 32, 40),
        "group_width": 4,
        "stat_grid": 4,
        "trust_radius": 0.05,
        "relation_kernel": 3,
    }


cases = {
    "fabric_rgb_segmentation": (
        CERCModel("segment", 2, backbone_kwargs=small_backbone(3), head_channels=16),
        torch.randn(1, 3, 128, 128),
    ),
    "medical_grayscale_classification": (
        CERCModel("classify", 4, backbone_kwargs=small_backbone(1)),
        torch.randn(2, 1, 28, 28),
    ),
    "dermoscopy_or_endoscopy_segmentation": (
        CERCModel("segment", 2, backbone_kwargs=small_backbone(3), head_channels=16),
        torch.randn(1, 3, 128, 112),
    ),
    "ultrasound_segmentation": (
        CERCModel("segment", 2, backbone_kwargs=small_backbone(1), head_channels=16),
        torch.randn(1, 1, 96, 80),
    ),
    "four_evidence_medical_segmentation": (
        CERCModel(
            "segment",
            3,
            backbone_kwargs=small_backbone({"t1": 1, "t2": 1, "flair": 1, "adc": 1}),
            head_channels=16,
        ),
        {
            "t1": torch.randn(1, 1, 96, 96),
            "t2": torch.randn(1, 1, 96, 96),
            "flair": torch.randn(1, 1, 96, 96),
            "adc": torch.randn(1, 1, 96, 96),
        },
    ),
}

for name, (model, inputs) in cases.items():
    model.eval()
    with torch.no_grad():
        output = model(inputs)
    if isinstance(output, dict):
        shape = {key: {field: tuple(value.shape) for field, value in item.items()} for key, item in output.items()}
    else:
        shape = tuple(output.shape)
    print(name, shape)
