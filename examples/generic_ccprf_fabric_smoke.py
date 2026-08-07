"""Minimal task-agnostic CCPRF surface-inspection smoke example."""

import torch

from ultra_modeling.nn.modules import TaskAgnosticCCPRFModel


backbone = {
    "input_mode": "appearance_texture",
    "widths": (32, 64, 128, 256),
    "fusion_stages": ("P2", "P3"),
    "group_width": 8,
    "stat_grid": 12,
    "trust_radius": {"P2": 0.03, "P3": 0.05},
    "local_windows": {"P2": 8, "P3": 4},
}

image = torch.randn(2, 3, 256, 256)

for task, classes in (("classify", 2), ("segment", 1), ("detect", 4), ("anomaly", None)):
    model = TaskAgnosticCCPRFModel(
        task=task,
        num_classes=classes,
        backbone_kwargs=backbone,
        head_channels=32,
        anomaly_embedding_channels=16,
    ).eval()
    with torch.no_grad():
        output = model(image)
    if torch.is_tensor(output):
        print(task, tuple(output.shape))
    else:
        print(task, {level: {name: tuple(t.shape) for name, t in values.items()}
                     for level, values in output.items()})
