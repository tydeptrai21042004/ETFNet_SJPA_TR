"""Runnable examples for Universal CCPRF v10.5."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules import UniversalCCPRFModel


def small_backbone(**overrides):
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


def main() -> None:
    torch.manual_seed(0)
    image = torch.randn(2, 3, 64, 80)
    depth = torch.randn(2, 1, 64, 80)
    thermal = torch.randn(2, 1, 64, 80)

    classifier = UniversalCCPRFModel(
        task="classify",
        num_classes=4,
        backbone_kwargs=small_backbone(),
    ).eval()
    segmenter = UniversalCCPRFModel(
        task="segment",
        num_classes=1,
        backbone_kwargs=small_backbone(input_mode="decomposed"),
        head_channels=16,
    ).eval()
    detector = UniversalCCPRFModel(
        task="detect",
        num_classes=3,
        backbone_kwargs=small_backbone(input_mode="multi"),
        head_channels=16,
    ).eval()
    anomaly = UniversalCCPRFModel(
        task="anomaly",
        backbone_kwargs=small_backbone(),
        anomaly_embedding_channels=8,
    ).eval()

    with torch.no_grad():
        # Direct one-view case: no second view is created and CCPRF is bypassed.
        classification = classifier(image)

        # One image, two self-complementary appearance/texture streams.
        segmentation = segmenter(image)

        # Three physical views.
        detection = detector({"image": image, "depth": depth, "thermal": thermal})

        # The same auto model also accepts a named subset of physical views.
        anomaly_features = anomaly({"image": image, "thermal": thermal})

    print("classification:", tuple(classification.shape))
    print("segmentation:", tuple(segmentation.shape))
    print("detection levels:", tuple(detection))
    print("anomaly features:", tuple(anomaly_features.shape))


if __name__ == "__main__":
    main()
