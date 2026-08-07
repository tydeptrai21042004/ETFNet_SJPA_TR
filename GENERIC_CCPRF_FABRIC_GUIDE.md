# Generic CCPRF for Surface and Fabric Inspection

## Scope

The generic extension separates three concerns:

1. **View construction**: one image can be decomposed into appearance and
   high-frequency texture views, or two physical modalities can be supplied.
2. **Task-independent fusion backbone**: local-global CCPRF produces P2--P5
   feature maps and contains no dataset, class, box-orientation, or application
   assumptions.
3. **Replaceable head**: classification, segmentation, dense axis-aligned
   detection, or anomaly-feature extraction.

## Single-image fabric mode

```python
from ultra_modeling.nn.modules import TaskAgnosticCCPRFModel

model = TaskAgnosticCCPRFModel(
    task="segment",
    num_classes=1,
    backbone_kwargs={
        "input_mode": "appearance_texture",
        "widths": (64, 128, 256, 512),
        "fusion_stages": ("P2", "P3"),
        "group_width": 8,
        "stat_grid": 16,
        "trust_radius": {"P2": 0.03, "P3": 0.05},
        "local_windows": {"P2": 8, "P3": 4},
    },
)
```

The adapter computes a constrained low-pass image and an exact residual texture
view. Its kernel is non-negative and sums to one, and the decomposition obeys

\[
X_{\mathrm{low}} + X_{\mathrm{texture}} = X.
\]

## Physical paired-view mode

```python
model = TaskAgnosticCCPRFModel(
    task="detect",
    num_classes=4,
    backbone_kwargs={
        "input_mode": "paired",
        "view_a_channels": 3,
        "view_b_channels": 1,
        "widths": (64, 128, 256, 512),
        "fusion_stages": ("P2", "P3"),
        "group_width": 8,
    },
)

prediction = model((visible_image, auxiliary_image))
```

The auxiliary image may be thermal, near-infrared, depth, polarization, or
another spatially aligned measurement. The fusion code uses neutral view names.

## Task selection

- `classify`: image-level logits.
- `segment`: full-resolution pixel logits.
- `detect`: per-level class, positive box-distance, and quality maps. Assignment
  and decoding remain training-pipeline responsibilities.
- `anomaly`: normalized multi-scale patch embeddings suitable for a nominal
  memory bank or another anomaly scorer.

## Mathematical safety

Every CCPRF stage is zero-initialized and applies a final samplewise trust
projection. At stage `s`,

\[
\|F_s^{\mathrm{out}} - [F_s^a;F_s^b]\|_F
\le \rho_s\|[F_s^a;F_s^b]\|_F.
\]

Local and global paths share the same fusion parameters. Their convexly blended
correction is projected again, so the bound still applies.

## What has and has not been validated

Validated locally:

- single-image and paired-view inputs;
- P2--P5 feature shapes;
- all four heads, backward gradients, and checkpoint reload;
- zero, constant, rank-deficient, random, and extreme fusion inputs;
- non-divisible local windows;
- exact appearance/texture reconstruction;
- a small synthetic woven-surface segmentation optimization whose loss falls;
- the complete existing repository test suite.

Not yet established:

- accuracy on a named real fabric dataset;
- superiority over a fabric-specific baseline;
- optimal window sizes or task losses for a particular annotation protocol.

Those claims require a real dataset experiment and should not be inferred from
software or synthetic tests.
