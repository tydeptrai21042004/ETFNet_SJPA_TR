# Universal CCPRF v10.5

Universal CCPRF removes the requirement that the model always receive two
modalities. The same P2--P5 backbone supports:

1. **Direct single view** — one image enters one stream; cross-view fusion is an
   exact identity and no artificial modality is generated.
2. **Self-complementary single view** — one image is decomposed into appearance
   and high-frequency texture streams.
3. **Paired physical views** — for example image + depth or RGB + thermal.
4. **Arbitrary configured view sets** — three or more aligned views.
5. **Named optional subsets** — one trained model can accept any non-empty
   configured subset such as image only, image + depth, or image + depth + thermal.

The output interface is always a task-neutral feature pyramid:

```text
P2, P3, P4, P5
```

The same backbone can be attached to classification, segmentation, dense
axis-aligned detection, or anomaly-feature heads.

## Core behavior

For one view, the set fusion is the identity:

\[
\mathcal F(F_1)=F_1.
\]

For two views, standard bounded CCPRF is used. For more than two views, each
view is paired with the leave-one-out consensus

\[
c_m=\frac{1}{M-1}\sum_{j\ne m}F_j.
\]

The first corrected branch is retained for view \(m\). All views then pass to a
permutation-invariant attention aggregator. Its score map is zero initialized,
so the initial aggregate is the arithmetic mean; for one view it is exactly the
input.

Shared stream stages use GroupNorm rather than BatchNorm. Therefore the
backbone has no view-order-dependent running statistics and works with batch
size one.

## Recommended constructors

### Direct single image

```python
model = UniversalCCPRFModel(
    task="segment",
    num_classes=1,
    backbone_kwargs={
        "input_mode": "single",
        "input_channels": 3,
        "widths": (64, 128, 256, 512),
    },
)
```

### Single image with appearance/texture decomposition

```python
model = UniversalCCPRFModel(
    task="segment",
    num_classes=1,
    backbone_kwargs={
        "input_mode": "decomposed",
        "input_channels": 3,
        "single_strategy": "appearance_texture",
    },
)
```

### Optional named physical modalities

```python
model = UniversalCCPRFModel(
    task="detect",
    num_classes=5,
    backbone_kwargs={
        "input_mode": "auto",
        "input_channels": 3,
        "single_strategy": "direct",
        "view_channels": {
            "image": 3,
            "depth": 1,
            "thermal": 1,
        },
    },
)

single = model(image)
paired = model({"image": image, "depth": depth})
triple = model({"image": image, "depth": depth, "thermal": thermal})
```

## Scope

The module contains no dataset classes, file paths, UAV assumptions, fabric
assumptions, or oriented-box geometry. Dataset loading, targets, losses,
assignment, decoding, and evaluation metrics remain outside the backbone.
