# CERC v10.6 cross-domain guide

CERC itself contains no dataset names or task-specific geometry. Dataset profiles
live under `configs/cerc_cross_domain_datasets.yaml` and only specify channels,
head type, supervision, and source information.

## Fabric / industrial validation targets

- **AITEX** — fabric defect images with defect masks; use segmentation or anomaly
  heads. A public mirror describes 245 images from 7 fabrics, with 140 defect-free
  and 105 defective images.
- **TILDA** — original textile texture benchmark; 3200 8-bit grayscale TIFF images
  across eight textile kinds and reference/defect classes. Useful for grayscale
  classification, anomaly detection, or derived localization protocols.
- **MVTec AD texture categories** — carpet, grid, leather, tile and wood provide a
  strong cross-domain surface-anomaly check with pixel-level anomaly masks.

## Medical 2-D validation targets

- **MedMNIST2D** — 12 standardized 2-D biomedical classification datasets at
  28x28, spanning X-ray, OCT, ultrasound, CT, dermatoscopy and microscopy. CERC
  supports the 2-D collection; MedMNIST3D is outside the scope of v10.6.
- **ISIC 2018 Task 1** — RGB dermoscopic lesion boundary segmentation with binary
  masks.
- **Kvasir-SEG** — RGB endoscopic polyp segmentation; the official downloadable
  archive is small enough for fast experimentation.
- **BUSI-WHU** — breast ultrasound tumor-region segmentation; the latest dataset
  release contains 927 images. The included profile decodes ultrasound as one
  grayscale channel.
- **Aligned multi-sequence 2-D medical slices** — configure named inputs such as
  T1/T2/FLAIR/ADC. Any non-empty configured subset can be supplied; CERC equations
  are unchanged when modalities are missing.

## Model examples

### Single RGB fabric image

```python
model = CERCModel(
    "segment",
    num_classes=2,
    backbone_kwargs={"input_channels": 3},
)
output = model(rgb)
```

The proposal does not manufacture a second view. The single feature field is
split into latent channel atoms and the same canonical relation operator is
applied internally.

### Single grayscale medical image

```python
model = CERCModel(
    "classify",
    num_classes=4,
    backbone_kwargs={"input_channels": 1},
)
output = model(xray_or_oct)
```

### Optional named medical evidence

```python
model = CERCModel(
    "segment",
    num_classes=3,
    backbone_kwargs={
        "input_channels": {"t1": 1, "t2": 1, "flair": 1, "adc": 1},
    },
)

output_full = model({"t1": t1, "t2": t2, "flair": flair, "adc": adc})
output_missing = model({"t1": t1, "flair": flair})
```

Only the raw-input projection changes with channel configuration. The CERC
relation equations and shared relational convolution are identical.

## Important scope boundary

CERC v10.6 is **2-D**. Volumetric CT/MRI or MedMNIST3D require replacing the
2-D backbone and relational convolution with 3-D counterparts and therefore are
not claimed as supported by this package.
