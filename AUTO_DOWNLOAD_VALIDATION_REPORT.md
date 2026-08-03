# ETFNet-SJPA-TR v4 — public-data and end-to-end validation

## Scope

This release adds automatic acquisition and deterministic preprocessing for the NII-CU Multispectral Aerial Person Detection Dataset, while retaining the corrected six-channel ETFNet-SJPA-TR train, validation, paired prediction, checkpoint, and export pipeline.

## Public source

- Dataset: NII-CU Multispectral Aerial Person Detection Dataset
- Associated publication DOI: `10.1002/rob.22082`
- Official labelled archive size: approximately 9.1 GB
- Modalities: aligned RGB and far-infrared UAV imagery
- Official count: 5,880 pairs
- License: CC BY-NC-SA 3.0
- Default conversion variant: `4-channel`

The repository records this source information and the user's preprocessing choices in `SOURCE_MANIFEST.json`.

## Added functionality

1. `list-data` public-dataset registry command.
2. `download-data` command with explicit license acceptance.
3. Direct training references such as `--data public:nii-cu-mapd`.
4. HTTP retry, backoff, Range resume, `.part` handling, and atomic completion.
5. HTML-preview detection for changed provider links.
6. ZIP signature and CRC validation.
7. Safe extraction rejecting traversal paths and archive symlinks.
8. Support for a provider ZIP that contains the labelled dataset ZIP as a nested archive.
9. Deterministic pair matching by exact relative paths and stems.
10. Conversion of official `x1 y1 x2 y2 type occluded bad` rows to YOLO detect or four-corner OBB labels.
11. Boundary clipping, sub-pixel rejection, visibility filtering, and optional bad-row filtering.
12. Hard-link, symlink, or copy materialization of the standardized paired layout.
13. Archive SHA-256, settings, counts, and filter statistics in a machine-readable manifest.
14. Idempotent extraction and preprocessing markers plus process locking.
15. Automatic post-conversion dataset validation and content fingerprints.

## Automated tests

### Pytest suite

```text
58 passed, 3 deprecation warnings
```

The warnings come from PyTorch's notice that `torch.jit.trace` is deprecated in favor of newer export APIs; no test failed.

Public-dataset-specific cases cover:

- Registry source, DOI, license, variant, and class metadata.
- Required license acceptance.
- Official miniature archive structure.
- OBB and axis-aligned label conversion.
- Header parsing, boundary clipping, visibility and bad-row handling.
- Missing-pair and dimension-mismatch rejection.
- Idempotent reruns and `public:` alias resolution.
- Malicious path traversal and invalid ZIP rejection.
- Interrupted HTTP download recovery.
- CLI argument availability.

### Public-data end-to-end test

A miniature archive using the documented NII-CU directory and annotation format completed:

```text
archive validation
→ extraction
→ preprocessing
→ content fingerprint
→ public:nii-cu-mapd resolution
→ one-epoch six-channel OBB training
→ validation
→ paired prediction
```

### General end-to-end deployment test

The process-isolated `tests/run_full_e2e.py` completed:

```text
PASS fixture
PASS preflight
PASS train
PASS validate
PASS paired-predict
PASS torchscript-export
PASS torchscript-predict
```

### Exact resume

Continuous two-epoch training and interrupted/resumed training were tensor-identical:

```json
{
  "exact": true,
  "model_differences": [],
  "ema_differences": [],
  "continuous_epoch": 1,
  "resumed_epoch": 1,
  "continuous_updates": 3,
  "resumed_updates": 3
}
```

### Built-wheel test

The v4 wheel was installed into an isolated target directory. From the installed wheel rather than the source tree, the following passed:

- Public dataset registry command.
- Local official-format archive preprocessing and validation.
- Import of the packaged `ultralytics` fork.
- Construction and six-channel forward execution of the full ETFNet-SJPA-TR model.

## Commands

```bash
python etfnet_cli.py list-data

python etfnet_cli.py download-data \
  --dataset nii-cu-mapd \
  --output datasets \
  --variant 4-channel \
  --task obb \
  --accept-license
```

Direct automatic preparation and training:

```bash
python train.py \
  --data public:nii-cu-mapd \
  --datasets-dir datasets \
  --dataset-variant 4-channel \
  --accept-dataset-license \
  --epochs 100 \
  --batch 4 \
  --imgsz 640 \
  --device 0
```

Offline/manual archive fallback:

```bash
python etfnet_cli.py download-data \
  --dataset nii-cu-mapd \
  --archive /path/to/NII_CU_MAPD_dataset.zip \
  --output datasets \
  --accept-license
```

## Honest limitation

The execution environment did not download the complete 9.1 GB official archive. The official source URL, documented structure, labels, DOI, and license were verified from the project page. Network behavior was tested through a local HTTP server with a deliberately interrupted ZIP, while the entire preprocessing and training pipeline was tested with an exact miniature replica of the documented archive format. A first real run should therefore begin with `download-data`; if Dropbox changes its public redirect behavior, the `--archive` fallback uses the same validated preprocessing path.

NII-CU is a person-detection benchmark and does not replace the real DroneVehicle, VTUAV-det, and RGBTDronePerson experiments needed to support a Q1 paper's accuracy claims.
