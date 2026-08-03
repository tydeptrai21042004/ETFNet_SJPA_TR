# Automatic public RGB–IR/RGB–NIR datasets

ETFNet–SJPA-TR v5 supports six public paired-modality datasets through one acquisition and preprocessing API. The five compact additions requested for development are M3FD, VEDAI, FLIR-Aligned, RGBTDronePerson, and CVC-14; NII-CU remains available from v4.

## Install acquisition dependencies

```bash
pip install -e ".[data]"
# or
pip install -r requirements-data.txt
```

Google Drive **folder** acquisition for M3FD and RGBTDronePerson uses `gdown>=6` and therefore requires Python 3.10 or newer. On older Python versions, download the source manually and pass `--archive`. Every dataset supports a local folder or archive fallback.

## Registry

```bash
python etfnet_cli.py list-data
python etfnet_cli.py list-data --json
```

| Alias | Modalities | Default variant | Classes | Acquisition route |
|---|---|---|---:|---|
| `public:m3fd` | visible + thermal | `default` | 6 | official TarDAL Google Drive folder |
| `public:vedai` | RGB + NIR | `512` | 9 | official GREYC multipart TAR files |
| `public:flir-aligned` | visible + thermal | `aligned` | 3 | aligned Google Drive file, then public archive fallback |
| `public:rgbtdroneperson` | visible + thermal UAV | `default` | 3 | project Google Drive folder |
| `public:cvc-14` | grayscale visible + FIR | `default` | 1 | ModelScope mirror; local original archive supported |
| `public:nii-cu-mapd` | RGB + FIR UAV | `4-channel` | 1 | official Dropbox archive |

The project does not redistribute dataset bytes. `--accept-license` records that the user reviewed the source terms; it is not a substitute for complying with them.

## One-command preparation

```bash
python etfnet_cli.py download-data \
  --dataset m3fd \
  --output datasets \
  --task obb \
  --accept-license
```

Change `--dataset` to `vedai`, `flir-aligned`, `rgbtdroneperson`, or `cvc-14`. For VEDAI, select `--variant 512` or `--variant 1024`.

A quick deterministic subset can be prepared with:

```bash
python etfnet_cli.py download-data \
  --dataset m3fd \
  --output datasets \
  --limit 250 \
  --accept-license
```

`--limit` is applied per split after deterministic splitting or the official split is loaded. It is intended for pipeline tests, not final reported experiments.

## Train directly from a public alias

```bash
python train.py \
  --data public:m3fd \
  --datasets-dir datasets \
  --accept-dataset-license \
  --epochs 100 \
  --batch 4 \
  --imgsz 640 \
  --device 0
```

Examples:

```bash
# VEDAI 512 RGB–NIR oriented vehicles
python train.py --data public:vedai --dataset-variant 512 \
  --datasets-dir datasets --accept-dataset-license --epochs 100 --device 0

# FLIR aligned visible–thermal
python train.py --data public:flir-aligned --dataset-variant aligned \
  --datasets-dir datasets --accept-dataset-license --epochs 100 --device 0

# UAV tiny-person benchmark
python train.py --data public:rgbtdroneperson \
  --datasets-dir datasets --accept-dataset-license --epochs 100 --device 0

# CVC-14 alignment stress test
python train.py --data public:cvc-14 \
  --datasets-dir datasets --accept-dataset-license --epochs 100 --device 0
```

After the first successful preparation, `data.yaml` is reused without another network request. The source archive/folder, preprocessing options, class names, counts, and checksums are recorded in `SOURCE_MANIFEST.json`.

## Local source fallback

Provider authentication, rate limits, or license forms can make browser download necessary. Use the exact same converter with a local folder or archive:

```bash
python etfnet_cli.py download-data \
  --dataset flir-aligned \
  --archive /data/FLIR_aligned.zip \
  --output datasets \
  --accept-license
```

Directories, `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, and `.7z` inputs are supported. VEDAI's official multipart files are joined automatically when downloaded by the project.

## Canonical output

Every adapter emits:

```text
datasets/<dataset>/processed/<variant>/
├── rgb/images/{train,val,test?}/
├── ir/images/{train,val,test?}/
├── labels/{train,val,test?}/
├── data.yaml
├── SOURCE_MANIFEST.json
└── .preprocess-complete.json
```

RGB and IR files have identical relative names. Differing source encodings are decoded and converted to PNG rather than copied under false extensions. Pair dimensions must match; the converter refuses silent resizing or modality duplication.

## Dataset-specific conversion

### M3FD

The adapter discovers the official `vi/`, `ir/`, `labels/`, and `meta/` layout, preserves `meta/train.txt` and `meta/val.txt`, and converts the existing YOLO boxes to detection or four-corner OBB labels.

### VEDAI

The downloader retrieves official multipart imagery and annotation TAR files. The 14-field annotation is parsed as center, angle, class, instance/flag metadata, followed by `x1..x4` and `y1..y4`. The nine-class order is:

```text
plane, boat, camping-car, car, pickup, tractor, truck, van, other
```

No RGB image is duplicated as NIR. A color-only derivative is rejected.

### FLIR-Aligned

COCO annotations and paired visible/thermal trees are discovered recursively. A common aligned-YOLO layout is also supported. The class names are normalized to person, car, and bicycle.

### RGBTDronePerson

The adapter reads train/validation thermal COCO files, matches corresponding visible images by normalized stem and split, and preserves person, rider, and crowd categories.

### CVC-14

Visible/FIR day/night trees are matched without assuming perfect registration. TXT and VOC XML pedestrian annotations are accepted. The original test tree is mapped to validation for train/validation experiments. Automatic acquisition uses a public ModelScope mirror because a stable unattended original archive URL was not found; `--archive` is the preferred route when the original release is already available.

## Safety and reproducibility

The acquisition layer includes:

- HTTP retry, byte-range resume, and atomic `.part` files;
- Google Drive and ModelScope provider error reporting;
- ZIP CRC checks;
- safe TAR/ZIP/7z extraction with traversal and link/device rejection;
- bounded nested-archive extraction;
- deterministic split generation and subset selection;
- strict pair existence, dimensions, image decoding, label, and class validation;
- identical RGB/IR canonical filenames;
- SHA-256 source and generated-data manifests;
- process locks, stale-lock recovery, and idempotent markers.

## Validation commands

```bash
python etfnet_cli.py check-data \
  --data datasets/m3fd/processed/default/data.yaml \
  --task obb \
  --fingerprint sha256

pytest -q
python tests/run_multidataset_full_pipeline.py --dataset m3fd --export --keep
```

The miniature end-to-end test reproduces each documented source layout locally and runs preprocessing, validation, one real training epoch, validation, paired inference, and optionally TorchScript export. It does not replace a full multi-gigabyte provider transfer or benchmark experiment.
