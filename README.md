# ETFNet-SJPA-TR — corrected full RGB–IR pipeline

This repository is a corrected research implementation of the ETFNet RGB–IR UAV detector with a **Selective Spatial Procrustes Alignment with Trust-Region Reliability module (SJPA-TR)**.

It fixes the original repository's train/inference modality swap, fragile RGB–IR path derivation, three-channel disk cache, OBB prediction crash, ignored fusion arguments, modern PyTorch checkpoint loading, six-channel plotting/export, and resumable-checkpoint handling.

> This is a research implementation derived from ETFNet, not a claim that the SJPA-TR extension has already exceeded the paper's published mAP. Real DroneVehicle, VTUAV-det, and RGBTDronePerson training is still required.

## Supported workflows

| Workflow | Supported input |
|---|---|
| Train / validate | Explicit paired RGB and IR dataset splits |
| Predict | Paired files, folders, text lists, videos, streams/cameras, or six-channel NumPy arrays |
| Cache | RAM and complete six-channel disk cache |
| Checkpoint | Deployable `last.pt`/`best.pt` plus unstripped `last-resume.pt` |
| Export | TorchScript tested; ONNX, OpenVINO, and TensorRT code paths included |
| Task | ETFNet configuration uses oriented bounding boxes (OBB) |

The canonical channel convention is fixed everywhere:

```text
HWC before preprocessing: [BGR_RGB, BGR_IR]
CHW model tensor:          [RGB_RGB, RGB_IR]
```

## 1. Installation

Use a clean virtual environment because this project contains its own `ultralytics` Python package.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The exact CPU environment used for the supplied validation report is available as:

```bash
pip install -r requirements-tested-cpu.txt
pip install -e .
```

For ONNX and OpenVINO export:

```bash
pip install -r requirements-export.txt
```

Install the PyTorch/CUDA build appropriate for your GPU before the remaining dependencies when necessary.


## 2. Automatic public-dataset download and preprocessing

The project now supports one-command acquisition and standardized paired preprocessing for **M3FD, VEDAI, FLIR-Aligned, RGBTDronePerson, CVC-14, and NII-CU MAPD**. Install the optional provider/archive packages first:

```bash
pip install -e ".[data]"
```

List the registry and prepare a dataset:

```bash
python etfnet_cli.py list-data

python etfnet_cli.py download-data \
  --dataset m3fd \
  --output datasets \
  --task obb \
  --accept-license
```

Training can trigger acquisition and preprocessing directly:

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

Available aliases are `public:m3fd`, `public:vedai`, `public:flir-aligned`, `public:rgbtdroneperson`, `public:cvc-14`, and `public:nii-cu-mapd`. VEDAI supports `--dataset-variant 512` and `1024`. Use `--dataset-limit 250` for a deterministic quick-run subset.

The pipeline performs provider download/resume, safe extraction, dataset-specific annotation conversion, exact RGB/IR pairing, strict validation, canonical six-channel-compatible layout generation, and a SHA-256 source manifest. Every source also supports an offline `--archive /path/to/source` fallback. Google Drive folder auto-downloads require Python 3.10+ and `gdown>=6`; older Python versions can still preprocess local archives.

See [`PUBLIC_DATASETS.md`](PUBLIC_DATASETS.md) for dataset-specific licenses, providers, class mappings, commands, and troubleshooting.

## 3. Manual dataset layout

The recommended layout is explicit and does not depend on replacing substrings in paths:

```text
dataset/
├── rgb/
│   ├── train/
│   ├── val/
│   └── test/
├── ir/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Every RGB item must have an IR item with the same relative path and filename. OBB labels use the Ultralytics polygon format:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

Coordinates are normalized to `[0, 1]`.

Create `data.yaml`:

```yaml
path: /absolute/path/to/dataset
train: rgb/train
val: rgb/val
test: rgb/test

train_ir: ir/train
val_ir: ir/val
test_ir: ir/test

train_labels: labels/train
val_labels: labels/val
test_labels: labels/test

pairing:
  strict: true
  resize_ir: false

names:
  0: car
  1: truck
  2: bus
  3: van
  4: freight-car
```

An annotated template is provided at:

```text
ultralytics/cfg/datasets/etfnet_rgb_ir_example.yaml
```

Alternative supported mappings are `rgb_root` + `ir_root`, or exact path-component mapping through `pairing.rgb_token` and `pairing.ir_token`.

## 4. Validate the dataset before training

```bash
python etfnet_cli.py check-data --data data.yaml --task obb
```

The checker verifies pair counts, image readability and dimensions, class ranges, finite normalized coordinates, positive OBB polygon area, and duplicate/degenerate labels. Use `--fingerprint sha256` for a content hash of every RGB image, IR image, label, and the dataset YAML.

## 5. Train

```bash
python train.py \
  --data data.yaml \
  --epochs 100 \
  --batch 4 \
  --imgsz 640 \
  --device 0 \
  --workers 4 \
  --project runs/etfnet \
  --name sjpa_dronevehicle
```

Disk caching stores the complete paired sample:

```bash
python train.py --data data.yaml --cache disk --epochs 100 --device 0
```

The default model is:

```text
ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml
```

## 6. Validate

```bash
python test.py \
  --weights runs/etfnet/sjpa_dronevehicle/weights/best.pt \
  --data data.yaml \
  --imgsz 640 \
  --device 0
```

## 7. Paired inference

### Image or folder

```bash
python predict.py \
  --weights runs/etfnet/sjpa_dronevehicle/weights/best.pt \
  --rgb /data/rgb/test \
  --ir /data/ir/test \
  --data data.yaml \
  --device 0
```

### Video

```bash
python predict.py \
  --weights best.pt \
  --rgb rgb_video.mp4 \
  --ir ir_video.mp4 \
  --device 0
```

The paired videos are consumed frame by frame. They must have compatible dimensions; use `--pair-resize` only when intentional resizing is acceptable.

### Cameras or network streams

```bash
python predict.py --weights best.pt --rgb 0 --ir 1 --device 0
```

Or use two RTSP/HTTP stream URLs. Live-stream code is included, but synchronization quality still depends on camera timestamps and capture hardware.

### Python API

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(
    source="/data/rgb/test",
    ir_source="/data/ir/test",
    data="data.yaml",
    imgsz=640,
    device=0,
    ch=6,
)
```

A pre-concatenated NumPy input may also be supplied with shape `H×W×6` in `[BGR_RGB, BGR_IR]` order.

## 8. Resume training safely

At the end of training, three files are produced:

```text
best.pt         deployable, optimizer stripped
last.pt         deployable, optimizer stripped
last-resume.pt  full optimizer/EMA state for resume
```

Resume or extend a normal run to a total target of 150 epochs:

```bash
python resume.py \
  --checkpoint runs/etfnet/sjpa_dronevehicle/weights/last-resume.pt \
  --epochs 150 \
  --device 0
```

For byte-exact epoch-boundary resume on the same software/hardware environment, create the original run with `--exact-resume`, then resume without changing the original target epoch count:

```bash
python train.py --data data.yaml --epochs 100 --exact-resume --workers 0
python resume.py --checkpoint runs/etfnet/sjpa_train/weights/last-resume.pt --exact-resume
```

Exact mode stores and restores the raw FP32 model, FP32 EMA, optimizer, AMP scaler, scheduler, Python/NumPy/PyTorch/CUDA RNG states, and data-loader generator. It also rejects changed source or dataset fingerprints. Do not resume from the stripped `best.pt` or `last.pt`. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## 9. Export

TorchScript:

```bash
python export.py --weights best.pt --format torchscript --imgsz 640 --device cpu
```

ONNX:

```bash
python export.py --weights best.pt --format onnx --imgsz 640 --device 0 --dynamic
```

OpenVINO:

```bash
python export.py --weights best.pt --format openvino --imgsz 640 --device cpu
```

TensorRT engine:

```bash
python export.py --weights best.pt --format engine --imgsz 640 --device 0 --half
```

The exporter uses a six-channel example input and freezes SJPA running statistics for portable graph export. TorchScript was executed end to end in the supplied test environment. Other formats require their platform-specific runtimes and are documented as implemented but not runtime-certified here.

## 10. Smoke tests

Run lightweight graph/data tests:

```bash
python tests/smoke_pipeline.py
```

Run the complete one-epoch train → validate → image/video predict → resume → TorchScript workflow:

```bash
bash tests/full_pipeline_smoke.sh
```

Verify exact resume and every shipped model configuration:

```bash
python tests/exact_resume_check.py
python tools/validate_model_configs.py
```

Run a fixed three-seed paper experiment and aggregate final metrics:

```bash
python tools/run_reproducible_experiment.py --data data.yaml --seeds 0 1 2 --device 0 --full-data-hash
```

See [`TEST_REPORT.md`](TEST_REPORT.md), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), and [`CHANGELOG_FIXES.md`](CHANGELOG_FIXES.md).

### Extended isolated validation

Run the four validation stages separately so that training workloads are isolated and every stage produces a machine-readable JSON report:

```bash
python tools/run_extended_validation.py --stage quick
python tools/run_extended_validation.py --stage exact-resume
python tools/run_extended_validation.py --stage proxy --timeout 1200
python tools/run_extended_validation.py --stage pipeline --timeout 1200
```

The `quick` stage compiles the source, runs the full unit/integration test suite, executes a paired-data smoke test, and constructs every shipped ETFNet model YAML. The other stages verify tensor-identical epoch-boundary resume, a five-seed controlled robustness proxy, and the complete train/validate/image-video-predict/TorchScript pipeline.

### Mathematical and paper support

- [`MATHEMATICAL_APPENDIX.md`](MATHEMATICAL_APPENDIX.md): exact optimization formulation, proofs, bounds, invariance properties, complexity, and export approximation.
- [`PAPER_SUPPORT_Q1.md`](PAPER_SUPPORT_Q1.md): literature positioning, defensible novelty wording, contribution statement, mandatory baselines, and claims that must be avoided.
- [`EXPERIMENT_PROTOCOL_Q1.md`](EXPERIMENT_PROTOCOL_Q1.md): public-benchmark protocol, statistical analysis, robustness tests, deployment measurements, and artifact checklist.

The production method is **sequential**: running groupwise whitening, closed-form orthogonal channel alignment, Procrustes-optimal finite translation selection, and bounded reliability correction. It must not be described as a jointly solved continuous spatial-channel optimization problem.


## Offline-by-default behavior

The repository performs no implicit update checks, font downloads, or dataset downloads. This prevents network state from changing reproducibility or startup time. Set `ULTRALYTICS_ALLOW_NETWORK=1` before process startup only when an upstream online feature is intentionally required. Automatic package installation is also disabled; opt in separately with `YOLO_AUTOINSTALL=true`.

## Shipped model configurations

All YAML files under `ultralytics/cfg/models/etfnet/` construct and execute. The specialized `train.py`, `test.py`, and `predict.py` entry points are intentionally fixed to the paired six-channel RGB–IR pipeline. The two three-channel files (`etfnet_yolo11.yaml` and `noCAFEM_noTGF.yaml`) are unimodal reference graphs and should be run with the standard single-modality Ultralytics data path rather than the paired CLI.

For controlled six-channel ablation, use `etfnet_dualstream_noCAFEM_noTGF.yaml`; it retains both inputs and the same OBB pipeline while removing CAFEM and cross-modal fusion modules. Previously broken `GPT`, P3-P4-P5 TGF, and elementwise-add configurations were repaired and are covered by automated model-construction tests. Class count is overridden from the selected dataset at training time. `etfnet_P2_CAFEM_GOCI.yaml` is retained only as a non-selected experimental comparison; the default and supported proposal is `etfnet_P2_CAFEM_SJPA.yaml`.

## 11. Scientific evaluation requirements

Synthetic smoke tests verify software correctness only. They do not establish superiority over ETFNet or other RGB–IR detectors. A publication claim requires matched training on the official splits, at least three seeds, corruption/misalignment tests, parameters, GFLOPs, GPU latency, and embedded-device latency.

## Citation and license

The base architecture is from:

```bibtex
@article{nguyen2026etfnet,
  title   = {ETFNet: An Efficient Transformer-Based RGB--IR Fusion Network for UAV Object Detection},
  author  = {Nguyen, Thi Lan and Tran, Cao Truong and Nguyen, Dinh Tan},
  journal = {Information Fusion},
  year    = {2026},
  pages   = {104658},
  doi     = {10.1016/j.inffus.2026.104658}
}
```

This repository remains under the GNU Affero General Public License v3.0. See [`LICENSE.txt`](LICENSE.txt).

## Organized source layout

Version 6 splits the implementation and documentation into bounded source packs while preserving all existing imports, commands, model YAMLs, public-dataset adapters, training, validation, prediction, export, and resume behavior. Every committed subfolder contains at most 100 files recursively. See [`REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md) and run `python tools/audit_repository_layout.py` to verify the rule.

## Recheck v9: DCSPF-Guard experimental candidate

The repository now includes a stricter, fair-backbone fusion candidate:

- `ultralytics/cfg/models/etfnet/etfnet_P2_C3k2_DCSPF_Guard.yaml`
- `RECHECK_V9_REPORT.md`
- `DCSPF_MATHEMATICAL_APPENDIX.md`
- `VALIDATION/recheck_v9/fair_component_seeded_5seed.json`
- `tools/certify_candidate.py`

DCSPF routes between raw RGB–IR coordinates and the corrected Procrustes-canonical coordinates using bounded coherence, pooled-domain typicality, and modality dominance. It is experimental: the controlled proxy supports superiority over direct concatenation, but not superiority over corrected SJPA on every seed or aggregate robustness measure. Full VEDAI training is required before making a highest-mAP claim.
