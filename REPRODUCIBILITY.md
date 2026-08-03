# Reproducibility protocol

## Reproduction levels

ETFNet-SJPA-TR exposes two intentionally different modes.

### Deterministic research mode

Use this for ordinary paper experiments. The seed controls Python, NumPy, PyTorch, CUDA, data sampling, and worker initialization. CuDNN benchmarking is disabled and deterministic algorithms are requested in warn-only mode.

```bash
python train.py \
  --data /absolute/path/data.yaml \
  --seed 0 \
  --deterministic \
  --data-fingerprint sha256
```

This mode retains normal augmentations and may use multiple workers. Exact floating-point identity across different GPUs, CUDA/cuDNN versions, or processor architectures is not promised.

### Exact epoch-boundary resume mode

Use this when an interrupted run must reproduce the same trajectory on the same software and hardware environment.

```bash
python train.py \
  --data /absolute/path/data.yaml \
  --epochs 100 \
  --seed 0 \
  --exact-resume \
  --workers 0 \
  --data-fingerprint sha256
```

Resume without changing the original target epoch count:

```bash
python resume.py \
  --checkpoint runs/etfnet/sjpa_train/weights/last-resume.pt \
  --exact-resume
```

Exact mode forces `workers=0`, deterministic execution, and disables stateful mosaic/mixup/copy-paste scheduling. The unstripped checkpoint stores the raw FP32 model, FP32 EMA, optimizer, AMP scaler, scheduler, Python/NumPy/PyTorch/CUDA RNG states, data-loader generator state, source fingerprint, dataset fingerprint, and original training arguments.

Changing the dataset, source tree, target epoch count, or using a checkpoint not created in exact mode causes exact resume to fail rather than silently continue a different experiment.

## Run artifacts

Every training directory contains:

```text
args.yaml                  resolved training arguments
dataset_validation.json   RGB/IR and label preflight report
reproducibility.json       environment, source, model, and data fingerprints
results.csv                epoch metrics
weights/best.pt            deployable checkpoint
weights/last.pt            deployable final checkpoint
weights/last-resume.pt     full resumable checkpoint
```

`best.pt` and `last.pt` are optimizer-stripped deployment artifacts. Resume only from `last-resume.pt` or an unstripped periodic checkpoint.

## Dataset identity

The default metadata fingerprint hashes normalized paths and file sizes while excluding generated `.cache` and `.npy` files. For archival experiments, use:

```bash
--data-fingerprint sha256
```

This content-hashes the dataset YAML, every RGB image, every IR image, and every label file. The checker also validates pair counts, readability, dimensions, OBB polygon schema, finite normalized coordinates, integer class IDs, positive polygon area, and class-range consistency.

## Exact environment

The CPU environment used for the supplied validation report is recorded in `requirements-tested-cpu.txt`. Install it in a clean environment with the CPU PyTorch index contained in that file. GPU users should install a PyTorch build matching their CUDA driver and archive `reproducibility.json` from every run.

Network access is disabled by default. This prevents implicit font downloads, update checks, or online assets from changing startup behavior. Set `ULTRALYTICS_ALLOW_NETWORK=1` explicitly when an online feature is genuinely required. Automatic dependency installation is disabled unless `YOLO_AUTOINSTALL=true` is also set.

## Three-seed paper experiment

```bash
python tools/run_reproducible_experiment.py \
  --data /absolute/path/data.yaml \
  --model ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml \
  --seeds 0 1 2 \
  --epochs 100 \
  --batch 4 \
  --imgsz 640 \
  --device 0 \
  --full-data-hash
```

The runner writes `experiment_plan.json` before training and `experiment_summary.json` afterward, including final per-seed metrics and sample mean/standard deviation.

## Verification commands

```bash
python -m compileall -q etfnet_cli.py ultralytics tools tests
pytest -q
python tests/smoke_pipeline.py
bash tests/full_pipeline_smoke.sh
python tests/exact_resume_check.py
python tools/validate_model_configs.py
```

Synthetic tests verify implementation integrity. They do not establish DroneVehicle, VTUAV-det, or RGBTDronePerson accuracy.
