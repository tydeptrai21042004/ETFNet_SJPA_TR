# Corrective implementation audit

## Data integrity

- Fixed the train/inference RGB–IR branch swap by enforcing one six-channel convention.
- Replaced substring-based IR path derivation with explicit roots, manifests, or exact path-component mapping.
- Added strict pair count, filename, readability, dimension, and label validation.
- Preserved RGB/IR alignment after label filtering and sorted manifest loading.
- Made disk-cache identity depend on RGB path, IR path, and resize policy.
- Changed disk and RAM caches to store the complete six-channel pair.
- Added stale-cache invalidation when either modality changes.
- Excluded generated caches from scientific dataset fingerprints.

## Model/configuration correctness

- Added the parameter-free SJPA-TR model and full parser integration.
- Replaced circular `torch.roll` alignment with zero-padded non-wrapping shifts.
- Fixed TGF parsing so all YAML hyperparameters reach the module.
- Repaired the undefined `GPT` ablation using the supported TGF implementation.
- Repaired the invalid P3-P4-P5 TGF argument list.
- Added parser validation for elementwise `Add` inputs.
- Standardized class count for DroneVehicle ablations while retaining unimodal channel counts.
- Made model scale selection explicit.
- Verified every shipped ETFNet YAML constructs and completes a forward pass.

## Training readiness

- Added full dataset preflight before optimizer initialization.
- Added deterministic Python, NumPy, PyTorch, CUDA, sampler, and worker seeding.
- Added epoch-aware deterministic sampling for single-process and distributed training.
- Updated AMP usage to the current `torch.amp` API.
- Added atomic checkpoint writes.
- Added an unstripped `last-resume.pt` every epoch.
- Preserved FP32 raw model/EMA, optimizer, scaler, scheduler, and RNG states.
- Fixed resume loading to restore the raw training model rather than EMA weights with a mismatched optimizer.
- Fixed scheduler/save ordering for epoch-boundary equivalence.
- Added exact-resume compatibility checks and source/dataset fingerprint enforcement.
- Invalidated SJPA evaluation caches when EMA parameters change.
- Fixed modern PyTorch checkpoint loading and final optimizer stripping.

## Validation, inference, and export

- Fixed OBB predictor return handling.
- Added paired files, folders, manifests, videos, cameras, and stream sources.
- Added synchronized six-channel preprocessing and safe RGB visualization.
- Forced OBB task selection for exported TorchScript inference.
- Added tested TorchScript export and paired inference.
- Removed implicit network downloads and update checks by default.

## Packaging and reproducibility

- Added bounded runtime dependencies and an exact tested CPU lock file.
- Added a Conda environment template and installable wheel configuration.
- Added machine-readable source, model, environment, and dataset manifests.
- Added a three-seed experiment runner and metric aggregation.
- Added unit, model-configuration, smoke, full-pipeline, disk-cache, and exact-resume tests.

## v4 public-data pipeline

- Added an official NII-CU MAPD downloader with HTTP resume, retry, archive validation, safe extraction, and preparation locks.
- Added deterministic conversion from the official tabular boxes to YOLO OBB or detect labels.
- Added standardized RGB/IR layout creation using hard links, symlinks, or copies.
- Added source DOI, license, archive SHA-256, filtering choices, and conversion counts to a machine-readable manifest.
- Added `list-data`, `download-data`, and `public:nii-cu-mapd` CLI support.
- Added public-dataset unit, security, resume-download, idempotency, validation, and end-to-end training tests.

## v6 — compact public multi-dataset acquisition and preprocessing

- Added automatic adapters for M3FD, VEDAI 512/1024, FLIR-Aligned, RGBTDronePerson, and CVC-14.
- Added official/direct, Google Drive, and ModelScope acquisition backends with local source fallback.
- Added safe ZIP/TAR/7z extraction, multipart VEDAI joining, resumable HTTP, process locking, and manifests.
- Added YOLO, COCO, VOC, simple TXT, and VEDAI-oriented annotation conversion.
- Added one strict canonical paired RGB/IR layout for training, validation, prediction, caching, and export.
- Added deterministic split/subset controls and source/generated SHA-256 fingerprints.
- Added 18 new multi-dataset tests; complete suite now reports 78 passing tests.
- Added actual one-epoch end-to-end miniature runs for all five datasets and TorchScript coverage on M3FD.
- Added optional `[data]` dependencies and the installed `etfnet` command.
