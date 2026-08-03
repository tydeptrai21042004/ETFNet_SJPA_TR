# ETFNet-SJPA-TR training-readiness and reproducibility report

**Validation date:** 2026-08-03  
**Environment:** Linux, Python 3.13.5, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu  
**Synthetic fixture:** paired RGB/IR OBB data, 4 training pairs, 2 validation pairs, 96×96 source images  
**Purpose:** software integrity and deterministic-resume verification; not an accuracy benchmark

## Readiness assessment

| Area | Original audit | Corrected release | Remaining reason it is not 10/10 |
|---|---:|---:|---|
| Training readiness | 4/10 | **9/10** | CUDA AMP and multi-GPU DDP could not be executed in the CPU-only validation environment |
| Reproducibility | 2/10 | **9/10** | Cross-hardware bitwise identity cannot be guaranteed; official benchmark data are not redistributed |

The scores describe implementation readiness in the tested environment, not benchmark accuracy.

## Automated results

| Check | Result | Evidence |
|---|---|---|
| Python compilation | PASS | Entry points, package, tools, and tests compile |
| Unit/integration tests | **9 passed** | `pytest -q`, including all shipped model YAMLs |
| Lightweight paired pipeline | PASS | Pairing, channel order, cache identity, forward/backward |
| Dataset preflight | PASS | Pair/readability/dimension/OBB schema and fingerprints |
| All ETFNet model configurations | **12/12 PASS** | Build and 64×64 forward execution |
| Full training pipeline | PASS | Train → val → image/video predict → resume → TorchScript |
| Exact epoch-boundary resume | **PASS, tensor-identical** | Raw model and EMA have zero differing tensors |
| Disk cache | PASS | Six generated cache arrays, all shape 96×96×6 |
| Wheel build/import | PASS | Wheel installed separately and executed six-channel model |

Machine-readable outputs are stored in:

```text
VALIDATION/exact_resume.json
VALIDATION/model_configs.json
VALIDATION/disk_cache.json
```

## Corrected training-critical failures

| Original failure | Corrected behavior |
|---|---|
| RGB and IR streams swapped between training and prediction | One canonical train/val/predict/export channel order |
| Dataset YAML absent and paths inferred by string replacement | Explicit RGB, IR, and label roots plus validated manifests |
| Disk cache contained only RGB | Pair-specific six-channel cache with stale-source checking |
| Missing dependencies and unbounded environment | Bounded runtime requirements and exact tested CPU lock file |
| Modern PyTorch checkpoint failure | Compatibility loader and current AMP/checkpoint APIs |
| Optimizer resumed against EMA weights | Resume restores the raw FP32 training model |
| No exact RNG/scheduler recovery | Full optimizer/scaler/scheduler/RNG/dataloader state stored |
| Non-atomic checkpoints | Temporary-file write followed by atomic replacement |
| TGF YAML arguments discarded | Parser forwards every declared TGF parameter |
| Three model YAMLs did not construct | Undefined GPT, invalid TGF args, and Add parser repaired |
| OBB prediction unpack crash | OBB predictor uses the correct result contract |
| Exported model guessed the wrong task | CLI forces OBB task and preserves export metadata |
| Hidden network-dependent startup/download behavior | Offline by default; explicit opt-in only |

## Exact-resume result

A two-epoch CPU run was executed in two forms:

1. uninterrupted for two epochs;
2. intentionally interrupted immediately after the epoch-one checkpoint and resumed from `last-resume.pt`.

The final comparison produced:

```json
{
  "continuous_best_fitness": 0.18905,
  "continuous_epoch": 1,
  "continuous_updates": 3,
  "ema_differences": [],
  "exact": true,
  "model_differences": [],
  "resumed_best_fitness": 0.18905,
  "resumed_epoch": 1,
  "resumed_updates": 3
}
```

This demonstrates exact epoch-boundary continuation in the tested CPU environment. It does not imply bitwise identity across different GPU models, CUDA/cuDNN builds, or processor architectures.

## Full workflow verified

- Dataset validation and metadata/SHA-256 fingerprint modes.
- One-epoch six-channel OBB training.
- Standalone validation.
- Paired image-folder prediction.
- Paired synchronized video prediction.
- Normal resume with relocated-data override support.
- Exact resume with compatibility enforcement.
- `best.pt` and `last.pt` deployment checkpoints.
- Unstripped `last-resume.pt` checkpoint.
- TorchScript export and paired TorchScript inference.
- Complete six-channel RAM and disk caching.
- Source/model/data/environment manifests.
- Three-seed experiment-plan and metric-aggregation tooling.

## Implemented but not runtime-certified here

| Pipeline | Limitation of the validation environment |
|---|---|
| CUDA mixed-precision training | CPU-only runtime |
| Multi-GPU DDP | No multi-GPU node |
| TensorRT engine | No NVIDIA CUDA/TensorRT runtime |
| ONNX Runtime and OpenVINO runtime | Optional runtimes not installed in the final test environment |
| Live camera/RTSP synchronization | No paired capture hardware or streams |
| Embedded latency/energy | No Jetson or UAV onboard platform |

These paths remain implemented, but the release does not label them as experimentally certified.

## Scientific limitation

The synthetic fixture validates code and reproducibility only. It does not demonstrate that SJPA-TR exceeds ETFNet on DroneVehicle, VTUAV-det, or RGBTDronePerson. Such a claim requires official splits, matched hyperparameters, at least three seeds, statistical uncertainty, corruption/misalignment tests, and measured GPU/embedded latency.
