# ETFNet-SJPA-TR corrected repository validation

**Version:** `8.0.238+etfnetsjpa.7`  
**Validation date:** 2026-08-04

## Why the earlier Kaggle cells failed

The initial validation concentrated on unit tests, model construction, CPU synthetic data, and repository organization. Kaggle later exposed environment-specific paths that were not covered end-to-end: official VEDAI sparse annotations, optional W&B and Ray callbacks, CUDA autocast around SVD/eigendecomposition, and RGB-only baselines being forced into a six-channel pipeline. Those problems belonged in the repository source rather than in an increasingly large notebook-only monkey patch.

This release moves the fixes into the repository and leaves the Kaggle cell as an experiment driver.

## Source corrections

- Official VEDAI sparse class IDs and `annotation512.txt` metadata handling.
- Deterministic VEDAI `_co`/`_ir` pairing and nine-class schema.
- FP32 `eigh`/SVD blocks under outer autocast and running-statistics dtype safety for GOCI/SJPA.
- W&B path sanitization and failure isolation.
- Ray Tune compatibility with old and new APIs; ordinary training no longer depends on a Ray session.
- TensorBoard graph input uses the model's real channel count.
- CLI infers 3-channel RGB versus 6-channel RGB–IR models.
- Explicit local `tests` package and new regression coverage.

## Executed validation

| Check | Result |
|---|---:|
| Full pytest suite | **87 passed**, 3 trace deprecation warnings |
| Kaggle regression subset | **6 passed** |
| Shipped model YAML build/forward validation | **12/12 passed** |
| Repository layout audit | **0** folders above the 100-file limit |
| Synthetic one-epoch training matrix | **10/10 logical configurations passed** |
| Tiny full pipeline | train, validation, paired prediction, TorchScript export, exported-model prediction passed |
| Optional integration failure simulation | training completed with failing W&B and new-style Ray stubs |
| Wheel build and isolated import | passed |

The ten logical one-epoch training configurations were RGB P3–P5, RGB YOLO11 P2–P5, dual concatenation, dual addition, CAFEM only, TGF only, corrected ETFNet-TGF, GOCI without spatial alignment, SJPA without reliability, and full SJPA-TR.

## Kaggle evidence and remaining boundary

The supplied Kaggle log reached a complete full-VEDAI proposal epoch and validation, then failed only in the old Ray callback after `on_fit_epoch_end`. The corrected callback has a direct regression test and a simulated-training integration test.

The final source package was validated in the available CPU environment. The corrected CUDA AMP path is exercised by the delivered Kaggle cell before the real benchmark, but the complete ten-model, ten-epoch VEDAI run was not executed locally because this environment has no CUDA GPU and no retained public VEDAI archive. No mAP superiority claim is made.
