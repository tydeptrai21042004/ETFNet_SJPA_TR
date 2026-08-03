# ETFNet–SJPA-TR v6 repository reorganization report

## Requirement

The repository may contain any total number of files, but every committed subfolder must contain at most 100 files recursively. The repository root is exempt.

## Result

- Total source-repository files: approximately 775.
- Subfolders checked: more than 140.
- Folder-limit violations: **0**.
- Largest committed subfolder: `docs_en_main/`, with **89 files**.
- `ultralytics/` now contains **39 files recursively**, reduced from 182.

## Compatibility-preserving source packs

| Physical source pack | Existing logical imports retained |
|---|---|
| `ultralytics/` | package bootstrap and all YAML resources under `ultralytics.cfg` |
| `ultra_modeling/` | `ultralytics.models`, `ultralytics.nn` |
| `ultra_runtime/` | `ultralytics.data`, `ultralytics.engine`, `ultralytics.hub` |
| `ultra_services/` | `ultralytics.utils`, `ultralytics.trackers`, `ultralytics.solutions` |

The source checkout extends `ultralytics.__path__` before public imports are loaded. The wheel build maps these physical packs back into ordinary `ultralytics.*` packages. Existing application code, model YAMLs, checkpoints, and CLI commands do not need to change.

## Documentation organization

The 478 documentation-content files are divided among bounded `docs_en_*` and `docs_i18n_*` packs. `docs/prepare_workspace.py` assembles them into a temporary `.docs_workspace/` before MkDocs is run. The assembled English tree contains the original 218 files, and each translated language contains its original 26 files.

## Preserved functionality

- Six-channel RGB–IR model construction and forward/backward execution.
- CAFEM, TGF, GOCI, and SJPA model variants.
- Training, validation, exact resume, prediction, and export code.
- RGB–IR pairing, preflight checks, cache handling, and preprocessing.
- Automatic/public-data adapters for NII-CU MAPD, M3FD, VEDAI, FLIR-Aligned, RGBTDronePerson, and CVC-14.
- Detection and OBB conversion.
- TorchScript export and paired inference.
- Existing `YOLO`, `yolo`, `ultralytics`, `etfnet`, and `etfnet-sjpa` entry points.

## Validation

| Validation | Result |
|---|---:|
| Pytest suite | **81 passed** |
| ETFNet YAML build/forward sweep | **12/12 passed** |
| Source public-import compatibility | **Passed** |
| Six-channel SJPA model forward | **Passed** |
| Public-dataset adapter tests | **Passed for all requested datasets** |
| Synthetic dataset preflight | **Passed** |
| One-epoch training | **Passed** |
| Validation | **Passed** |
| Paired prediction | **Passed** |
| TorchScript export | **Passed** |
| TorchScript paired prediction | **Passed** |
| Wheel build | **Passed** |
| Isolated wheel installation and model forward | **Passed** |
| Documentation workspace reconstruction | **478/478 content files assembled** |
| Recursive folder-limit audit | **0 violations; maximum 89** |

## Enforcement

Run:

```bash
python tools/audit_repository_layout.py --json VALIDATION/repository_layout_v6.json
pytest -q tests/test_repository_layout.py tests/test_split_source_layout.py
```

A GitHub Actions workflow at `.github/workflows/repository-layout.yml` rejects future changes that make any committed subfolder exceed 100 recursive files.
