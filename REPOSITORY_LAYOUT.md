# Repository layout

This release preserves all public Python imports while enforcing a source-control rule:

> Every committed subfolder contains at most 100 files recursively. The repository root is exempt.

## Python source packs

| Folder | Logical public packages |
|---|---|
| `ultralytics/` | package bootstrap, YAML configuration, dataset/model definitions |
| `ultra_modeling/` | `ultralytics.models`, `ultralytics.nn` |
| `ultra_runtime/` | `ultralytics.data`, `ultralytics.engine`, `ultralytics.hub` |
| `ultra_services/` | `ultralytics.utils`, `ultralytics.trackers`, `ultralytics.solutions` |

`ultralytics/__init__.py` extends the package search path in a source checkout. The wheel build maps all source packs back into normal `ultralytics.*` package names, so existing user code and CLI commands remain unchanged.

## Documentation packs

Documentation content is divided among `docs_en_*` and `docs_i18n_*`. Run:

```bash
python docs/prepare_workspace.py
python docs/build_docs.py
```

The first command assembles a temporary `.docs_workspace/`; that generated directory is ignored by the layout audit.

## Enforcement

```bash
python tools/audit_repository_layout.py --json VALIDATION/repository_layout.json
pytest -q tests/test_repository_layout.py
```
