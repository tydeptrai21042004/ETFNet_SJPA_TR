# ETFNet–SJPA-TR v5 Multi-Dataset Validation Report

**Release:** `8.0.238+etfnetsjpa.5`  
**Date:** 2026-08-03  
**Scope:** automatic acquisition and canonical preprocessing for M3FD, VEDAI, FLIR-Aligned, RGBTDronePerson, and CVC-14, while retaining NII-CU MAPD support.

## 1. Supported dataset registry

| Dataset key | Variants | Modality | Output tasks | Automatic provider | Local source fallback |
|---|---|---|---|---|---|
| `m3fd` | `default` | RGB + thermal | detect, OBB | Google Drive folder | folder/ZIP/TAR/7z |
| `vedai` | `512`, `1024` | RGB + NIR | detect, OBB | official GREYC multipart TAR | folder/ZIP/TAR/7z |
| `flir-aligned` | `aligned` | RGB + thermal | detect, OBB | Google Drive file, HTTP fallback | folder/ZIP/TAR/7z |
| `rgbtdroneperson` | `default` | RGB + thermal UAV | detect, OBB | Google Drive folder | folder/ZIP/TAR/7z |
| `cvc-14` | `default` | grayscale visible + FIR | detect, OBB | ModelScope mirror | folder/ZIP/TAR/7z |
| `nii-cu-mapd` | `4-channel`, `rgb-t` | RGB + FIR UAV | detect, OBB | official HTTP archive | folder/ZIP/TAR/7z |

Every adapter emits the same strict paired layout:

```text
processed/<variant>/
├── rgb/images/{train,val,test?}/
├── ir/images/{train,val,test?}/
├── labels/{train,val,test?}/
├── data.yaml
├── SOURCE_MANIFEST.json
└── .preprocess-complete.json
```

## 2. Automated test result

Command:

```bash
python -m compileall -q .
pytest -q
```

Result:

```text
78 passed, 3 warnings in 6.19 s
```

The warnings are PyTorch deprecation notices for `torch.jit.trace`; no test failed.

### Dataset-specific coverage

- Detection and four-corner OBB conversion for all five new datasets.
- License acceptance enforcement.
- Public-alias resolution and idempotent reuse.
- M3FD YOLO metadata/split handling.
- VEDAI 14-field coordinate order and nine-class mapping.
- FLIR and RGBTDronePerson COCO category and pair mapping.
- CVC-14 TXT/VOC and train/test tree mapping.
- Mixed image-extension canonicalization to valid PNG pairs.
- Pair existence, equal dimensions, image decodability, class range, and geometry validation.
- Deterministic splitting, subset limits, manifests, and fingerprints.
- ZIP/TAR/7z traversal and unsafe-entry rejection.
- HTTP interrupted-download resume.
- Google Drive folder/file provider behavior using mocked provider responses.
- Rejection of old `gdown` folder behavior that cannot reliably fetch large folders.
- ModelScope provider dispatch using a mocked snapshot response.

## 3. Full paired-model pipeline evidence

The exact documented miniature source layout for each dataset was processed through:

```text
source layout
→ converter
→ canonical six-channel paired dataset
→ preflight validation
→ one actual OBB training epoch
→ validation
→ paired RGB/IR prediction
```

| Dataset | Prepare | Preflight | Train | Validate | Paired predict |
|---|---:|---:|---:|---:|---:|
| M3FD | PASS | PASS | PASS | PASS | PASS |
| VEDAI | PASS | PASS | PASS | PASS | PASS |
| FLIR-Aligned | PASS | PASS | PASS | PASS | PASS |
| RGBTDronePerson | PASS | PASS | PASS | PASS | PASS |
| CVC-14 | PASS | PASS | PASS | PASS | PASS |

M3FD additionally passed TorchScript export and paired TorchScript prediction. Individual successful run summaries are retained in `VALIDATION/multidata_e2e_v5.json`.

A single sequential invocation covering all five datasets exceeded the external execution harness time limit; the same stages passed in the five individual invocations. This is a harness-duration limitation, not a reported individual pipeline failure.

## 4. Wheel/package validation

Wheel built:

```text
etfnet_sjpa_tr-8.0.238+etfnetsjpa.5-py3-none-any.whl
```

The built wheel was installed into a clean target directory and tested outside the source tree. The following passed:

- Importing `ultralytics` from the wheel.
- Version check: `8.0.238+etfnetsjpa.5`.
- Registry discovery for all six public datasets.
- `python -m etfnet_cli list-data --json` from the installed wheel.

Acquisition extras are intentionally optional and can be installed with:

```bash
pip install -e ".[data]"
# or
pip install -r requirements-data.txt
```

## 5. Acquisition and safety behavior

- Resumable HTTP downloads use atomic `.part` files and byte-range requests.
- Google Drive failures explain rate-limit/authentication recovery and local fallback.
- VEDAI multipart archives are downloaded from the official server, concatenated, and verified before extraction.
- ModelScope is used only as the automated CVC-14 mirror route; original local releases remain supported.
- ZIP CRC checks are performed.
- TAR, ZIP, and 7z extraction rejects absolute paths, parent traversal, links, and special/device entries.
- Nested extraction depth is bounded.
- The converter never silently duplicates RGB as IR/NIR and never silently resizes unequal modality pairs.
- Source and generated outputs are fingerprinted in machine-readable manifests.

## 6. Validation boundary

The full multi-gigabyte public archives were not downloaded in this environment. The test suite validates downloader mechanics, provider dispatch, archive safety, source-layout interpretation, preprocessing, and the real model pipeline using exact miniature source replicas. Consequently:

- **Code path and pipeline runability:** validated.
- **Provider URLs/identifiers:** configured and documented.
- **Full external transfers under every provider account/rate-limit state:** not certified here.
- **Real benchmark mAP:** not claimed by this release.
- **Dataset licenses:** users must review and comply with each source's terms.

For final scientific results, retain `SOURCE_MANIFEST.json`, use official splits where available, report source versions/checksums, and do not mix NIR datasets such as VEDAI with thermal datasets under one undifferentiated claim.
