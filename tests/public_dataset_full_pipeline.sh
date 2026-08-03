#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
WORK="$(mktemp -d -t etfnet_public_data_XXXXXX)"
KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1
cleanup() {
  if [[ "$KEEP" -eq 1 ]]; then echo "Kept: $WORK"; else rm -rf "$WORK"; fi
}
trap cleanup EXIT

ARCHIVE="$WORK/NII_CU_MAPD_dataset.zip"
python - "$ARCHIVE" <<'PY'
from pathlib import Path
import sys
from tests.public_dataset_fixture import create_nii_cu_archive
create_nii_cu_archive(Path(sys.argv[1]))
PY

DATASETS="$WORK/datasets"
RUNS="$WORK/runs"
MODEL="$ROOT/tests/fixtures/tiny_sjpa_obb.yaml"

python "$ROOT/etfnet_cli.py" download-data --dataset nii-cu-mapd --output "$DATASETS" \
  --archive "$ARCHIVE" --accept-license --variant 4-channel --task obb --exclude-bad --limit 2
DATA="$DATASETS/nii-cu-mapd/processed/4-channel/data.yaml"
test -f "$DATA"

# Exercise the public:<name> alias rather than passing the generated YAML.
python "$ROOT/train.py" --data public:nii-cu-mapd --datasets-dir "$DATASETS" --model "$MODEL" \
  --epochs 1 --batch 2 --imgsz 64 --device cpu --workers 0 --project "$RUNS" --name public \
  --no-amp --no-plots --data-fingerprint sha256
BEST="$RUNS/public/weights/best.pt"
test -f "$BEST"

python "$ROOT/test.py" --weights "$BEST" --data public:nii-cu-mapd --datasets-dir "$DATASETS" \
  --imgsz 64 --batch 2 --device cpu --workers 0 --no-plots
python "$ROOT/predict.py" --weights "$BEST" \
  --rgb "$DATASETS/nii-cu-mapd/processed/4-channel/rgb/images/val" \
  --ir "$DATASETS/nii-cu-mapd/processed/4-channel/ir/images/val" \
  --data "$DATA" --imgsz 64 --device cpu --no-save --project "$RUNS" --name predict


echo "PASS: public archive -> preprocessing -> alias train -> val -> paired prediction"
