"""Run and summarize a fixed multi-seed ETFNet-SJPA-TR experiment.

The experiment plan, exact commands, dataset SHA-256 fingerprint, source version,
and final per-seed metrics are written under the project directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.data.rgb_ir_check import validate_dataset  # noqa: E402
from ultralytics.utils.reproducibility import sha256_file, source_tree_fingerprint  # noqa: E402

DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml"


def _final_metrics(csv_path: Path) -> dict[str, float]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No metric rows in {csv_path}")
    result = {}
    for key, value in rows[-1].items():
        if value is None or value.strip() == "":
            continue
        try:
            result[key.strip()] = float(value)
        except ValueError:
            pass
    return result


def _aggregate(per_seed: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    common = set.intersection(*(set(metrics) for metrics in per_seed.values())) if per_seed else set()
    summary = {}
    for key in sorted(common):
        values = [metrics[key] for metrics in per_seed.values()]
        summary[key] = {
            "mean": statistics.fmean(values),
            "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", choices=("none", "ram", "disk"), default="none")
    parser.add_argument("--project", type=Path, default=Path("runs/etfnet_reproducible"))
    parser.add_argument("--prefix", default="sjpa")
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full-data-hash", action="store_true")
    args = parser.parse_args()

    data = args.data.resolve()
    model = args.model.resolve()
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    fingerprint_mode = "sha256" if args.full_data_hash else "metadata"
    data_report = validate_dataset(str(data), task="obb", fingerprint=fingerprint_mode)
    if not data_report["ok"]:
        raise SystemExit(json.dumps(data_report, indent=2))

    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(data),
        "data_yaml_sha256": sha256_file(data),
        "dataset_fingerprint": data_report["fingerprint"],
        "model": str(model),
        "model_sha256": sha256_file(model),
        "source": source_tree_fingerprint(ROOT / "ultralytics"),
        "seeds": args.seeds,
        "settings": {
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "cache": args.cache,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "weight_decay": args.weight_decay,
            "amp": args.amp,
            "deterministic": True,
        },
        "commands": [],
    }
    plan_path = project / "experiment_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("ULTRALYTICS_ALLOW_NETWORK", "0")
    per_seed = {}
    for seed in args.seeds:
        name = f"{args.prefix}_seed{seed}"
        command = [
            sys.executable,
            str(ROOT / "train.py"),
            "--data", str(data),
            "--model", str(model),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--imgsz", str(args.imgsz),
            "--device", args.device,
            "--workers", str(args.workers),
            "--cache", args.cache,
            "--project", str(project),
            "--name", name,
            "--seed", str(seed),
            "--optimizer", args.optimizer,
            "--lr0", str(args.lr0),
            "--weight-decay", str(args.weight_decay),
            "--data-fingerprint", fingerprint_mode,
            "--deterministic",
            "--amp" if args.amp else "--no-amp",
            "--plots" if args.plots else "--no-plots",
        ]
        plan["commands"].append(command)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        per_seed[str(seed)] = _final_metrics(project / name / "results.csv")

    summary = {
        "plan": str(plan_path),
        "per_seed": per_seed,
        "aggregate": _aggregate(per_seed),
    }
    (project / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
