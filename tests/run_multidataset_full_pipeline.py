"""Process-isolated end-to-end test for all five compact public-dataset adapters.

This intentionally uses miniature local replicas of each documented source layout;
it does not download multi-gigabyte archives during CI.  Every adapter runs through
preprocessing, strict validation, a real one-epoch six-channel OBB training pass,
validation, and paired prediction.  One resulting checkpoint is also exported to
TorchScript and used for inference.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.multidataset_fixtures import create_cvc14, create_flir, create_m3fd, create_rgbtdrone, create_vedai

DATASETS = (
    ("m3fd", "default", create_m3fd),
    ("vedai", "512", create_vedai),
    ("flir-aligned", "aligned", create_flir),
    ("rgbtdroneperson", "default", create_rgbtdrone),
    ("cvc-14", "default", create_cvc14),
)


def run(name: str, command: list[str], work: Path, env: dict[str, str], timeout: int) -> None:
    log = work / f"{name}.log"
    with log.open("wb") as stream:
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
    if code:
        tail = log.read_text(encoding="utf-8", errors="replace")[-12000:]
        raise RuntimeError(f"{name} failed with status {code}\n{tail}")
    print(f"PASS {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dataset", default="all", choices=("all",) + tuple(x[0] for x in DATASETS))
    parser.add_argument("--export", action="store_true", help="Also exercise TorchScript export/prediction")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="etfnet_multidata_e2e_"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    python = sys.executable
    model = ROOT / "tests/fixtures/tiny_sjpa_obb.yaml"
    datasets_root = work / "datasets"
    runs_root = work / "runs"
    summary: dict[str, dict[str, str]] = {}
    try:
        checkpoints: list[Path] = []
        selected_datasets = DATASETS if args.dataset == "all" else tuple(x for x in DATASETS if x[0] == args.dataset)
        for key, variant, fixture in selected_datasets:
            source = fixture(work / "sources" / key)
            prefix = key.replace("-", "_")
            run(f"{prefix}_prepare", [
                python, str(ROOT / "etfnet_cli.py"), "download-data", "--dataset", key,
                "--output", str(datasets_root), "--variant", variant, "--task", "obb",
                "--archive", str(source), "--accept-license", "--link-mode", "copy",
                "--force-preprocess", "--validate",
            ], work, env, args.timeout)
            data = datasets_root / key / "processed" / variant / "data.yaml"
            run(f"{prefix}_check", [python, str(ROOT / "etfnet_cli.py"), "check-data",
                                    "--data", str(data), "--task", "obb", "--fingerprint", "sha256"],
                work, env, args.timeout)
            run(f"{prefix}_train", [
                python, str(ROOT / "train.py"), "--data", str(data), "--model", str(model),
                "--epochs", "1", "--batch", "2", "--imgsz", "64", "--device", "cpu",
                "--workers", "0", "--project", str(runs_root), "--name", prefix,
                "--no-amp", "--no-plots", "--close-mosaic", "0", "--exist-ok",
            ], work, env, args.timeout)
            best = runs_root / prefix / "weights" / "best.pt"
            checkpoints.append(best)
            run(f"{prefix}_val", [
                python, str(ROOT / "test.py"), "--weights", str(best), "--data", str(data),
                "--imgsz", "64", "--batch", "2", "--device", "cpu", "--workers", "0", "--no-plots",
            ], work, env, args.timeout)
            cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
            root = Path(cfg["path"])
            rgb = root / cfg["val"]
            ir = root / cfg["val_ir"]
            run(f"{prefix}_predict", [
                python, str(ROOT / "predict.py"), "--weights", str(best), "--rgb", str(rgb),
                "--ir", str(ir), "--data", str(data), "--imgsz", "64", "--device", "cpu",
                "--no-save", "--project", str(runs_root), "--name", f"{prefix}_predict", "--exist-ok",
            ], work, env, args.timeout)
            summary[key] = {"data": str(data), "checkpoint": str(best)}

        if args.export:
            exported_from = checkpoints[0]
            run("torchscript_export", [python, str(ROOT / "export.py"), "--weights", str(exported_from),
                                       "--format", "torchscript", "--imgsz", "64", "--device", "cpu"],
                work, env, args.timeout)
            exported = exported_from.with_suffix(".torchscript")
            first_key = selected_datasets[0][0]
            first_data = Path(summary[first_key]["data"])
            cfg = yaml.safe_load(first_data.read_text(encoding="utf-8")); root = Path(cfg["path"])
            run("torchscript_predict", [
                python, str(ROOT / "predict.py"), "--weights", str(exported),
                "--rgb", str(root / cfg["val"]), "--ir", str(root / cfg["val_ir"]),
                "--imgsz", "64", "--device", "cpu", "--no-save", "--project", str(runs_root),
                "--name", "torchscript_predict", "--exist-ok",
            ], work, env, args.timeout)
            summary["export"] = {"torchscript": str(exported)}
        report = work / "multidataset_e2e_summary.json"
        report.write_text(json.dumps({"ok": True, "datasets": summary}, indent=2) + "\n", encoding="utf-8")
        print(report.read_text())
    finally:
        if args.keep:
            print(f"Kept workspace: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
