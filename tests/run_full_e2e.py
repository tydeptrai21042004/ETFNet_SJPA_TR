"""Process-isolated end-to-end smoke test for the corrected six-channel pipeline."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str], work: Path, env: dict[str, str], timeout: int) -> None:
    log = work / f"{name}.log"
    with log.open("wb") as stream:
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
            status = completed.returncode
        except subprocess.TimeoutExpired:
            status = 124
    if status:
        tail = log.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"Step {name!r} failed with status {status}.\n{tail}")
    print(f"PASS {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="etfnet_full_e2e_"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    python = sys.executable
    try:
        dataset = work / "dataset"
        runs = work / "runs"
        model = ROOT / "tests/fixtures/tiny_sjpa_obb.yaml"
        run_step("fixture", [python, str(ROOT / "tests/create_synthetic_fixture.py"), str(dataset)],
                 work, env, args.timeout)
        data = dataset / "data.yaml"
        run_step("preflight", [python, str(ROOT / "etfnet_cli.py"), "check-data", "--data", str(data),
                               "--task", "obb"], work, env, args.timeout)
        run_step("train", [python, str(ROOT / "train.py"), "--data", str(data), "--model", str(model),
                           "--epochs", "1", "--batch", "2", "--imgsz", "64", "--device", "cpu",
                           "--workers", "0", "--project", str(runs), "--name", "e2e", "--no-amp",
                           "--no-plots"], work, env, args.timeout)
        best = runs / "e2e/weights/best.pt"
        run_step("validate", [python, str(ROOT / "test.py"), "--weights", str(best), "--data", str(data),
                              "--imgsz", "64", "--batch", "2", "--device", "cpu", "--workers", "0",
                              "--no-plots"], work, env, args.timeout)
        run_step("paired-predict", [python, str(ROOT / "predict.py"), "--weights", str(best),
                                    "--rgb", str(dataset / "rgb/val"), "--ir", str(dataset / "ir/val"),
                                    "--data", str(data), "--imgsz", "64", "--device", "cpu", "--no-save",
                                    "--project", str(runs), "--name", "predict"], work, env, args.timeout)
        run_step("torchscript-export", [python, "-u", str(ROOT / "export.py"), "--weights", str(best),
                                        "--format", "torchscript", "--imgsz", "64", "--device", "cpu"],
                 work, env, args.timeout)
        exported = best.with_suffix(".torchscript")
        run_step("torchscript-predict", [python, str(ROOT / "predict.py"), "--weights", str(exported),
                                         "--rgb", str(dataset / "rgb/val"), "--ir", str(dataset / "ir/val"),
                                         "--imgsz", "64", "--device", "cpu", "--no-save", "--project",
                                         str(runs), "--name", "torchscript_predict"], work, env, args.timeout)
        print(json.dumps({"ok": True, "best": str(best), "torchscript": str(exported)}, indent=2))
    finally:
        if args.keep:
            print(f"Kept smoke workspace: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
