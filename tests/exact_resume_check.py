"""Verify byte-exact epoch-boundary resume on a deterministic CPU fixture.

This is intentionally a standalone integration check rather than a normal unit test.
It trains the same two-epoch run continuously and with a planned interruption after
checkpointing epoch one, then compares the raw model and EMA tensors.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402


class PlannedStop(RuntimeError):
    """Controlled interruption after an epoch-boundary checkpoint is safely written."""


def _state_differences(left, right) -> list[str]:
    a, b = left.state_dict(), right.state_dict()
    keys = sorted(set(a) | set(b))
    return [key for key in keys if key not in a or key not in b or not torch.equal(a[key], b[key])]


def run_check(workspace: Path) -> dict:
    data_root = workspace / "dataset"
    subprocess.run(
        [sys.executable, str(ROOT / "tests/create_synthetic_fixture.py"), str(data_root)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    data = str(data_root / "data.yaml")
    model = str(ROOT / "tests/fixtures/tiny_sjpa_obb.yaml")
    runs = workspace / "runs"
    common = dict(
        data=data,
        epochs=2,
        batch=2,
        imgsz=64,
        device="cpu",
        workers=0,
        cache=False,
        ch=6,
        seed=123,
        deterministic=True,
        exact_resume=True,
        optimizer="SGD",
        amp=False,
        plots=False,
        preflight=True,
        data_fingerprint="metadata",
        close_mosaic=0,
        val=True,
    )

    YOLO(model).train(project=str(runs), name="continuous", exist_ok=True, **common)

    interrupted = YOLO(model)

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            raise PlannedStop("planned epoch-boundary interruption")

    interrupted.add_callback("on_model_save", stop_after_first_epoch)
    try:
        interrupted.train(project=str(runs), name="interrupted", exist_ok=True, **common)
    except PlannedStop:
        pass

    resume_path = runs / "interrupted/weights/last-resume.pt"
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint was not created: {resume_path}")
    YOLO(str(resume_path)).train(
        resume=str(resume_path), exact_resume=True, device="cpu", plots=False
    )

    continuous_ckpt = torch_load(runs / "continuous/weights/last-resume.pt", map_location="cpu")
    resumed_ckpt = torch_load(runs / "interrupted/weights/last-resume.pt", map_location="cpu")
    result = {
        "model_differences": _state_differences(continuous_ckpt["model"], resumed_ckpt["model"]),
        "ema_differences": _state_differences(continuous_ckpt["ema"], resumed_ckpt["ema"]),
        "continuous_epoch": continuous_ckpt["epoch"],
        "resumed_epoch": resumed_ckpt["epoch"],
        "continuous_best_fitness": continuous_ckpt["best_fitness"],
        "resumed_best_fitness": resumed_ckpt["best_fitness"],
        "continuous_updates": continuous_ckpt["updates"],
        "resumed_updates": resumed_ckpt["updates"],
        "exact": False,
    }
    result["exact"] = (
        not result["model_differences"]
        and not result["ema_differences"]
        and result["continuous_epoch"] == result["resumed_epoch"]
        and result["continuous_best_fitness"] == result["resumed_best_fitness"]
        and result["continuous_updates"] == result["resumed_updates"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.workspace is None:
        with tempfile.TemporaryDirectory(prefix="etfnet_exact_resume_") as tmp:
            result = run_check(Path(tmp))
    else:
        args.workspace.mkdir(parents=True, exist_ok=True)
        result = run_check(args.workspace.resolve())

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not result["exact"]:
        raise SystemExit("Exact-resume equivalence check failed")


if __name__ == "__main__":
    main()
