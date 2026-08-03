"""Build and execute every shipped ETFNet model YAML."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ultralytics.nn.tasks import OBBModel, yaml_model_load

DEFAULT_DIR = ROOT / "ultralytics/cfg/models/etfnet"


def validate(config_dir: Path, image_size: int = 64) -> dict:
    results = []
    for path in sorted(config_dir.glob("*.yaml")):
        item = {"config": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), "ok": False}
        try:
            model = OBBModel(yaml_model_load(path), verbose=False).eval()
            channels = int(model.yaml.get("ch", 3))
            with torch.inference_mode():
                model(torch.zeros(1, channels, image_size, image_size))
            item.update(ok=True, parameters=sum(p.numel() for p in model.parameters()), channels=channels)
        except Exception as error:  # report every bad research configuration in one pass
            item["error"] = f"{type(error).__name__}: {error}"
        results.append(item)
    return {"ok": all(item["ok"] for item in results), "models": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = validate(args.config_dir.resolve(), args.imgsz)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
