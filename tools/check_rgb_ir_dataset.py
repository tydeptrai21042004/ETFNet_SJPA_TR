"""Command-line wrapper for :mod:`ultralytics.data.rgb_ir_check`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.data.rgb_ir_check import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset YAML")
    parser.add_argument("--task", choices=("obb", "detect"), default="obb")
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--fingerprint", choices=("none", "metadata", "sha256"), default="metadata")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    args = parser.parse_args()
    report = validate_dataset(args.data, args.task, args.max_errors, args.fingerprint)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
