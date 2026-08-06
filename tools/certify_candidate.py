#!/usr/bin/env python3
"""Finite-family paired non-inferiority certificate for bounded image utilities.

The input CSV must contain one row per validation example and the columns
``candidate``, ``baseline``, and optionally ``candidate_id``. Both utility
columns must lie in [0, 1]. For K candidate configurations, the script reports

    LCB_k = mean(candidate - baseline) - sqrt(2 log(K / delta) / n_k).

By Hoeffding's inequality and a union bound, under i.i.d./exchangeable
validation examples, all reported lower bounds hold simultaneously with
probability at least 1-delta. This certificate is deliberately separate from
mAP: use a bounded per-image utility fixed before viewing the test set.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="CSV with candidate/baseline utilities in [0,1]")
    parser.add_argument("--delta", type=float, default=0.05, help="family-wise error probability")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output")
    return parser.parse_args()


def load_pairs(path: Path) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"candidate", "baseline"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            candidate = float(row["candidate"])
            baseline = float(row["baseline"])
            if not (0.0 <= candidate <= 1.0 and 0.0 <= baseline <= 1.0):
                raise ValueError(f"line {line}: utilities must lie in [0,1]")
            candidate_id = (row.get("candidate_id") or "candidate").strip() or "candidate"
            groups[candidate_id].append(candidate - baseline)
    if not groups:
        raise ValueError("CSV contains no data rows")
    return dict(groups)


def certify(groups: dict[str, list[float]], delta: float) -> dict:
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between 0 and 1")
    family_size = len(groups)
    results = {}
    for candidate_id, differences in groups.items():
        n = len(differences)
        mean_difference = sum(differences) / n
        radius = math.sqrt(2.0 * math.log(family_size / delta) / n)
        lower_bound = mean_difference - radius
        results[candidate_id] = {
            "n": n,
            "mean_paired_improvement": mean_difference,
            "simultaneous_hoeffding_radius": radius,
            "lower_confidence_bound": lower_bound,
            "certified_strict_improvement": lower_bound > 0.0,
            "certified_non_inferiority": lower_bound >= 0.0,
        }
    return {
        "assumptions": [
            "examples are i.i.d. or exchangeable with the deployment population",
            "candidate and baseline utilities are paired and bounded in [0,1]",
            "the candidate family and utility were fixed before test-set inspection",
        ],
        "delta": delta,
        "family_size": family_size,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    report = certify(load_pairs(args.csv), args.delta)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
