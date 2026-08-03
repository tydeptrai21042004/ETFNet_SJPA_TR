"""Fail when a committed subfolder contains more than the configured recursive file limit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repository_layout import MAX_FILES_PER_SUBFOLDER, ROOT

IGNORED_NAMES = {".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", "build", "dist", "site", ".docs_workspace"}


def recursive_file_count(folder: Path) -> int:
    return sum(1 for path in folder.rglob("*") if path.is_file() and not any(part in IGNORED_NAMES for part in path.parts))


def audit(root: Path = ROOT, limit: int = MAX_FILES_PER_SUBFOLDER):
    rows = []
    violations = []
    for folder in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: str(p)):
        if any(part in IGNORED_NAMES for part in folder.relative_to(root).parts):
            continue
        count = recursive_file_count(folder)
        row = {"folder": str(folder.relative_to(root)), "recursive_files": count}
        rows.append(row)
        if count > limit:
            violations.append(row)
    return rows, violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--limit", type=int, default=MAX_FILES_PER_SUBFOLDER)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows, violations = audit(args.root.resolve(), args.limit)
    payload = {"limit": args.limit, "folders_checked": len(rows), "violations": violations, "folders": rows}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Checked {len(rows)} subfolders; limit={args.limit}; violations={len(violations)}")
    for item in violations:
        print(f"ERROR {item['folder']}: {item['recursive_files']} files")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
