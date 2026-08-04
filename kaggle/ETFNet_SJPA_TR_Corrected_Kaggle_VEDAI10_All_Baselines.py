# ETFNet-SJPA-TR corrected-repository Kaggle benchmark.
# Kaggle settings: Internet=On, Accelerator=GPU T4/P100.
# This cell does not monkey-patch model or dataset source code.

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path

EXPECTED_VERSION = "8.0.238+etfnetsjpa.7"
REPO_URL = "https://github.com/tydeptrai21042004/ETFNet_SJPA_TR.git"
REPO_REF = "main"
SOURCE_MODE = "auto"  # auto | uploaded_zip | github
CORRECTED_ZIP_GLOB = "ETFNet_SJPA_TR_Corrected_v7.zip"

EPOCHS = 10
SEEDS = [0]
IMGSZ = 512
INITIAL_BATCH = 4
WORKERS = 2
VAL_FRACTION = 0.20
SPLIT_SEED = 0
RUN_FULL_PYTEST = False
RUN_PROPOSAL_SMOKE = True
REPROCESS_DATA = False
RETRAIN_COMPLETED = False

WORK_ROOT = Path("/kaggle/working")
REPO_DIR = WORK_ROOT / "ETFNet_SJPA_TR_Corrected"
DATA_ROOT = WORK_ROOT / "etfnet_vedai512_full"
RUN_ROOT = WORK_ROOT / "etfnet_corrected_vedai512_10epoch_all_baselines"
RESULT_ZIP = WORK_ROOT / "ETFNet_SJPA_TR_Corrected_VEDAI10_results.zip"


def run(command, *, cwd=None, env=None, check=True, log_path=None):
    command = [str(value) for value in command]
    print("\n$", " ".join(command), flush=True)
    if log_path is None:
        return subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, text=True, check=check)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode and check:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        raise RuntimeError(f"Command failed with status {completed.returncode}: {' '.join(command)}\n{tail}")
    return completed


def find_repo_root(parent: Path) -> Path:
    candidates = [
        path.parent
        for path in parent.rglob("pyproject.toml")
        if (path.parent / "ultralytics/__init__.py").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No ETFNet repository root found under {parent}")
    return min(candidates, key=lambda path: len(path.parts))


# 1) Load corrected repository from an uploaded ZIP, or clone a corrected GitHub branch.
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

zip_candidates = sorted(Path("/kaggle/input").rglob(CORRECTED_ZIP_GLOB))
use_zip = SOURCE_MODE == "uploaded_zip" or (SOURCE_MODE == "auto" and bool(zip_candidates))
if use_zip:
    if not zip_candidates:
        raise FileNotFoundError(
            f"Upload {CORRECTED_ZIP_GLOB} as a Kaggle Dataset input, then rerun the cell."
        )
    extraction = WORK_ROOT / "_etfnet_corrected_extract"
    shutil.rmtree(extraction, ignore_errors=True)
    extraction.mkdir(parents=True)
    shutil.unpack_archive(str(zip_candidates[0]), str(extraction))
    extracted_root = find_repo_root(extraction)
    shutil.move(str(extracted_root), str(REPO_DIR))
    shutil.rmtree(extraction, ignore_errors=True)
    print(f"Loaded corrected repository ZIP: {zip_candidates[0]}")
else:
    run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, REPO_DIR])

os.chdir(REPO_DIR)
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "--disable-pip-version-check",
    "--no-deps",
    "-e",
    ".",
])
run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "pytest"])

version_check = subprocess.run(
    [sys.executable, "-c", "import ultralytics; print(ultralytics.__version__)"],
    cwd=REPO_DIR,
    text=True,
    capture_output=True,
    check=True,
).stdout.strip()
if version_check != EXPECTED_VERSION:
    raise RuntimeError(
        f"Repository version is {version_check!r}, expected {EXPECTED_VERSION!r}. "
        f"Upload {CORRECTED_ZIP_GLOB} from the provided artifact as a Kaggle Dataset, "
        "or push the corrected repository to GitHub before using SOURCE_MODE='github'."
    )
print("Corrected repository version verified:", version_check)

# Normal runtime configuration: disable optional experiment-loggers for this benchmark.
runner = WORK_ROOT / "etfnet_clean_runner.py"
runner.write_text(
    textwrap.dedent(
        """
        from __future__ import annotations
        import runpy
        import sys
        from pathlib import Path
        from ultralytics.utils import SETTINGS

        for key in ("wandb", "raytune", "tensorboard", "clearml", "comet", "dvc", "mlflow", "neptune"):
            if key in SETTINGS:
                SETTINGS[key] = False

        target = Path(sys.argv[1]).resolve()
        sys.argv = [str(target), *sys.argv[2:]]
        runpy.run_path(str(target), run_name="__main__")
        """
    ).strip()
    + "\n",
    encoding="utf-8",
)

base_env = os.environ.copy()
base_env.update(
    {
        "WANDB_MODE": "disabled",
        "WANDB_SILENT": "true",
        "COMET_MODE": "DISABLED",
        "CLEARML_OFFLINE_MODE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "PYTHONHASHSEED": "0",
    }
)

# 2) Validate corrected source before the long benchmark.
RUN_ROOT.mkdir(parents=True, exist_ok=True)
run(
    [
        sys.executable,
        "tools/audit_repository_layout.py",
        "--json",
        RUN_ROOT / "repository_layout_audit.json",
    ],
    cwd=REPO_DIR,
    env=base_env,
)
run(
    [sys.executable, "-m", "pytest", "-q", "tests/test_kaggle_regressions.py"],
    cwd=REPO_DIR,
    env=base_env,
)
if RUN_FULL_PYTEST:
    run([sys.executable, "-m", "pytest", "-q"], cwd=REPO_DIR, env=base_env)
run(
    [
        sys.executable,
        "tools/validate_model_configs.py",
        "--imgsz",
        "64",
        "--output",
        RUN_ROOT / "all_model_forward_report.json",
    ],
    cwd=REPO_DIR,
    env=base_env,
)

# 3) Download and validate the complete smallest supported dataset: VEDAI-512.
import yaml

EXPECTED_CLASSES = (
    "plane",
    "boat",
    "camping-car",
    "car",
    "pickup",
    "tractor",
    "truck",
    "van",
    "other",
)
PROCESSED_DIR = DATA_ROOT / "vedai" / "processed" / "512"
DATA_YAML = PROCESSED_DIR / "data.yaml"
lock_file = DATA_ROOT / "vedai" / ".prepare.lock"
if lock_file.exists():
    lock_file.unlink()


def data_classes(path: Path):
    if not path.is_file():
        return ()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {})
    if isinstance(names, list):
        return tuple(map(str, names))
    return tuple(str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0])))


if REPROCESS_DATA or (DATA_YAML.is_file() and data_classes(DATA_YAML) != EXPECTED_CLASSES):
    shutil.rmtree(PROCESSED_DIR, ignore_errors=True)

if not DATA_YAML.is_file():
    run(
        [
            sys.executable,
            runner,
            REPO_DIR / "etfnet_cli.py",
            "download-data",
            "--dataset",
            "vedai",
            "--variant",
            "512",
            "--output",
            DATA_ROOT,
            "--task",
            "obb",
            "--accept-license",
            "--val-fraction",
            str(VAL_FRACTION),
            "--split-seed",
            str(SPLIT_SEED),
            "--force-preprocess",
            "--validate",
        ],
        cwd=REPO_DIR,
        env=base_env,
    )
else:
    print("Reusing prepared VEDAI-512:", DATA_YAML)

if data_classes(DATA_YAML) != EXPECTED_CLASSES:
    raise RuntimeError(f"Wrong VEDAI class schema: {data_classes(DATA_YAML)!r}")
run(
    [
        sys.executable,
        runner,
        REPO_DIR / "etfnet_cli.py",
        "check-data",
        "--data",
        DATA_YAML,
        "--task",
        "obb",
        "--fingerprint",
        "metadata",
        "--output",
        RUN_ROOT / "dataset_preflight.json",
    ],
    cwd=REPO_DIR,
    env=base_env,
)

# 4) Test the corrected CUDA AMP path. Fall back globally to FP32 only if the probe fails.
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running this benchmark.")
AMP_ENABLED = True
try:
    from ultralytics.nn.modules.block import SJPA

    module = SJPA(128, 32, 8).cuda().train()
    rgb = torch.randn(2, 128, 16, 16, device="cuda", requires_grad=True)
    ir = torch.randn(2, 128, 16, 16, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        probe_output = module([rgb, ir])
        probe_loss = probe_output.float().square().mean()
    probe_loss.backward()
    assert torch.isfinite(probe_output.float()).all()
    assert rgb.grad is not None and ir.grad is not None
    print("Corrected SJPA CUDA AMP forward/backward probe passed.")
except Exception:
    AMP_ENABLED = False
    print("CUDA AMP probe failed; all models will use FP32 for a fair comparison.")
    traceback.print_exc()
finally:
    for variable in ("module", "rgb", "ir", "probe_output", "probe_loss"):
        if variable in locals():
            del locals()[variable]
    torch.cuda.empty_cache()

# 5) Model matrix and one ablation generated outside the source repository.
proposal_yaml = REPO_DIR / "ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml"
no_reliability_yaml = RUN_ROOT / "etfnet_P2_CAFEM_SJPA_noReliability.yaml"
proposal_cfg = yaml.safe_load(proposal_yaml.read_text(encoding="utf-8"))
for layer in proposal_cfg.get("backbone", []):
    if len(layer) >= 4 and layer[2] == "SJPA":
        layer[3][4] = False
        break
else:
    raise RuntimeError("SJPA layer not found while creating the reliability ablation")
no_reliability_yaml.write_text(yaml.safe_dump(proposal_cfg, sort_keys=False), encoding="utf-8")

MODEL_MATRIX = {
    "rgb_p3p5": REPO_DIR / "ultralytics/cfg/models/etfnet/noCAFEM_noTGF.yaml",
    "rgb_yolo11_p2p5": REPO_DIR / "ultralytics/cfg/models/etfnet/etfnet_yolo11.yaml",
    "dual_concat": REPO_DIR / "ultralytics/cfg/models/etfnet/etfnet_dualstream_noCAFEM_noTGF.yaml",
    "dual_add": REPO_DIR / "ultralytics/cfg/models/etfnet/noCAFEM_noTGF_add.yaml",
    "cafem_only": REPO_DIR / "ultralytics/cfg/models/etfnet/etfnet_CAFEM_noTGF.yaml",
    "tgf_only": REPO_DIR / "ultralytics/cfg/models/etfnet/etfnet_noCAFEM_TGF.yaml",
    "corrected_etfnet_tgf": REPO_DIR / "ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_TGF.yaml",
    "goci_no_spatial": REPO_DIR / "ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_GOCI.yaml",
    "sjpa_no_reliability": no_reliability_yaml,
    "sjpa_tr_proposal": proposal_yaml,
}
for label, path in MODEL_MATRIX.items():
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def read_last_result(run_dir: Path):
    path = run_dir / "results.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def is_complete(run_dir: Path):
    row = read_last_result(run_dir)
    if not row or not (run_dir / "weights/best.pt").is_file():
        return False
    for key, value in row.items():
        if key.strip().lower() == "epoch":
            try:
                return int(float(value)) >= EPOCHS - 1
            except (TypeError, ValueError):
                return False
    return False


def metric(row, required, excluded=()):
    if not row:
        return float("nan")
    for key, value in row.items():
        normalized = key.lower().replace(" ", "")
        if all(token in normalized for token in required) and not any(token in normalized for token in excluded):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float("nan")


def train_command(model_path, name, epochs, batch, seed):
    return [
        sys.executable,
        runner,
        REPO_DIR / "etfnet_cli.py",
        "train",
        "--data",
        DATA_YAML,
        "--model",
        model_path,
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--imgsz",
        str(IMGSZ),
        "--device",
        "0",
        "--workers",
        str(WORKERS),
        "--project",
        RUN_ROOT,
        "--name",
        name,
        "--exist-ok",
        "--seed",
        str(seed),
        "--optimizer",
        "SGD",
        "--lr0",
        "0.01",
        "--lrf",
        "0.01",
        "--momentum",
        "0.937",
        "--weight-decay",
        "0.0005",
        "--no-plots",
        "--close-mosaic",
        "0",
        "--data-fingerprint",
        "metadata",
        "--amp" if AMP_ENABLED else "--no-amp",
    ]


def train_with_oom_retry(model_path, name, epochs, seed):
    run_dir = RUN_ROOT / name
    batches = []
    for candidate in (INITIAL_BATCH, max(1, INITIAL_BATCH // 2), 1):
        if candidate not in batches:
            batches.append(candidate)
    for batch in batches:
        shutil.rmtree(run_dir, ignore_errors=True)
        log_path = RUN_ROOT / "logs" / f"{name}.batch{batch}.log"
        result = run(
            train_command(model_path, name, epochs, batch, seed),
            cwd=REPO_DIR,
            env=base_env,
            check=False,
            log_path=log_path,
        )
        if result.returncode == 0 and (run_dir / "weights/best.pt").is_file():
            print(f"PASS {name}: batch={batch}, amp={AMP_ENABLED}")
            return batch, ""
        text = log_path.read_text(encoding="utf-8", errors="replace")
        oom = "out of memory" in text.lower() or "cuda error: out of memory" in text.lower()
        print(text[-5000:])
        if not oom:
            return batch, f"training return code {result.returncode}; see {log_path}"
        print(f"CUDA OOM for {name} at batch={batch}; retrying smaller batch.")
        torch.cuda.empty_cache()
    return batches[-1], "CUDA OOM at every attempted batch size"


if RUN_PROPOSAL_SMOKE:
    smoke_dir = RUN_ROOT / "_proposal_smoke"
    if not (smoke_dir / "weights/best.pt").is_file():
        used_batch, error = train_with_oom_retry(proposal_yaml, "_proposal_smoke", 1, 0)
        if error:
            raise RuntimeError(f"Corrected proposal smoke test failed: {error}")
    print("Corrected full SJPA-TR real-data smoke training passed.")

# 6) Ten-epoch comparison.
records = []
for seed in SEEDS:
    for label, model_path in MODEL_MATRIX.items():
        name = f"{label}_seed{seed}"
        run_dir = RUN_ROOT / name
        started = time.time()
        status = "ok"
        error = ""
        used_batch = INITIAL_BATCH
        if RETRAIN_COMPLETED or not is_complete(run_dir):
            used_batch, error = train_with_oom_retry(model_path, name, EPOCHS, seed)
            if error:
                status = "failed"
        else:
            print("Skipping completed run:", name)
        row = read_last_result(run_dir)
        model_cfg = yaml.safe_load(Path(model_path).read_text(encoding="utf-8"))
        records.append(
            {
                "model": label,
                "seed": seed,
                "status": status,
                "error": error,
                "channels": int(model_cfg.get("ch", 6)),
                "batch": used_batch,
                "amp": AMP_ENABLED,
                "seconds_this_cell": round(time.time() - started, 2),
                "mAP50": metric(row, ("map50",), ("95",)),
                "mAP50_95": metric(row, ("map50-95",)),
                "precision": metric(row, ("precision",)),
                "recall": metric(row, ("recall",)),
            }
        )
        import pandas as pd

        pd.DataFrame(records).to_csv(RUN_ROOT / "per_run_results.csv", index=False)

# 7) Aggregate and package.
import pandas as pd
from IPython.display import display

df = pd.DataFrame(records)
ok = df[df.status == "ok"].copy()
if ok.empty:
    raise RuntimeError("All model runs failed; inspect RUN_ROOT/logs.")
summary = (
    ok.groupby("model", as_index=False)
    .agg(
        runs=("seed", "count"),
        channels=("channels", "first"),
        batch=("batch", "min"),
        amp=("amp", "first"),
        mAP50_mean=("mAP50", "mean"),
        mAP50_std=("mAP50", "std"),
        mAP50_95_mean=("mAP50_95", "mean"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
    )
    .sort_values(["mAP50_mean", "mAP50_95_mean"], ascending=False, na_position="last")
)
summary.to_csv(RUN_ROOT / "summary.csv", index=False)
(RUN_ROOT / "benchmark_settings.json").write_text(
    json.dumps(
        {
            "repository_version": EXPECTED_VERSION,
            "source_mode": "uploaded_zip" if use_zip else "github",
            "dataset": "complete VEDAI-512",
            "dataset_yaml": str(DATA_YAML),
            "classes": list(EXPECTED_CLASSES),
            "epochs": EPOCHS,
            "seeds": SEEDS,
            "imgsz": IMGSZ,
            "initial_batch": INITIAL_BATCH,
            "amp": AMP_ENABLED,
            "models": {key: str(value) for key, value in MODEL_MATRIX.items()},
            "note": "Ten epochs and one seed are a development benchmark, not a paper-grade final result.",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print("\n=== PER-RUN RESULTS ===")
display(df.sort_values(["status", "mAP50"], ascending=[True, False]))
print("\n=== RANKING ===")
display(summary)
if (df.status != "ok").any():
    print("\nFailed configurations:")
    display(df[df.status != "ok"])

if RESULT_ZIP.exists():
    RESULT_ZIP.unlink()
archive = shutil.make_archive(str(RESULT_ZIP.with_suffix("")), "zip", RUN_ROOT)
print("\nResults directory:", RUN_ROOT)
print("Kaggle output ZIP:", archive)
