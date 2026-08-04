"""Unified, reproducible ETFNet-SJPA-TR train/val/predict/export interface."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Set deterministic CUDA defaults before importing torch through Ultralytics.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from ultralytics import YOLO
from ultralytics.data.rgb_ir_check import validate_dataset
from ultralytics.data.public_rgb_ir import (list_public_datasets, prepare_public_dataset,
                                             resolve_data_reference)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = str(ROOT / "ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml")


def _device(value: str):
    return value if value else None




def _model_channels(model: YOLO, fallback: int = 6) -> int:
    """Infer the expected input channels from a YAML model or loaded checkpoint."""
    candidates = []
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(getattr(inner, "yaml", None))
        candidates.append(getattr(inner, "args", None))
    candidates.append(getattr(model, "overrides", None))
    for candidate in candidates:
        if isinstance(candidate, dict):
            value = candidate.get("ch")
        else:
            value = getattr(candidate, "ch", None) if candidate is not None else None
        if value is not None:
            try:
                channels = int(value)
            except (TypeError, ValueError):
                continue
            if channels > 0:
                return channels
    return int(fallback)

def _add_common_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=4)


def _add_public_data_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets-dir", default=os.getenv("ETFNET_DATASETS_DIR", "datasets"),
                        help="Root for auto-downloaded public datasets")
    parser.add_argument("--dataset-variant", default=None,
                        help="Dataset-specific variant (e.g. 512 for VEDAI, aligned for FLIR)")
    parser.add_argument("--dataset-limit", type=int, default=None,
                        help="Deterministic maximum pairs per split for quick experiments")
    parser.add_argument("--dataset-val-fraction", type=float, default=0.2,
                        help="Deterministic validation fraction when the source has no official split")
    parser.add_argument("--dataset-split-seed", type=int, default=0)
    parser.add_argument("--accept-dataset-license", action="store_true",
                        help="Accept the public dataset license when --data uses public:<name>")


def _resolve_data(value: str, args: argparse.Namespace, task: str = "obb") -> str:
    return resolve_data_reference(
        value, datasets_dir=getattr(args, "datasets_dir", "datasets"),
        variant=getattr(args, "dataset_variant", None), task=task,
        accept_license=getattr(args, "accept_dataset_license", False),
        limit=getattr(args, "dataset_limit", None),
        val_fraction=getattr(args, "dataset_val_fraction", 0.2),
        split_seed=getattr(args, "dataset_split_seed", 0),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the SJPA-TR OBB model")
    train.add_argument("--data", required=True, help="YAML path or public:<dataset> (m3fd, vedai, flir-aligned, rgbtdroneperson, cvc-14)")
    _add_public_data_options(train)
    train.add_argument("--model", default=DEFAULT_MODEL)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch", type=int, default=4)
    _add_common_runtime(train)
    train.add_argument("--cache", choices=("none", "ram", "disk"), default="none")
    train.add_argument("--project", default="runs/etfnet")
    train.add_argument("--name", default="sjpa_train")
    train.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--exact-resume", action=argparse.BooleanOptionalAction, default=False,
                       help="Use workers=0 and checkpoint RNG/scaler/scheduler state for exact epoch-boundary resume")
    train.add_argument("--optimizer", choices=("SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"),
                       default="SGD")
    train.add_argument("--lr0", type=float, default=0.01)
    train.add_argument("--lrf", type=float, default=0.01)
    train.add_argument("--momentum", type=float, default=0.937)
    train.add_argument("--weight-decay", type=float, default=0.0005)
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--data-fingerprint", choices=("none", "metadata", "sha256"), default="metadata")
    train.add_argument("--save-period", type=int, default=-1,
                       help="Also save resumable epochN.pt every N epochs; -1 disables periodic copies")
    train.add_argument("--close-mosaic", type=int, default=10)

    val = sub.add_parser("val", help="Validate a checkpoint")
    val.add_argument("--weights", required=True)
    val.add_argument("--data", required=True, help="YAML path or public:<dataset> (m3fd, vedai, flir-aligned, rgbtdroneperson, cvc-14)")
    _add_public_data_options(val)
    val.add_argument("--split", choices=("train", "val", "test"), default="val")
    val.add_argument("--batch", type=int, default=4)
    _add_common_runtime(val)
    val.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)

    pred = sub.add_parser("predict", help="Run synchronized paired RGB-IR inference")
    pred.add_argument("--weights", required=True)
    pred.add_argument("--rgb", required=True, help="RGB image/folder/list/video/camera/stream")
    pred.add_argument("--ir", required=True, help="Matching IR image/folder/list/video/camera/stream")
    pred.add_argument("--data", default="", help="Optional dataset YAML for classes and pair mapping")
    pred.add_argument("--imgsz", type=int, default=640)
    pred.add_argument("--device", default="")
    pred.add_argument("--conf", type=float, default=0.25)
    pred.add_argument("--iou", type=float, default=0.7)
    pred.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    pred.add_argument("--project", default="runs/etfnet")
    pred.add_argument("--name", default="predict")
    pred.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=False)
    pred.add_argument("--pair-resize", action="store_true")
    pred.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                      help="Consume results as a generator to prevent video/folder RAM accumulation")

    export = sub.add_parser("export", help="Export a trained six-channel model")
    export.add_argument("--weights", required=True)
    export.add_argument("--format", default="torchscript", choices=("torchscript", "onnx", "openvino", "engine"))
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--device", default="")
    export.add_argument("--dynamic", action="store_true")
    export.add_argument("--half", action="store_true")

    resume = sub.add_parser("resume", help="Resume from unstripped last-resume.pt")
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--data", default=None, help="Optional relocated dataset YAML")
    resume.add_argument("--epochs", type=int, default=None, help="Total target epochs")
    resume.add_argument("--batch", type=int, default=None)
    resume.add_argument("--imgsz", type=int, default=None)
    resume.add_argument("--device", default="")
    resume.add_argument("--workers", type=int, default=None)
    resume.add_argument("--cache", choices=("none", "ram", "disk"), default=None)
    resume.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    resume.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    resume.add_argument("--exact-resume", action=argparse.BooleanOptionalAction, default=False,
                        help="Require the original target epoch count and an exact-mode checkpoint")

    check = sub.add_parser("check-data", help="Validate RGB/IR pairs, labels, and dataset fingerprint")
    check.add_argument("--data", required=True, help="YAML path or public:<dataset> (m3fd, vedai, flir-aligned, rgbtdroneperson, cvc-14)")
    _add_public_data_options(check)
    check.add_argument("--task", choices=("obb", "detect"), default="obb")
    check.add_argument("--max-errors", type=int, default=100)
    check.add_argument("--fingerprint", choices=("none", "metadata", "sha256"), default="metadata")
    check.add_argument("--output", default="")

    datasets = sub.add_parser("list-data", help="List supported automatically downloadable public datasets")
    datasets.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    download = sub.add_parser("download-data", help="Download and preprocess a supported public RGB-IR dataset")
    download.add_argument("--dataset", default="nii-cu-mapd", choices=tuple(sorted(row["key"] for row in list_public_datasets())))
    download.add_argument("--output", default=os.getenv("ETFNET_DATASETS_DIR", "datasets"))
    download.add_argument("--variant", default=None,
                          help="Dataset-specific variant; defaults to the registry value")
    download.add_argument("--task", default="obb", choices=("obb", "detect"))
    download.add_argument("--accept-license", action="store_true",
                          help="Confirm review and acceptance of the dataset license")
    download.add_argument("--archive", default="",
                          help="Optional official archive already on disk; skips network download")
    download.add_argument("--force-download", action="store_true")
    download.add_argument("--force-extract", action="store_true")
    download.add_argument("--force-preprocess", action="store_true")
    download.add_argument("--keep-archive", action=argparse.BooleanOptionalAction, default=True)
    download.add_argument("--link-mode", default="auto", choices=("auto", "hardlink", "symlink", "copy"))
    download.add_argument("--visibility", default="all", choices=("all", "both", "thermal", "rgb"))
    download.add_argument("--exclude-bad", action="store_true")
    download.add_argument("--limit", type=int, default=None, help="Optional deterministic pairs per split for a quick run")
    download.add_argument("--val-fraction", type=float, default=0.2,
                          help="Validation fraction when the source has no official split")
    download.add_argument("--split-seed", type=int, default=0,
                          help="Seed for deterministic split generation")
    download.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _print_or_save_report(report: dict, output: str = "") -> None:
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list-data":
        rows = list_public_datasets()
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            for row in rows:
                doi = row["dataset_doi"] or row["citation_doi"]
                print(f'{row["key"]}: {row["title"]} | DOI {doi} | {row["license_name"]} | ~{row["expected_download_gb"]} GB')
        return

    if args.command == "download-data":
        data_yaml = prepare_public_dataset(
            args.dataset, args.output, variant=args.variant, task=args.task,
            accept_license=args.accept_license, archive=args.archive or None,
            force_download=args.force_download, force_extract=args.force_extract,
            force_preprocess=args.force_preprocess, keep_archive=args.keep_archive,
            link_mode=args.link_mode, visibility=args.visibility, exclude_bad=args.exclude_bad, limit=args.limit,
            val_fraction=args.val_fraction, split_seed=args.split_seed,
        )
        print(data_yaml)
        if args.validate:
            report = validate_dataset(str(data_yaml), args.task, fingerprint="sha256")
            _print_or_save_report(report)
            if not report["ok"]:
                raise SystemExit("Prepared dataset failed validation")
        return

    if args.command == "check-data":
        args.data = _resolve_data(args.data, args, args.task)
        report = validate_dataset(args.data, args.task, args.max_errors, args.fingerprint)
        _print_or_save_report(report, args.output)
        raise SystemExit(0 if report["ok"] else 1)

    if args.command == "train":
        args.data = _resolve_data(args.data, args, "obb")
        if args.preflight:
            report = validate_dataset(args.data, "obb", fingerprint=args.data_fingerprint)
            if not report["ok"]:
                _print_or_save_report(report)
                raise SystemExit("Dataset preflight failed; training was not started.")
        cache = False if args.cache == "none" else args.cache
        workers = 0 if args.exact_resume else args.workers
        model = YOLO(args.model)
        channels = _model_channels(model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=_device(args.device),
            workers=workers,
            cache=cache,
            ch=channels,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            seed=args.seed,
            deterministic=args.deterministic,
            exact_resume=args.exact_resume,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            amp=args.amp,
            plots=args.plots,
            preflight=args.preflight,
            data_fingerprint=args.data_fingerprint,
            save_period=args.save_period,
            close_mosaic=args.close_mosaic,
        )
        return

    if args.command == "val":
        args.data = _resolve_data(args.data, args, "obb")
        model = YOLO(args.weights, task="obb")
        channels = _model_channels(model)
        model.val(data=args.data, split=args.split, batch=args.batch, imgsz=args.imgsz,
                  device=_device(args.device), workers=args.workers, plots=args.plots, ch=channels)
        return

    if args.command == "predict":
        model = YOLO(args.weights, task="obb")
        channels = _model_channels(model)
        results = model.predict(
            source=args.rgb,
            ir_source=args.ir,
            data=args.data or None,
            imgsz=args.imgsz,
            device=_device(args.device),
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            pair_resize=args.pair_resize,
            stream=args.stream,
            ch=channels,
        )
        if args.stream:
            count = sum(1 for _ in results)
            print(f"Processed {count} paired frame(s).")
        return

    if args.command == "export":
        model = YOLO(args.weights, task="obb")
        channels = _model_channels(model)
        output = model.export(format=args.format, imgsz=args.imgsz, device=_device(args.device),
                              dynamic=args.dynamic, half=args.half, ch=channels)
        print(output)
        return

    model = YOLO(args.checkpoint)
    overrides = {
        "resume": args.checkpoint,
        "device": _device(args.device),
        "plots": args.plots,
        "ch": _model_channels(model),
        "exact_resume": args.exact_resume,
        "deterministic": True,
    }
    for key in ("data", "epochs", "batch", "imgsz", "workers", "amp"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.cache is not None:
        overrides["cache"] = False if args.cache == "none" else args.cache
    if args.exact_resume:
        overrides["workers"] = 0
    model.train(**overrides)


if __name__ == "__main__":
    main()
