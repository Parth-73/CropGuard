"""CropGuard command line interface."""
import argparse
import json
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="Crop/weed segmentation: train, evaluate, and locate plants")
    sub = root.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("check-data", help="Validate polygons, class coverage, and exact duplicate split leakage")
    audit.add_argument("--data", type=Path, default=Path("configs/plants.yaml"))
    audit.add_argument("--require-test", action="store_true")
    fit = sub.add_parser("train")
    fit.add_argument("--data", type=Path, default=Path("configs/plants.yaml"))
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--base", default="yolo11s-seg.pt")
    fit.add_argument("--epochs", type=int, default=150)
    fit.add_argument("--batch", type=int, default=4)
    fit.add_argument("--workers", type=int, default=0)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--allow-partial", action="store_true", help="Train only classes present in the data; record restricted crop coverage")
    fit.add_argument("--imgsz", type=int, default=1024)
    fit.add_argument("--device", default="cpu")
    continued = sub.add_parser("resume", help="Resume an interrupted run, optionally on another device")
    continued.add_argument("--weights", type=Path, required=True)
    continued.add_argument("--device", default="cpu")
    continued.add_argument("--workers", type=int, default=0)
    val = sub.add_parser("evaluate")
    val.add_argument("--data", type=Path, default=Path("configs/plants.yaml"))
    val.add_argument("--weights", type=Path, required=True)
    val.add_argument("--output", type=Path, required=True)
    val.add_argument("--split", choices=["val", "test"], default="test")
    val.add_argument("--imgsz", type=int, default=1024)
    val.add_argument("--device", default="cpu")
    spatial = sub.add_parser("localization", help="Audit original WeedsGalore pixel masks at a deployment threshold")
    spatial.add_argument("--data", type=Path, default=Path("configs/weedsgalore.yaml"))
    spatial.add_argument("--weights", type=Path, required=True)
    spatial.add_argument("--output", type=Path, required=True, help="New output JSON file")
    spatial.add_argument("--split", choices=["val", "test"], default="test")
    spatial.add_argument("--confidence", type=float, default=0.6)
    spatial.add_argument("--imgsz", type=int, default=512)
    spatial.add_argument("--device", default="cpu")
    infer = sub.add_parser("predict")
    infer.add_argument("--source", type=Path, required=True)
    infer.add_argument("--weights", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--primary-crop", required=True)
    infer.add_argument("--confidence", type=float, default=0.25)
    infer.add_argument("--review-threshold", type=float, default=0.6)
    infer.add_argument("--imgsz", type=int, default=1024)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--max-detections", type=int, default=300)
    infer.add_argument("--no-masks", action="store_true", help="Omit lossless mask RLE from JSON")
    infer.add_argument("--no-annotated", action="store_true")
    return root


def main():
    import os
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    args = vars(parser().parse_args())
    command = args.pop("command")
    try:
        if command == "check-data":
            from .dataset import inspect_dataset
            result = inspect_dataset(args.pop("data"), **args)
        elif command == "resume":
            from .training import resume
            result = str(resume(**args))
        elif command == "train":
            from .training import train
            result = str(train(**args))
        elif command == "localization":
            from .localization import evaluate_localization
            result = evaluate_localization(**args)
        elif command == "evaluate":
            from .training import evaluate
            result = evaluate(**args)
        else:
            from .inference import predict
            args["include_rle"] = not args.pop("no_masks")
            args["annotated"] = not args.pop("no_annotated")
            result = str(predict(**args))
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        raise SystemExit(f"CropGuard: {exc}") from exc
