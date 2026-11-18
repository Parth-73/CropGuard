"""Transfer learning and held-out evaluation for the fixed plant taxonomy."""
from pathlib import Path
import json
import hashlib
from .dataset import inspect_dataset, resolved_config
from .taxonomy import REQUIRED_NAMES, validate_names


def train(data: Path, output: Path, base: str = "yolo11s-seg.pt", epochs: int = 150,
          imgsz: int = 1024, batch: int = 4, device: str = "cpu", workers: int = 0,
          seed: int = 42, allow_partial: bool = False) -> Path:
    report = inspect_dataset(data)
    if epochs < 1 or batch < 1 or imgsz < 32 or imgsz % 32 or workers < 0:
        raise ValueError("Require positive epochs/batch, imgsz multiple of 32, and workers >= 0")
    # Missing classes are fatal for training: a six-class name list alone is not training coverage.
    if report["warnings"] and not allow_partial:
        raise ValueError("Provide all six classes in train and val before training: " + "; ".join(report["warnings"]))
    counts = report["splits"]["train"]["instances_by_class"]
    val_counts = report["splits"]["val"]["instances_by_class"]
    available = sorted(i for i in counts if val_counts.get(i, 0) > 0)
    if 0 not in available or not any(i in available for i in [1, 2, 3]):
        raise ValueError("Training and validation both need weeds and at least one primary crop")
    from ultralytics import YOLO
    model = YOLO(base)
    if model.task != "segment":
        raise ValueError("Base model must support instance segmentation (e.g. yolo11s-seg.pt)")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    config = resolved_config(data, output / "dataset.resolved.yaml")
    model.train(data=str(config), epochs=epochs, imgsz=imgsz, batch=batch,
                device=device, workers=workers, seed=seed, deterministic=True,
                project=str(output), name="fit", exist_ok=False, patience=30,
                optimizer="AdamW", lr0=0.001, cos_lr=True, close_mosaic=min(15, epochs // 3),
                degrees=15, translate=0.1, scale=0.5, fliplr=0.5, flipud=0.1,
                hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, mosaic=0.5,
                mixup=0.0, copy_paste=0.1, save=True, plots=True)
    return _write_training_outputs(model, output, report, base, seed, available)


def _write_training_outputs(model, output, report, base, seed, available):
    checkpoint = Path(model.trainer.best)
    if not checkpoint.is_file():
        raise RuntimeError("Training completed without a best checkpoint")
    (output / "checkpoint.json").write_text(json.dumps({"best": str(checkpoint.resolve()),
        "base": base, "seed": seed, "status": "trained; requires independent field evaluation"}, indent=2), encoding="utf-8")
    coverage = {"available_class_ids": available,
                "supported_primary_crops": [REQUIRED_NAMES[i] for i in available if i in [1, 2, 3]],
                "partial_taxonomy": len(available) < 6,
                "weights_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "dataset": report, "evaluation_status": "requires held-out evaluation"}
    checkpoint.with_suffix(".cropguard.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return checkpoint


def resume(weights: Path, device: str = "cpu", workers: int = 0) -> Path:
    weights = Path(weights).resolve()
    if not weights.is_file() or workers < 0:
        raise ValueError("Resume requires an existing last.pt checkpoint and workers >= 0")
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if model.task != "segment":
        raise ValueError("Resume requires a segmentation checkpoint")
    validate_names(model.names)
    args = model.ckpt["train_args"]
    report = inspect_dataset(Path(args["data"]))
    counts = report["splits"]["train"]["instances_by_class"]
    available = sorted(i for i in counts if report["splits"]["val"]["instances_by_class"].get(i, 0))
    model.train(resume=True, device=device, workers=workers)
    output = Path(model.trainer.save_dir).parent
    return _write_training_outputs(model, output, report, str(weights), int(args.get("seed", 42)), available)


def evaluate(data: Path, weights: Path, output: Path, split: str = "test",
             imgsz: int = 1024, device: str = "cpu") -> dict:
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    inspect_dataset(data, require_test=split == "test")
    if not Path(weights).is_file():
        raise ValueError("Weights must be an existing trained checkpoint")
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if model.task != "segment":
        raise ValueError("A segmentation checkpoint is required")
    validate_names(model.names)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = resolved_config(data, output / "dataset.resolved.yaml")
    metrics = model.val(data=str(config), split=split, imgsz=imgsz, device=device,
                        workers=0, project=str(output), name="metrics", plots=True)
    report = {"split": split, "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
              "per_class": [], "speed_ms": {k: float(v) for k, v in metrics.speed.items()}}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        bp, br, bap50, bap = metrics.box.class_result(index)
        mp, mr, map50, map_ = metrics.seg.class_result(index)
        report["per_class"].append({"label": model.names[int(class_id)],
            "box_precision": float(bp), "box_recall": float(br), "box_map50": float(bap50), "box_map50_95": float(bap),
            "mask_precision": float(mp), "mask_recall": float(mr), "mask_map50": float(map50), "mask_map50_95": float(map_)})
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
