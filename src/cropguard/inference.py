"""Bounded-memory inference; no generic model silently presented as crop trained."""

from pathlib import Path
import json
import math
import hashlib

import numpy as np

from .geometry import encode_rle, mask_geometry
from .taxonomy import normalize_crop, role_for, validate_names

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def result_to_record(result, primary_crop: str, review_threshold: float, include_rle: bool = True) -> dict:
    """Convert an Ultralytics result without losing original-image mask resolution."""
    names = validate_names(result.names)
    height, width = map(int, result.orig_shape)
    record = {
        "schema_version": "1.0", "width": width, "height": height,
        "coordinate_system": "original image pixels; origin top-left; x right, y down; xyxy upper bounds exclusive",
        "primary_crop": primary_crop, "detections": [],
    }
    if result.boxes is None or len(result.boxes) == 0:
        return record
    if result.masks is None:
        raise ValueError("Segmentation masks are required. Use a trained segmentation checkpoint.")
    masks = result.masks.data.cpu().numpy()
    if masks.shape != (len(result.boxes), height, width):
        raise ValueError("Masks must match original image dimensions; enable retina_masks=True.")
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
    for index, (box, score, class_id, raw_mask) in enumerate(zip(boxes, scores, classes, masks)):
        mask = raw_mask > 0.5
        if not mask.any():
            continue
        label = names[int(class_id)]
        bbox = [float(np.clip(box[0], 0, width)), float(np.clip(box[1], 0, height)),
                float(np.clip(box[2], 0, width)), float(np.clip(box[3], 0, height))]
        item = {
            "instance_id": index, "track_id": int(ids[index]) if ids is not None else None,
            "class_id": int(class_id), "label": label, "confidence": float(score),
            "role": role_for(label, primary_crop, float(score), review_threshold),
            "bbox_xyxy": bbox, **mask_geometry(mask),
        }
        if include_rle:
            item["mask_rle"] = encode_rle(mask)
        record["detections"].append(item)
    return record


def load_coverage(weights: Path, primary_crop: str) -> dict:
    """Bind supported crop profiles to the exact trained weight file."""
    weights = Path(weights)
    coverage_path = weights.with_suffix(".cropguard.json")
    if not coverage_path.is_file():
        raise ValueError("Missing checkpoint coverage metadata (.cropguard.json). Use a checkpoint trained through CropGuard and keep its sidecar beside it.")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if not isinstance(coverage, dict) or not isinstance(coverage.get("available_class_ids"), list) or not isinstance(coverage.get("supported_primary_crops"), list):
        raise ValueError("Invalid checkpoint coverage metadata schema")
    available = coverage["available_class_ids"]
    if not available or any(type(i) is not int or i not in range(6) for i in available) or len(set(available)) != len(available):
        raise ValueError("Invalid available class IDs in checkpoint metadata")
    expected_crops = [name for i, name in [(1, "maize"), (2, "tomato"), (3, "chilli")] if i in available]
    if coverage["supported_primary_crops"] != expected_crops:
        raise ValueError("Checkpoint supported crops disagree with available class IDs")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != coverage.get("weights_sha256"):
        raise ValueError("Checkpoint SHA-256 does not match coverage metadata")
    if primary_crop not in coverage["supported_primary_crops"]:
        raise ValueError(f"Checkpoint has no training coverage for {primary_crop}; supported: {coverage['supported_primary_crops']}")
    return coverage


def predict(source: Path, weights: Path, output: Path, primary_crop: str,
            confidence: float = 0.25, review_threshold: float = 0.6,
            imgsz: int = 1024, device: str = "cpu", include_rle: bool = True,
            annotated: bool = True, max_detections: int = 300) -> Path:
    primary_crop = normalize_crop(primary_crop)
    if not 0 < confidence <= review_threshold <= 1:
        raise ValueError("Require 0 < confidence <= review_threshold <= 1")
    if max_detections < 1:
        raise ValueError("max_detections must be positive")
    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be a positive multiple of 32")
    source, weights, output = Path(source), Path(weights), Path(output)
    if not source.is_file() or not weights.is_file():
        raise ValueError("Source and trained weights must be existing local files")
    suffix = source.suffix.lower()
    if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported image/video extension: {suffix}")
    coverage = load_coverage(weights, primary_crop)
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if model.task != "segment":
        raise ValueError("A segmentation checkpoint is required")
    validate_names(model.names)
    output.mkdir(parents=True, exist_ok=False)
    options = dict(conf=confidence, imgsz=imgsz, device=device, retina_masks=True, verbose=False, max_det=max_detections)
    if coverage:
        options["classes"] = coverage["available_class_ids"]
    metadata = {"coverage": coverage, "weights": str(weights.resolve()), "source": str(source.resolve()),
                "primary_crop": primary_crop, "confidence": confidence,
                "review_threshold": review_threshold, "imgsz": imgsz, "device": device,
                "mask_rle": include_rle, "max_detections": max_detections, "status": "running"}
    metadata_path = output / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        if suffix in IMAGE_SUFFIXES:
            results = model.predict(str(source), **options)
            record = result_to_record(results[0], primary_crop, review_threshold, include_rle)
            (output / "predictions.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
            if annotated:
                if not cv2.imwrite(str(output / "annotated.jpg"), draw_record(results[0].orig_img, record)):
                    raise OSError("Unable to write annotated image")
        else:
            _predict_video(model, source, output, primary_crop, review_threshold, include_rle, annotated, options)
        metadata["status"] = "complete"
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def draw_record(frame, record):
    import cv2
    frame = frame.copy()
    colors = {"weed": (70, 70, 240), "primary_crop": (80, 200, 60),
              "other_crop": (240, 180, 60), "uncertain": (30, 210, 240)}
    for item in record["detections"]:
        x1, y1, x2, y2 = (int(v) for v in item["bbox_xyxy"])
        color = colors[item["role"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cx, cy = item["mask_point_xy"]
        cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
        label = f'{item["label"]} {item["confidence"]:.2f} [{item["role"]}]'
        if item["track_id"] is not None:
            label += f' #{item["track_id"]}'
        cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return frame


def _predict_video(model, source, output, primary_crop, review_threshold, include_rle, annotated, options):
    import cv2
    cap = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has invalid FPS; transcode with a known frame rate first")
        expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # One model per source keeps IDs scoped to a single video.
                result = model.track(frame, persist=True, tracker="bytetrack.yaml", **options)[0]
                record = result_to_record(result, primary_crop, review_threshold, include_rle)
                record.update(frame_index=count, timestamp_seconds=count / fps, fps=fps,
                              timestamp_basis="frame_index/fps (constant-frame-rate estimate)")
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                if annotated:
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise OSError("Cannot initialize MP4 writer; use --no-annotated or install video codecs")
                    writer.write(draw_record(frame, record))
                count += 1
        if count == 0:
            raise ValueError("Video contains no decodable frames")
        if expected_frames > count + 1:
            raise ValueError(f"Video decoding stopped after {count} frames; container reported {expected_frames}. Outputs are partial.")
        (output / "video.json").write_text(json.dumps({"frames_processed": count, "frames_reported": expected_frames, "fps": fps}), encoding="utf-8")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
