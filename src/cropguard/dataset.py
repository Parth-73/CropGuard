"""Strict YOLO polygon dataset checks before expensive training."""
from collections import Counter
from pathlib import Path
import hashlib
import math
import yaml
from PIL import Image
from .taxonomy import validate_names

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def inspect_dataset(config: Path, require_test: bool = False) -> dict:
    config = Path(config).resolve()
    try:
        spec = yaml.safe_load(config.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid dataset YAML: {exc}") from exc
    if not isinstance(spec, dict) or not isinstance(spec.get("names"), (dict, list)):
        raise ValueError("Dataset YAML must contain a names mapping or list")
    if any(not isinstance(spec.get(key, fallback), str) for key, fallback in [("path", "."), ("train", "images/train"), ("val", "images/val"), ("test", "images/test")]):
        raise ValueError("Dataset path and split locations must be directory strings")
    validate_names(spec["names"])
    root = Path(spec.get("path", "."))
    root = (config.parent / root).resolve() if not root.is_absolute() else root.resolve()
    errors, summary, hashes, missing_classes = [], {}, {}, []
    splits = ["train", "val"] + (["test"] if require_test else [])
    for split in splits:
        folder = root / spec.get(split, f"images/{split}")
        if not folder.is_dir() or folder.parent.name != "images":
            errors.append(f"{split}: expected an images/{split} directory: {folder}")
            continue
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            errors.append(f"{split}: no images in {folder}")
        counts, backgrounds = Counter(), 0
        expected_labels = set()
        for image in images:
            label = folder.parent.parent / "labels" / folder.name / image.relative_to(folder).with_suffix(".txt")
            if label in expected_labels:
                errors.append(f"{image}: multiple image extensions share a label path")
            expected_labels.add(label)
            try:
                with Image.open(image) as im:
                    im.verify()
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in hashes and hashes[digest][0] != split:
                    errors.append(f"Split leakage: {image} duplicates {hashes[digest][1]}")
                hashes[digest] = (split, image)
            except (OSError, ValueError) as exc:
                errors.append(f"{image}: invalid image: {exc}")
            if not label.is_file():
                errors.append(f"Missing label: {label}. Use an empty file only for reviewed background images.")
                continue
            lines = label.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                backgrounds += 1
            for row, line in enumerate(lines, 1):
                try:
                    fields = line.split()
                    class_id = int(fields[0])
                    points = [float(v) for v in fields[1:]]
                    if class_id not in range(6) or len(points) < 6 or len(points) % 2:
                        raise ValueError("expected class 0..5 and at least 3 polygon vertices")
                    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in points):
                        raise ValueError("coordinates must be finite and normalized to [0,1]")
                    xy = list(zip(points[::2], points[1::2]))
                    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(xy, xy[1:]+xy[:1]))) / 2
                    if area <= 1e-10:
                        raise ValueError("polygon has zero area")
                    counts[class_id] += 1
                except (ValueError, IndexError) as exc:
                    errors.append(f"{label}:{row}: {exc}")
        labels_dir = folder.parent.parent / "labels" / folder.name
        for label in labels_dir.rglob("*.txt"):
            if label not in expected_labels:
                errors.append(f"Orphan annotation: {label}")
        absent = [i for i in range(6) if counts[i] == 0]
        if absent:
            missing_classes.append(f"{split}: classes with no instances: {absent}")
        summary[split] = {"images": len(images), "backgrounds": backgrounds,
                          "instances_by_class": dict(sorted(counts.items()))}
    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors[:50]))
    return {"root": str(root), "splits": summary, "warnings": missing_classes}


def resolved_config(config: Path, destination: Path) -> Path:
    """Ultralytics gets an absolute root, independent of its global datasets_dir setting."""
    spec = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    root = Path(spec.get("path", "."))
    spec["path"] = str((Path(config).resolve().parent / root).resolve())
    destination.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return destination
