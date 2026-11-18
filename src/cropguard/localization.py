"""Deployment-threshold localization audit against original dense WeedsGalore masks."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .dataset import inspect_dataset
from .inference import load_coverage


def match_instances(predictions, references, iou_threshold=0.5):
    """Greedy one-to-one matching ordered by mask IoU, preserving category."""
    def bounds(mask):
        rows, cols = np.nonzero(mask)
        return (int(cols.min()), int(rows.min()), int(cols.max()+1), int(rows.max()+1))
    pred_bounds = [bounds(mask) for _, mask in predictions]
    ref_bounds = [bounds(mask) for _, mask in references]
    pred_areas = [int(mask.sum()) for _, mask in predictions]
    ref_areas = [int(mask.sum()) for _, mask in references]
    candidates = []
    for pi, (pc, pm) in enumerate(predictions):
        for gi, (gc, gm) in enumerate(references):
            if pc != gc:
                continue
            a, b = pred_bounds[pi], ref_bounds[gi]
            x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
            if x1 >= x2 or y1 >= y2:
                continue
            inter = np.count_nonzero(pm[y1:y2, x1:x2] & gm[y1:y2, x1:x2])
            if not inter:
                continue
            union = pred_areas[pi] + ref_areas[gi] - inter
            iou = inter / union
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))
    used_p, used_g, matches = set(), set(), []
    for iou, pi, gi in sorted(candidates, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matches.append((pi, gi, iou))
    return matches


def evaluate_localization(data: Path, weights: Path, output: Path, split="test",
                          confidence=0.6, imgsz=512, device="cpu"):
    if split not in {"val", "test"} or not 0 < confidence <= 1:
        raise ValueError("Use val/test and a confidence threshold in (0,1]")
    audit = inspect_dataset(data, require_test=split == "test")
    root = Path(audit["root"])
    references_dir = root / "ground_truth_instances" / split
    if not references_dir.is_dir():
        raise ValueError("This audit requires original WeedsGalore instance masks from the importer")
    coverage = load_coverage(weights, "maize")
    from ultralytics import YOLO
    model = YOLO(str(weights))
    counts = {c: {"predicted": 0, "reference": 0, "matched": 0, "centroid_errors": []} for c in [0, 1]}
    intersections, unions = np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    crop_overlaps, predicted_weeds = 0, 0
    per_image = []
    for image in sorted((root / "images" / split).glob("*.png")):
        labels = np.asarray(Image.open(root / "ground_truth" / split / image.name))
        ids = np.asarray(Image.open(references_dir / image.name))
        refs = []
        for instance_id in np.unique(ids):
            if instance_id == 0:
                continue
            mask = (ids == instance_id) & (labels > 0)
            if mask.any():
                cls = 1 if np.count_nonzero(labels[mask] == 1) > mask.sum()/2 else 0
                refs.append((cls, mask))
                counts[cls]["reference"] += 1
        result = model.predict(str(image), conf=confidence, imgsz=imgsz, device=device,
                               retina_masks=True, classes=coverage["available_class_ids"],
                               verbose=False)[0]
        preds = []
        predicted_semantics = np.zeros(labels.shape, dtype=np.uint8)
        if result.masks is not None:
            raw_masks = result.masks.data.cpu().numpy() > .5
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            # Higher-confidence instances win any overlap in the semantic map.
            for index in np.argsort(scores):
                cls, mask = int(classes[index]), raw_masks[index]
                if cls not in counts or not mask.any():
                    continue
                preds.append((cls, mask))
                counts[cls]["predicted"] += 1
                predicted_semantics[mask] = 1 if cls == 1 else 2
                if cls == 0:
                    predicted_weeds += 1
                    if np.count_nonzero(mask & (labels == 1)) / mask.sum() >= .1:
                        crop_overlaps += 1
        matched = match_instances(preds, refs)
        for pi, gi, iou in matched:
            cls, pm = preds[pi]
            gm = refs[gi][1]
            py, px = np.nonzero(pm)
            gy, gx = np.nonzero(gm)
            error = float(np.hypot(px.mean()-gx.mean(), py.mean()-gy.mean()))
            counts[cls]["matched"] += 1
            counts[cls]["centroid_errors"].append(error)
        for cls in [0, 1, 2]:
            p, g = predicted_semantics == cls, labels == cls
            intersections[cls] += np.count_nonzero(p & g)
            unions[cls] += np.count_nonzero(p | g)
        per_image.append({"image": image.name, "references": len(refs), "predictions": len(preds), "matched": len(matched)})
    per_class = {}
    for cls, stats in counts.items():
        errors = stats.pop("centroid_errors")
        per_class["maize" if cls == 1 else "weed"] = {**stats,
            "precision_at_mask_iou_0_5": stats["matched"] / stats["predicted"] if stats["predicted"] else 0,
            "recall_at_mask_iou_0_5": stats["matched"] / stats["reference"] if stats["reference"] else 0,
            "matched_centroid_mean_error_px": float(np.mean(errors)) if errors else None,
            "matched_centroid_p95_error_px": float(np.percentile(errors, 95)) if errors else None}
    report = {"split": split, "confidence_threshold": confidence, "imgsz": imgsz,
              "mask_iou_match_threshold": .5, "per_class": per_class,
              "semantic_iou": {name: float(intersections[i] / unions[i]) if unions[i] else None for i,name in enumerate(["background", "maize", "weed"])},
              "weed_predictions_overlapping_crop_by_at_least_10_percent": crop_overlaps,
              "predicted_weeds": predicted_weeds, "per_image": per_image,
              "limitations": "Centroid error covers correctly classified matched masks only; unmatched plants counted by precision/recall. Greedy matching and original dense masks differ from polygon-based COCO mAP. No stem, physical distance, or new-field accuracy is established."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report
