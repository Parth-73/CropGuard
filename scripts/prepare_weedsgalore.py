"""Convert original WeedsGalore semantic + instance masks into YOLO segmentation.

Preserves official splits; RGB is assembled from 16-bit R/G/B bands.
Never interprets arbitrary instance IDs as species labels.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import urllib.request
import zipfile

import cv2
import numpy as np
from PIL import Image
import yaml

from cropguard.taxonomy import REQUIRED_NAMES

SPLIT_BASE = "https://raw.githubusercontent.com/GFZ/weedsgalore/main/src/datasets/weedsgalore/splits/"
EXPECTED_SHA256 = "5d8395c274d990cd8fbfa728445f65c398be2b5d777bd5145eb9ae5ca4d9b68c"
SOURCE = "https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip"


def polygon_for_mask(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c.reshape(-1, 2).astype(float) for c in contours if cv2.contourArea(c) >= 2 and len(c) >= 3]
    if not contours:
        return None
    if len(contours) == 1:
        return contours[0]
    # Bridges encode disconnected leaves in YOLO's single polygon per instance.
    # Ultralytics uses this conversion for multi-part COCO polygons as well.
    from ultralytics.data.converter import merge_multi_segment
    merged = merge_multi_segment([c.reshape(-1).tolist() for c in contours])
    return np.concatenate(merged, axis=0)


def prepare(archive: Path, output: Path, config: Path):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    archive = Path(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Archive checksum differs from verified original WeedsGalore release")
    split_cache = archive.parent / "splits"
    split_cache.mkdir(exist_ok=True)
    split_names = {}
    for split in ["train", "val", "test"]:
        target = split_cache / f"{split}.txt"
        if not target.exists():
            urllib.request.urlretrieve(SPLIT_BASE + target.name, target)
        split_names[split] = [s.strip() for s in target.read_text().splitlines() if s.strip()]
    seen = set()
    for split, names in split_names.items():
        if len(names) != len(set(names)) or seen.intersection(names):
            raise ValueError("Duplicate samples in official split lists")
        seen.update(names)
    summary = {}
    with zipfile.ZipFile(archive) as z:
        files = {name: name for name in z.namelist()}
        def read_ending(suffix):
            matches = [name for name in files if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected one archive member ending in {suffix}; found {len(matches)}")
            return np.asarray(Image.open(io.BytesIO(z.read(matches[0]))))
        for split, names in split_names.items():
            image_dir, label_dir = output / "images" / split, output / "labels" / split
            gt_dir = output / "ground_truth" / split
            gt_instances = output / "ground_truth_instances" / split
            for folder in [image_dir, label_dir, gt_dir, gt_instances]:
                folder.mkdir(parents=True, exist_ok=True)
            counts = Counter()
            for sample in names:
                if Path(sample).name != sample:
                    raise ValueError(f"Invalid sample ID: {sample}")
                date = sample[:10]
                bands = [read_ending(f"{date}/images/{sample}_{band}.png") for band in ["R", "G", "B"]]
                rgb = np.stack(bands, axis=-1)
                if rgb.dtype == np.uint16:
                    rgb = np.round(rgb.astype(np.float32) / 257).astype(np.uint8)
                elif rgb.dtype != np.uint8:
                    raise ValueError(f"Unexpected band dtype: {rgb.dtype}")
                semantics = read_ending(f"{date}/semantics/{sample}.png")
                instances = read_ending(f"{date}/instances/{sample}.png")
                if semantics.ndim != 2 or instances.ndim != 2 or semantics.shape != rgb.shape[:2] or instances.shape != semantics.shape:
                    raise ValueError(f"Annotation shape mismatch: {sample}")
                if not set(np.unique(semantics)).issubset(set(range(6))):
                    raise ValueError(f"Unexpected semantic IDs: {np.unique(semantics)}")
                height, width = semantics.shape
                labels = []
                for instance_id in np.unique(instances):
                    if instance_id == 0:
                        continue
                    mask = instances == instance_id
                    values, freq = np.unique(semantics[mask], return_counts=True)
                    foreground = values > 0
                    if not foreground.any():
                        continue
                    values, freq = values[foreground], freq[foreground]
                    category = int(values[freq.argmax()])
                    # A single instance must not be a mixture of crop and weed.
                    maize_pixels = int(freq[values == 1].sum())
                    if maize_pixels and maize_pixels != int(freq.sum()):
                        raise ValueError(f"Mixed crop/weed instance in {sample}: {instance_id}")
                    class_id = 1 if category == 1 else 0
                    mask &= semantics > 0
                    poly = polygon_for_mask(mask)
                    if poly is None:
                        continue
                    poly = np.clip(poly / [width, height], 0, 1)
                    labels.append(str(class_id) + " " + " ".join(f"{v:.7f}" for v in poly.reshape(-1)))
                    counts[class_id] += 1
                Image.fromarray(rgb).save(image_dir / f"{sample}.png")
                (label_dir / f"{sample}.txt").write_text("\n".join(labels), encoding="utf-8")
                # Retain original dense labels for an independent pixel IoU check.
                binary = np.where(semantics == 0, 0, np.where(semantics == 1, 1, 2)).astype(np.uint8)
                Image.fromarray(binary).save(gt_dir / f"{sample}.png")
                Image.fromarray(instances).save(gt_instances / f"{sample}.png")
            summary[split] = {"images": len(names), "instances": dict(counts)}
        licenses = [n for n in files if n.lower().endswith(("license.txt", "licence.txt"))]
        if licenses:
            (output / "SOURCE_LICENSE.txt").write_bytes(z.read(licenses[0]))
    with archive.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {"dataset": "WeedsGalore", "source": SOURCE,
                  "paper": "https://openaccess.thecvf.com/content/WACV2025/html/Celikkan_WeedsGalore_A_Multispectral_and_Multitemporal_UAV-Based_Dataset_for_Crop_and_WACV_2025_paper.html",
                  "license": "CC BY 4.0", "archive_sha256": archive_hash,
                  "supported_primary_crops": ["maize"], "splits": summary,
                  "conversion": "Official instance IDs labeled by semantic masks; external contours bridge disconnected leaves; polygon holes are not represented; tiny contours under 2 px area omitted. RGB uint16 converted by fixed /257 scale.",
                  "split_source": SPLIT_BASE}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump({"path": Path(os.path.relpath(output.resolve(), config.resolve().parent)).as_posix(), "train": "images/train", "val": "images/val",
                                     "test": "images/test", "names": dict(enumerate(REQUIRED_NAMES))}, sort_keys=False), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/weedsgalore/original.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/weedsgalore"))
    parser.add_argument("--config", type=Path, default=Path("configs/weedsgalore.yaml"))
    args = parser.parse_args()
    prepare(args.archive, args.output, args.config)
