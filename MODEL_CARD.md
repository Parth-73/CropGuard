# CropGuard maize/weed baseline model card

Evaluated 18 November 2026. **Research baseline; not yet a robust multi-crop model.**

## Supported behavior and limits

The checkpoint predicts maize and an aggregate weed class, with original-image mask and box coordinates. It does **not** have training coverage for tomato, chilli, other crops, or unknown plants. The inference pipeline rejects unsupported primary-crop selections and filters untrained class IDs. Six names in the checkpoint vocabulary do not imply six trained classes.

Weed recall is low, especially at the conservative review threshold. Many small weeds are missed. This checkpoint must not be described as detecting every plant or returning exact ground-truth coordinates. Pixel coordinates describe its predicted masks. Generalization to new crop species, ground-rover camera views, new fields, seasons, motion, and occlusion has not been demonstrated.

## Artifact and training

- Weights: [models/maize_baseline/best.pt](models/maize_baseline/best.pt)
- Required matching coverage metadata: [best.cropguard.json](models/maize_baseline/best.cropguard.json)
- Weight SHA-256: `cba6a5814373e4e9973ef585f44136f8160db9eb2cec02bfd038d2d7edeff4b6`
- Architecture/base: YOLO11n instance segmentation, initialized from Ultralytics COCO pretrained `yolo11n-seg.pt`.
- Training: 60 completed epochs, 512px input, batch 8, seed 42, AdamW initial LR 0.001, cosine decay, moderate geometric/color augmentation, mosaic disabled for the last 15 epochs.
- Epochs 1-36 used PyTorch 2.15.0+cpu; epochs 37-60 resumed on an RTX 4050 (6 GB), PyTorch 2.11.0+cu128. Resume restored optimizer, scheduler, epoch, and best-fitness state. This device/version transition can change numerical results and augmentation sequences.
- Best checkpoint selected by the training framework's validation fitness; inference resolution selected using validation mask mAP50-95. The test set was then evaluated once for this selected checkpoint/resolution. The separate localization audit uses that same test split and fixed 0.60 threshold; no threshold was tuned using test results.

## Dataset and provenance

[WeedsGalore, Celikkan et al., WACV 2025](https://github.com/GFZ/weedsgalore), CC BY 4.0. Original RGB bands and paired semantic/instance annotations were downloaded from the authors. Official image splits: 104 train / 26 validation / 26 test. Converted instance counts: 7,954 / 2,256 / 1,964 respectively. The original dense test masks contain four additional tiny weed instances omitted by polygon conversion.

Each source instance is assigned its semantic crop/weed label. All four weed categories are collapsed into weed. Tiny contour parts below 2 px area are omitted, holes cannot be represented in YOLO polygon annotations, and disconnected leaves are joined for training. The localization audit instead uses the original dense masks. See [DATASETS.md](DATASETS.md) and [data provenance](reports/dataset-provenance.json).

Exact duplicate image leakage was checked. Official image splits do not establish independence across fields, acquisition dates, adjacent locations, or real videos. The source is overhead multispectral imagery assembled into RGB, rather than a broad collection of ground-level farm-camera images.

## Validation resolution selection

| Input size | Box mAP50 | Mask mAP50 | Mask mAP50-95 |
| --- | --- | --- | --- |
| 512 | 65.4% | 51.7% | 15.8% |
| 768 | 68.5% | 53.4% | 15.5% |
| 1024 | 67.7% | 48.8% | 13.4% |

512px was selected from this comparison. Increasing inference resolution alone did not improve this trained checkpoint. [Full selection report](reports/validation-resolution-study.json).

## Test detection and segmentation

26 reserved test images. Macro box mAP50: **64.9%**; macro mask mAP50: **52.2%**. These are average-precision metrics, not overall accuracy percentages.

| Class | Box precision | Box recall | Box mAP50 | Mask mAP50 | Mask mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| weed | 72.2% | 34.6% | 43.1% | 35.1% | 10.9% |
| maize | 80.4% | 85.5% | 86.7% | 69.3% | 21.6% |

Precision/recall here are the Ultralytics evaluation operating point, not necessarily confidence 0.60. [Machine-readable evaluation](reports/evaluation.json) and selected plots are in [reports/figures](reports/figures).

## Coordinate audit at confidence 0.60

Original dense masks, original 600x600 pixel images, same-class one-to-one greedy matches with mask IoU >= 0.5. Centroid error is measured **only for matched predictions**. Low error on this subset does not compensate for missed plants or demonstrate exact stem localization.

| Class | Precision | Recall | Matched/reference plants | Mean centroid error (px) | P95 error (px) |
| --- | --- | --- | --- | --- | --- |
| weed | 90.2% | 11.8% | 202/1712 | 1.64 | 5.85 |
| maize | 90.9% | 58.6% | 150/256 | 2.40 | 5.31 |

Semantic IoU at that threshold: maize 45.4%, weeds 22.2%. 1 predicted weed masks overlap crop pixels by at least 10% of their mask area. These measurements are image-space diagnostics, not a physical targeting validation. [Detailed localization audit](reports/localization-test.json).

Default prediction confidence floor is 0.25; predictions from 0.25 to 0.60 are retained with role `uncertain`. Confidence scores are not calibrated probabilities. The audit above reflects predictions at/above 0.60, so it should not be confused with the count of all retained low-confidence predictions.

## Image/video checks and outputs

- 16 automated tests pass in both CPU and CUDA environments.
- All 60 epoch records were verified; released checkpoint SHA-256 matches its coverage metadata.
- Actual image inference passed on GPU and CPU. RLE masks round-trip exactly, dimensions match original images, mask areas agree, and returned mask points lie inside their masks.
- Actual ByteTrack/video inference, JSONL writing, and annotated MP4 writing passed on a five-frame repeated-image clip. Tracking IDs were returned; this is a code-path check, **not a moving-field video benchmark or proof of stable identity tracking**.
- Unsupported tomato prediction was rejected before creating an output folder.
- [Verification record](reports/verification.json).

Inspect [image JSON](examples/image-predictions.json), [annotated image](examples/annotated-image.jpg), [video JSONL](examples/video-predictions.jsonl), and [annotated video](examples/annotated-video.mp4). A [dense-scene image](examples/dense-annotated.jpg) was selected for illustration by greatest annotated weed pixel coverage, not model confidence. The supplied stock reference image was also processed but has no ground truth and supplies no accuracy evidence.

## What remains before the requested robust multi-crop model exists

Obtain whole-plant instance labels for tomato/chilli and representative additional vegetation, add multiple field/camera/season domains, and retrain. Improve small-weed recall on validation data, measure localization against independently checked ground truth, and reserve genuinely new fields/videos for an independent generalization test. Crop selection in software and a public maize-only dataset cannot supply that missing supervision. Metric ground coordinates additionally require camera/ground calibration and rover pose; a mask centroid or in-mask point is not a stem or physical actuation coordinate.
