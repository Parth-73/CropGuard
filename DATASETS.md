# Dataset selection and coverage

## Used for the first checkpoint

**WeedsGalore (WACV 2025)** provides multispectral imagery and dense semantic and instance annotations for maize and weeds. We use only the RGB channels, preserving official 104/26/26 image splits. Source: https://github.com/GFZ/weedsgalore . Original archive: https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip . Dataset license: CC BY 4.0. It does not contain tomato or chilli crop labels.

The importer derives each instance's class from the original semantic mask (background=0, maize=1, weed species=2..5), collapses weed species into `weed`, and maps maize to `maize`. External mask contours become YOLO polygons. Tiny contours below 2 px area are omitted; holes cannot be represented in YOLO's polygon labels. Disconnected leaves are joined using the Ultralytics polygon conversion utility. Raw semantic ground truth is retained for independent pixel-level analysis. These details affect how instance/mask metrics should be interpreted.

We inspected the Project-AgML Hugging Face mirror (SHA-256 e6547f722cf6d00d0222ea6e8cc3550aa1393b860ff97bc951ec765e024bfec7). Its `mask` column contains numerous instance IDs rather than 0..5 semantic classes, so it is deliberately not used to assign crop/weed labels. The original archive contains both annotations and avoids this ambiguity.

## Further sources checked

| Source | Usable supervision | Coverage / limitation |
| --- | --- | --- |
| [CropAndWeed](https://github.com/cropandweed/cropandweed-dataset) | Whole-plant boxes, stems, semantic masks | Broad crop/weed diversity and maize; the published taxonomy lacks tomato/chilli. Non-commercial dataset terms. Not downloaded for this baseline. |
| [DIWEED / maize-tomato dataset](https://doi.org/10.20350/DIGITALCSIC/16559) | Primarily cropped species-classification examples | Has maize and tomato plus weeds, but classification images alone cannot supervise full-frame plant coordinates. Needs source localization annotations or manual annotation. |
| [Ground-based tomato field dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC12670878/) | Pascal VOC bounding boxes, 2021/2022 tomato fields | Appropriate for tomato detection; needs instance masks for this segmentation trainer. Preserve season splits. |
| [Chilli-weed repository](https://github.com/sbhaktavatsala/chilli-weed-dataset) | Downloadable JPGs at inspection time | README describes annotations, but the actual Git tree inspected contained JPGs, two text placeholders, README and license, with no XML/YOLO annotation files. Cannot treat image filenames as plant locations. |

No checked public source supplies complete maize/tomato/chilli instance segmentation coverage in one ready-to-train dataset. Tomato and chilli remain explicit data gaps; this implementation does not claim unseen-species robustness or invent those labels.
