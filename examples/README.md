# Example inputs and predictions

The input imagery is derived from **WeedsGalore**, by Ekin Celikkan, Timo Kunzmann, Yertay Yeskaliyev, Sibylle Itzerott, Nadja Klein, and Martin Herold (WACV 2025), under CC BY 4.0. [Source](https://github.com/GFZ/weedsgalore), [license](../docs/licenses/WEEDSGALORE-CC-BY-4.0.txt), [provenance](../reports/dataset-provenance.json).

Modifications: original 16-bit R/G/B bands were assembled into an 8-bit RGB image using a fixed /257 scale. Annotated images/video and JSON predictions were generated with the CropGuard maize/weed baseline.

- `input-image.png`: the first official test sample in filename order, used for the image smoke test.
- `annotated-image.jpg` / `image-predictions.json`: its image inference outputs.
- `dense-input.png`: test sample `2023-06-15_0735`, selected by greatest annotated weed pixel coverage among test samples containing maize, not prediction confidence.
- `dense-annotated.jpg` / `dense-predictions.json`: its predictions. The many uncertain detections illustrate current limitations.
- `annotated-video.mp4` / `video-predictions.jsonl`: a five-frame clip formed by repeating the first test image at 5 FPS. It checks tracking, JSONL, and video encoding, not moving-field accuracy or stable plant identities.

Coordinate conventions and limitations are described in the [README](../README.md) and [model card](../MODEL_CARD.md). The annotated images reflect a baseline model; visible boxes are predictions rather than ground truth.
