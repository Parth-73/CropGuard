"""Run with the ML extra installed; real Ultralytics tensor/result contracts."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("ultralytics")
import torch
from ultralytics.engine.results import Results
from cropguard.inference import result_to_record, predict, _predict_video
from cropguard.taxonomy import REQUIRED_NAMES


def make_result(frame=None):
    if frame is None:
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
    mask = torch.zeros((1, 64, 96))
    mask[0, 12:40, 20:60] = 1
    # Real seven-column tracked Results use xyxy, track_id, confidence, class_id.
    boxes = torch.tensor([[20, 12, 60, 40, 9, 0.92, 1]])
    return Results(frame, path="sample.png", names=dict(enumerate(REQUIRED_NAMES)), boxes=boxes, masks=mask)


def test_real_result_contract_preserves_mask_and_tracking():
    record = result_to_record(make_result(), "maize", .6)
    item = record["detections"][0]
    assert item["track_id"] == 9
    assert item["centroid_xy"] == [40, 26]
    assert item["area_px"] == 1120
    assert sum(item["mask_rle"]["counts"]) == 64 * 96


def test_video_decode_stream_jsonl_and_cleanup(tmp_path):
    import cv2
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5, (96, 64))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((64, 96, 3), index * 30, dtype=np.uint8))
    writer.release()
    output = tmp_path / "predicted"
    output.mkdir()
    class Model:
        calls = 0
        def track(self, frame, **kwargs):
            assert kwargs["persist"] is True
            self.calls += 1
            return [make_result(frame)]
    model = Model()
    _predict_video(model, source, output, "maize", .6, True, True, {})
    records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [r["frame_index"] for r in records] == [0, 1, 2]
    assert records[2]["timestamp_seconds"] == .4
    assert model.calls == 3
    cap = cv2.VideoCapture(str(output / "annotated.mp4"))
    try:
        assert cap.isOpened()
        assert cap.read()[0]
    finally:
        cap.release()
