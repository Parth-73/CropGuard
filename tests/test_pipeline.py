from types import SimpleNamespace
import json
import numpy as np
import pytest
import yaml
from PIL import Image
from cropguard.geometry import encode_rle, mask_geometry
from cropguard.taxonomy import normalize_crop, role_for, validate_names, REQUIRED_NAMES
from cropguard.inference import result_to_record
from cropguard.dataset import inspect_dataset


@pytest.mark.parametrize("shape", [(1, 1), (4, 7), (17, 11)])
def test_mask_rle_roundtrip(shape):
    rng = np.random.default_rng(42)
    for mask in [np.zeros(shape, dtype=np.uint8), np.ones(shape, dtype=np.uint8), rng.integers(0, 2, shape, dtype=np.uint8)]:
        encoded = encode_rle(mask)
        pixels = np.repeat(np.arange(len(encoded["counts"])) % 2, encoded["counts"])
        assert np.array_equal(pixels.reshape(encoded["size"]), mask)


def test_concave_mask_point_stays_inside():
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[1:9, 1] = 1
    mask[8, 1:18] = 1
    geo = mask_geometry(mask)
    x, y = geo["mask_point_xy"]
    assert mask[int(y), int(x)]
    assert geo["mask_bbox_xyxy"] == [1, 1, 18, 9]
    assert geo["area_px"] == 24


def test_other_crops_and_uncertainty_are_preserved():
    assert normalize_crop("chilly") == "chilli"
    assert role_for("tomato", "maize", .95, .6) == "other_crop"
    assert role_for("weed", "maize", .45, .6) == "uncertain"
    assert role_for("unknown_plant", "maize", .99, .6) == "uncertain"
    with pytest.raises(ValueError, match="COCO"):
        validate_names({0: "person"})


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value


class Boxes:
    def __init__(self):
        self.xyxy = Tensor([[-2, 0, 22, 9]])
        self.conf = Tensor([.9])
        self.cls = Tensor([1])
        self.id = Tensor([7])
    def __len__(self): return 1


def test_original_resolution_coordinates_and_track_ids():
    mask = np.zeros((1, 10, 20), dtype=np.uint8)
    mask[0, 2:6, 4:10] = 1
    result = SimpleNamespace(names=dict(enumerate(REQUIRED_NAMES)), orig_shape=(10, 20),
                             boxes=Boxes(), masks=SimpleNamespace(data=Tensor(mask)))
    record = result_to_record(result, "maize", .6)
    item = record["detections"][0]
    assert item["bbox_xyxy"] == [0, 0, 20, 9]
    assert item["centroid_xy"] == [7, 4]
    assert item["track_id"] == 7
    assert item["role"] == "primary_crop"
    json.dumps(record, allow_nan=False)
    result.masks = SimpleNamespace(data=Tensor(np.zeros((1, 5, 10))))
    with pytest.raises(ValueError, match="original"):
        result_to_record(result, "maize", .6)


def make_dataset(root):
    for index, split in enumerate(["train", "val", "test"]):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (12, 12), (index * 50, 0, 0)).save(root / "images" / split / "a.png")
        (root / "labels" / split / "a.txt").write_text("1 0.1 0.1 0.8 0.1 0.8 0.8", encoding="utf-8")
    config = root / "dataset.yaml"
    config.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                     "test": "images/test", "names": list(REQUIRED_NAMES)}), encoding="utf-8")
    return config


def test_dataset_rejects_leakage_and_bad_annotations(tmp_path):
    config = make_dataset(tmp_path)
    report = inspect_dataset(config, require_test=True)
    assert report["splits"]["train"]["images"] == 1
    (tmp_path / "images/val/a.png").write_bytes((tmp_path / "images/train/a.png").read_bytes())
    with pytest.raises(ValueError, match="Split leakage"):
        inspect_dataset(config)
    Image.new("RGB", (12, 12), "blue").save(tmp_path / "images/val/a.png")
    (tmp_path / "labels/train/a.txt").write_text("1 nan 0 1 0 1 1", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        inspect_dataset(config)


def test_missing_labels_are_not_assumed_background(tmp_path):
    config = make_dataset(tmp_path)
    (tmp_path / "labels/val/a.txt").unlink()
    with pytest.raises(ValueError, match="Missing label"):
        inspect_dataset(config)


def test_coverage_is_bound_to_weights_and_supported_species(tmp_path):
    import hashlib
    from cropguard.inference import load_coverage
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"test-only-checkpoint")
    with pytest.raises(ValueError, match="coverage metadata"):
        load_coverage(weights, "maize")
    sidecar = weights.with_suffix(".cropguard.json")
    sidecar.write_text(json.dumps({"weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                                  "supported_primary_crops": ["maize"], "available_class_ids": [0, 1]}), encoding="utf-8")
    assert load_coverage(weights, "maize")["available_class_ids"] == [0, 1]
    with pytest.raises(ValueError, match="no training coverage"):
        load_coverage(weights, "tomato")
    weights.write_bytes(b"different-checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        load_coverage(weights, "maize")


def test_instance_matching_is_one_to_one_and_preserves_class():
    from cropguard.localization import match_instances
    a = np.zeros((10, 10), dtype=bool)
    a[2:7, 2:7] = True
    b = np.zeros_like(a)
    b[6:9, 6:9] = True
    matches = match_instances([(0, a), (0, a.copy()), (1, b)], [(0, a), (0, b)])
    assert len(matches) == 1
    assert matches[0][1] == 0
    assert matches[0][2] == 1


@pytest.mark.parametrize("text", ["", "[]", "names: null", "names: [weed]\npath: [bad]"])
def test_dataset_rejects_invalid_config(tmp_path, text):
    config = tmp_path / "invalid.yaml"
    config.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        inspect_dataset(config)
