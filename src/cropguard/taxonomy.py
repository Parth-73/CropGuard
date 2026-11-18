"""Explicit roles: another crop is never automatically a weed."""

REQUIRED_NAMES = ("weed", "maize", "tomato", "chilli", "other_crop", "unknown_plant")
ALIASES = {"corn": "maize", "chili": "chilli", "chilly": "chilli"}


def normalize_crop(value: str) -> str:
    name = value.strip().lower()
    name = ALIASES.get(name, name)
    if name not in {"maize", "tomato", "chilli"}:
        raise ValueError("Primary crop must be maize, tomato, or chilli (corn/chili/chilly accepted). "
                         "Add species annotations and retrain before extending this list.")
    return name


def validate_names(names: dict | list) -> dict[int, str]:
    mapping = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if mapping != dict(enumerate(REQUIRED_NAMES)):
        raise ValueError(f"Checkpoint/dataset taxonomy must be {dict(enumerate(REQUIRED_NAMES))}; got {mapping}. "
                         "Generic COCO weights cannot identify these plants. Train a CropGuard checkpoint first.")
    return mapping


def role_for(label: str, primary_crop: str, confidence: float, review_threshold: float) -> str:
    if confidence < review_threshold or label == "unknown_plant":
        return "uncertain"
    if label == "weed":
        return "weed"
    if label == primary_crop:
        return "primary_crop"
    return "other_crop"
