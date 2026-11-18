"""Evaluate a completed training run and produce reviewable image/video examples."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/maize"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=512)
    args = parser.parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    run = args.run
    weights = run / "fit/weights/best.pt"
    # Sidecars from an interrupted or older training entry point get a verified digest.
    coverage_path = weights.with_suffix(".cropguard.json")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    from cropguard.training import evaluate
    from cropguard.localization import evaluate_localization
    from cropguard.inference import predict
    import cv2
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    metrics = evaluate(Path("configs/weedsgalore.yaml"), weights, output / "test-evaluation", imgsz=args.imgsz, device=args.device)
    localization = evaluate_localization(Path("configs/weedsgalore.yaml"), weights, output / "localization-test.json", confidence=.6, imgsz=args.imgsz, device=args.device)
    coverage["evaluation_status"] = "evaluated on official image test split; not independently field validated"
    coverage["test_metrics"] = metrics["metrics"]
    coverage["recommended_imgsz"] = args.imgsz
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    source = sorted(Path("data/weedsgalore/images/test").glob("*.png"))[0]
    predict(source, weights, output / "image", "maize", imgsz=args.imgsz, device=args.device)
    # A static-frame clip checks the actual tracking path without implying real motion validation.
    source_frame = cv2.imread(str(source))
    height, width = source_frame.shape[:2]
    demo = output / "static_test_clip.avi"
    writer = cv2.VideoWriter(str(demo), cv2.VideoWriter_fourcc(*"MJPG"), 5, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Unable to create smoke-test clip")
    try:
        for _ in range(5):
            writer.write(source_frame)
    finally:
        writer.release()
    predict(demo, weights, output / "video", "maize", imgsz=args.imgsz, device=args.device)
    # Also check an existing stock reference image, which has no ground truth.
    reference = Path("weed Image.webp")
    if reference.is_file():
        predict(reference, weights, output / "reference-image", "maize", imgsz=args.imgsz, device=args.device)
    release = Path("models/maize_baseline")
    release.mkdir(parents=True, exist_ok=False)
    shutil.copy2(weights, release / "best.pt")
    shutil.copy2(coverage_path, release / "best.cropguard.json")
    (output / "summary.json").write_text(json.dumps({"weights": str((release / "best.pt").resolve()),
        "test_metrics": metrics["metrics"], "localization": localization["per_class"],
        "smoke_video": "Five repeated test-image frames; verifies tracking/output code, not moving-field performance",
        "status": "baseline; maize/weed coverage only"}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "release": str(release), "outputs": str(output)}, indent=2))


if __name__ == "__main__":
    main()
