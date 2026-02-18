import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

BASE_PATH = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_PATH.parent   # one level up


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference")

    parser.add_argument(
        "--exp",
        type=str,
        default="kitti_exp5",
        help="Experiment folder name containing trained weights"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="valid/",
        help="Source for inference (image, folder, video file, webcam index)"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (should match training size)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save inference results"
    )

    return parser.parse_args()


def load_model(exp_name: str) -> YOLO:
    """Load trained YOLO model weights."""

    weights_path = (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "runs01"
        / exp_name
        / "weights"
        / "best.pt"
    )

    if not weights_path.exists():
        raise FileNotFoundError(f"Model not found at: {weights_path}")

    print(f"✅ Using weights: {weights_path.resolve()}")

    return YOLO(weights_path)


def run_inference(model: YOLO, args):
    """Run inference on the specified source."""
    print("🚀 Running inference...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_path = PROJECT_ROOT / args.source

    results = model.predict(
        source=source_path,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        save=args.save
    )

    print("✅ Inference completed.")

    return results


def main():
    args = parse_args()

    model = load_model(args.exp)

    run_inference(model, args)


if __name__ == "__main__":
    main()