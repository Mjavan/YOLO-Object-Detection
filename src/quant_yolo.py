import torch
import torch.nn as nn
from ultralytics import YOLO
import argparse 
import subprocess
from pathlib import Path


import argparse
from ultralytics import YOLO

def export_int8_engine(model_path):
    print("\n🚀 Exporting TensorRT INT8 engine...")

    model = YOLO(model_path)

    model.export(
        format="engine",
        int8=True,
        data="kitti.yaml"
    )

    print("✅ INT8 engine export complete.\n")


def run_validation(args):

    print("\n==============================")
    print("Running validation")
    print("==============================")

    device = 0 if torch.cuda.is_available() else "cpu"
    BASE_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_PATH.parent   # one level up
    MODEL_PATH = (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "runs01"
        / "kitti_exp15"
        / "weights"
        / "best.pt"
    )

    # 🔹 Optional INT8 export
    if args.export_int8:
        export_int8_engine(MODEL_PATH)

    if args.use_engine:
        print(f"Using TensorRT engine for inference: {MODEL_PATH.with_suffix('.engine')}")
        model = YOLO(str(MODEL_PATH).replace(".pt", ".engine"))
    else:
        model = YOLO(MODEL_PATH)

    print(f"half percesion: {args.half}")

    results = model.val(
        data='kitti.yaml',
        imgsz= 832,
        device=device,
        half=args.half,
        batch=args.batch,
        verbose=False
    )

    # Speed metrics (ms/image)
    speed = results.speed

    preprocess = speed["preprocess"]
    inference = speed["inference"]
    postprocess = speed["postprocess"]

    total_latency = preprocess + inference + postprocess
    fps = 1000 / total_latency if total_latency > 0 else 0

    print("\n📊 Accuracy Metrics")
    print("------------------------------")
    print(f"mAP50:      {results.box.map50:.4f}")
    print(f"mAP50-95:   {results.box.map:.4f}")
    print(f"Precision:  {results.box.mp:.4f}")
    print(f"Recall:     {results.box.mr:.4f}")

    print("\n⚡ Speed Metrics")
    print("------------------------------")
    print(f"Preprocess:  {preprocess:.2f} ms")
    print(f"Inference:   {inference:.2f} ms")
    print(f"Postprocess: {postprocess:.2f} ms")
    print(f"Total Latency: {total_latency:.2f} ms/image")
    print(f"Throughput:  {fps:.2f} FPS")

    print("\nConfiguration")
    print("------------------------------")
    print(f"Batch size: {args.batch}")
    print(f"FP16:       {args.half}")
    print("==============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLO model and measure speed")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--export-int8", action="store_true", help="Export TensorRT INT8 engine")
    parser.add_argument("--use-engine", action="store_true", help="Use TensorRT engine for inference")

    args = parser.parse_args()

    run_validation(args)


        




