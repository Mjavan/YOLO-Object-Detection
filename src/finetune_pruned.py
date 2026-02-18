import torch
from ultralytics import YOLO
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_PATH.parent 
PRUNED_PATH = PROJECT_ROOT / "pruned_models" / "pruned_model_30.pt"

# Lets uplaod model
model = YOLO(PRUNED_PATH)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.train(
    data="kitti.yaml",
    project= 'fine-tune',
    epochs=15,
    imgsz=832,
    lr0=0.001,
    batch=64,
    mixup=0.0,
    device=device,
    workers=0,
    deterministic=True,
    seed=42
)