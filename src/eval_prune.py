from ultralytics import YOLO
from pathlib import Path



BASE_PATH = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_PATH.parent   # one level up
MODEL_PATH = (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "runs01"
        / "kitti_exp12"
        / "weights"
        / "best.pt"
    )

PRUNED_PATH = PROJECT_ROOT / "pruned_models" / "pruned_model_30.pt"

pruned = YOLO(PRUNED_PATH)

orig = YOLO(MODEL_PATH)

orig_params = sum(p.numel() for p in orig.model.parameters())
pruned_params = sum(p.numel() for p in pruned.model.parameters())

# compute percentage reduction
reduction = 100 * (orig_params - pruned_params) / orig_params

print(f"Original parameters: {orig_params:,}")
print(f"Pruned parameters:   {pruned_params:,}")
print(f"Reduction:           {reduction:.2f}%")
