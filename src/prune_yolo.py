import torch
from ultralytics import YOLO
import torch_pruning as tp
from pathlib import Path

# 🔹 UPDATE THIS PATH after training finishes
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


# load model: it passes underlying pytorch wrapper 
model = YOLO(MODEL_PATH).model
model.eval()

# dummy input (match your training size)
example_inputs = torch.randn(1, 3, 832, 832)

# # 1. Build dependency graph. This requires a dummy input for forwarding
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# pruning amount (start conservative)
PRUNE_RATIO = 0.3

# we set m.out_channels> 8, This prevents pruning very small layers that could break the network.
# create pruner (stable API)
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=tp.importance.MagnitudeImportance(p=1),  # L1 norm
    pruning_ratio=PRUNE_RATIO,
    ignored_layers=[]
)

# execute pruning
pruner.step()

# save pruned model
PRUNED_PATH = PROJECT_ROOT / "pruned_models" / "pruned_model_30.pt"

torch.save(model, PRUNED_PATH)

#torch.save(model.state_dict(), PRUNED_PATH )

print("✅ Pruning finished.")