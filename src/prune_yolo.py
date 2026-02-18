import torch
from ultralytics import YOLO
import torch_pruning as tp
from pathlib import Path
from ultralytics.nn.modules import Detect

# Count number of parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters())

# 🔹 UPDATE THIS PATH after training finishes
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


# load model: it passes underlying pytorch wrapper 
model = YOLO(MODEL_PATH).model
model.eval()

base_params = count_params(model)

# dummy input (match your training size)
example_inputs = torch.randn(1, 3, 832, 832)

# # 1. Build dependency graph. This requires a dummy input for forwarding
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# pruning amount (start conservative)
PRUNE_RATIO = 0.3

# ---------------------------
# 1️⃣ protect sensitive layers
# ---------------------------
# Define ignored layers (e.g., output layers, small layers)
ignored_layers = []

for m in model.modules():
    if isinstance(m, Detect):
        ignored_layers.append(m)
    elif hasattr(m, "out_channels") and m.out_channels <= 16:
        ignored_layers.append(m)

# ---------------------------
# 2️⃣ importance metric
# ---------------------------
importance = tp.importance.MagnitudeImportance(p=1)

# ---------------------------
# 3️⃣ pruner
# ---------------------------
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=importance,
    pruning_ratio=PRUNE_RATIO,        # start conservative
    iterative_steps=3,        # gradual pruning
    ignored_layers=ignored_layers,
    global_pruning=False
)


# ---------------------------
# 4️⃣ iterative pruning
# ---------------------------
for i in range(3):
    pruner.step()

    pruned_params = count_params(model)

    reduction = 100 * (base_params - pruned_params) / base_params

    print(f"Step {i+1} done")

    print(f"Params: {base_params/1e6:.2f}M → {pruned_params/1e6:.2f}M")
    print(f"Reduction: {reduction:.2f}%\n")
   
# ---------------------------
# 5️⃣ save pruned model
# ---------------------------
# save pruned model
#PRUNED_PATH = PROJECT_ROOT / "pruned_models" / "pruned_model_30.pt"

#torch.save(model, PRUNED_PATH)

#torch.save(model.state_dict(), PRUNED_PATH )

print("✅ Pruning finished.")