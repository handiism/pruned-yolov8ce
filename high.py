"""
High-level YOLOv8 model pruning script using torch-pruning library.

This script demonstrates how to prune a YOLOv8 model to reduce computational complexity
while maintaining reasonable accuracy. It uses the LAMP (Layer-Adaptive Magnitude-based Pruning)
importance metric for structured pruning.

Key features:
- Loads a pre-trained YOLOv8 model
- Replaces C2f modules with optimized C2f_v2 modules
- Performs structured pruning using torch-pruning
- Evaluates model performance before and after pruning
- Calculates speedup and parameter reduction metrics
"""

from copy import deepcopy
from ultralytics import YOLO
from ultralytics.nn.modules import Detect, CoordAtt
from ultralytics.utils.torch_utils import initialize_weights
import torch_pruning as tp
from helper import replace_c2f_with_c2f_v2
import torch

# Configuration
DATASET_CONFIG = "./data.yaml"

# Load pre-trained YOLOv8 model
model = YOLO("./model/realca2.pt")
model.model.eval()  # Set model to evaluation mode

# Replace C2f modules with optimized C2f_v2 modules for better performance
replace_c2f_with_c2f_v2(model.model)

# Initialize weights for any new modules that were added
initialize_weights(model.model)

# Enable gradient computation for all parameters (required for pruning)
for name, param in model.model.named_parameters():
    param.requires_grad = True

# Create example input tensor for MACs and parameter counting
# Shape: (batch_size, channels, height, width) - standard YOLO input size
example_inputs = torch.rand(1, 3, 640, 640)

# Calculate baseline model complexity metrics
base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

# Create a copy of the model for validation to avoid affecting the original
validation_model = deepcopy(model)

# Evaluate baseline model performance on test set
metric = validation_model.val(data=DATASET_CONFIG, split="test", conf=0.5)
init_map = metric.box.map      # Mean Average Precision
init_map50 = metric.box.map50  # Mean Average Precision at IoU=0.5

print(
    f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M, mAP={init_map: .5f}"
)

# Pruning configuration
pruning_ratio = 0.1  # Remove 10% of parameters

# Set model to training mode for pruning (required by torch-pruning)
model.model.train()

# Ensure all parameters are trainable
for name, param in model.model.named_parameters():
    param.requires_grad = True

# Define layers to ignore during pruning
# These layers are critical for model functionality and should not be pruned
ignored_layers = []
unwrapped_parameters = []

# Identify and ignore detection heads and attention modules
for m in model.model.modules():
    if isinstance(m, (Detect, CoordAtt)):
        ignored_layers.append(m)

# Move example inputs to the same device as the model
example_inputs = example_inputs.to(model.device)

# Initialize the pruner with LAMP importance metric
# LAMP (Layer-Adaptive Magnitude-based Pruning) is effective for structured pruning
pruner = tp.pruner.BasePruner(
    model.model,
    example_inputs,
    importance=tp.importance.LAMPImportance(),  # Layer-adaptive importance metric
    iterative_steps=1,                          # Single pruning step
    pruning_ratio=pruning_ratio,                # 10% pruning ratio
    ignored_layers=ignored_layers,              # Layers to preserve
    unwrapped_parameters=unwrapped_parameters,  # Parameters to preserve
    round_to=8,                                 # Round channel numbers to multiples of 8
)

# Print model structure before pruning
tp.utils.print_tool.before_pruning(model.model)

# Execute the pruning step
pruner.step()

# Save the pruned model
model.save("real-pruned.pt")

# Create a copy of the pruned model for evaluation
validation_model.model = deepcopy(model.model)

# Print model structure after pruning to see the changes
tp.utils.print_tool.after_pruning(model.model)

# Move validation model to the correct device
validation_model.model.to(model.device)

# Evaluate pruned model performance
# Note: This may throw an error if using yolov8ce.pt due to compatibility issues
metric = validation_model.val(data=DATASET_CONFIG, split="test", conf=0.5)
pruned_map = metric.box.map

# Calculate pruned model complexity metrics
pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(
    validation_model.model, example_inputs.to(validation_model.device)
)

# Calculate speedup factor
current_speed_up = base_macs / pruned_macs

# Print comprehensive results comparing before and after pruning
print(
    f"After pruning: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
    f"mAP={pruned_map}, speed up={current_speed_up},  pruned_param_ratio={pruned_nparams / base_nparams * 100:.3f} %"
)
