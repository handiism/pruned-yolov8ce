from copy import deepcopy
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.utils.torch_utils import initialize_weights
import torch_pruning as tp
from helper import replace_c2f_with_c2f_v2
import torch

DATASET_CONFIG = "./data.yaml"

# model = YOLO("./model/yolov8n.pt") # this work
model = YOLO("./model/yolov8ce.pt") # my goal is to use this one, but it doesn't work
replace_c2f_with_c2f_v2(model.model)
initialize_weights(model.model)

for name, param in model.model.named_parameters():
    param.requires_grad = True

example_inputs = torch.rand(1, 3, 640, 640)

base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)
print(
    f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M"
)

validation_model = deepcopy(model)
tp.utils.print_tool.before_pruning(model.model)

pruning_ratio = 0.2

model.model.train()
for name, param in model.model.named_parameters():
    param.requires_grad = True
ignored_layers = []
unwrapped_parameters = []
for m in model.model.modules():
    if isinstance(m, (Detect)):
        ignored_layers.append(m)

example_inputs = example_inputs.to(model.device)

DG = tp.DependencyGraph().build_dependency(
    model=model.model,
    example_inputs=example_inputs,
    unwrapped_parameters=unwrapped_parameters,
)

for group in DG.get_all_groups(ignored_layers=ignored_layers):
    print(group)
    group.prune()

validation_model.model = deepcopy(model.model)

pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(
    validation_model.model, example_inputs.to(validation_model.device)
)
print(
    f"After Pruning: MACs={pruned_macs / 1e9: .5f} G, #Params={pruned_nparams / 1e6: .5f} M"
)
