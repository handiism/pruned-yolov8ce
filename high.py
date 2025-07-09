from copy import deepcopy
from ultralytics import YOLO
from ultralytics.nn.modules import Detect,CoordAtt
from ultralytics.utils.torch_utils import initialize_weights
import torch_pruning as tp
from helper import replace_c2f_with_c2f_v2
import torch

DATASET_CONFIG = "/Users/handirachmawan/Documents/Developments/Informatics/yolo-tsr/tsr/CCTSDB2021-YOLO/data.yaml"

model = YOLO("./model/yolov8ce.pt")
model.model.eval()
replace_c2f_with_c2f_v2(model.model)
initialize_weights(model.model)

for name, param in model.model.named_parameters():
    param.requires_grad = True

example_inputs = torch.rand(1, 3, 640, 640)

base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

validation_model = deepcopy(model)
metric = validation_model.val(data=DATASET_CONFIG, split="test", conf=0.5)
init_map = metric.box.map
init_map50 = metric.box.map50

print(
    f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M, mAP={init_map: .5f}"
)

pruning_ratio = 0.2

model.model.train()
for name, param in model.model.named_parameters():
    param.requires_grad = True
ignored_layers = []
unwrapped_parameters = []
for m in model.model.modules():
    if isinstance(m, (Detect,CoordAtt)):
        ignored_layers.append(m)

example_inputs = example_inputs.to(model.device)

pruner = tp.pruner.BasePruner(
    model.model,
    example_inputs,
    importance=tp.importance.LAMPImportance(),
    iterative_steps=1,
    pruning_ratio=pruning_ratio,
    ignored_layers=ignored_layers,
    unwrapped_parameters=unwrapped_parameters,
    round_to=8,
)

tp.utils.print_tool.before_pruning(model.model)

pruner.step()

model.save("real-pruned.pt")

validation_model.model = deepcopy(model.model)

tp.utils.print_tool.after_pruning(model.model)
validation_model.model.to(model.device)
metric = validation_model.val(data=DATASET_CONFIG, split="test", conf=0.5)
pruned_map = metric.box.map
pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(
    validation_model.model, example_inputs.to(validation_model.device)
)
current_speed_up = base_macs / pruned_macs

print(
    f"After pruning: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
    f"mAP={pruned_map}, speed up={current_speed_up},  pruned_param_ratio={pruned_nparams / base_nparams * 100:.3f} %"
)
