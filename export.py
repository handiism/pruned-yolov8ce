from ultralytics import YOLO

# Configuration
DATASET_CONFIG = "./data.yaml"

# Load pre-trained YOLOv8 model
model = YOLO("./real-pruned.pt")
model.val(data=DATASET_CONFIG, split="test", conf=0.5)

model.export(format="onnx", imgsz=(320, 320))
