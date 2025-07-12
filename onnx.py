from ultralytics import YOLO

# Configuration
DATASET_CONFIG = "./data.yaml"

# Load pre-trained YOLOv8 model
model = YOLO("./real-pruned.onnx")
model.val(data=DATASET_CONFIG, split="test", conf=0.5, imgsz=(320, 320))

