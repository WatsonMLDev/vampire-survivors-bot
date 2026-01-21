import os
from clearml import Task
from ultralytics import YOLO

def train_vampire_detector():
    # 1. Initialize the YOLO26 model
    # 'yolo26n.pt' refers to the Nano version - good for high FPS real-time detection
    # Use 'yolo26s.pt' or 'yolo26m.pt' if you want higher accuracy at the cost of speed
    model = YOLO("yolo26n.pt") 

    # 2. Set up the training parameters
    # Pointing to the data.yaml generated in the previous step
    results = model.train(
        data="synthetic_dataset/data.yaml",
        epochs=100,            # Adjust based on when loss plateaus
        imgsz=640,             # Matches your synthetic image size
        batch=18,              # Adjust based on your GPU VRAM
        device=0,              # Use 0 for NVIDIA GPU, 'cpu' for CPU, or 'mps' for Apple Silicon
        workers=4,             # Number of dataloader workers
        project="VampireTraining", 
        name="yolo26_swarm_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",     # Often performs better for varied synthetic data
        lr0=0.01,              # Initial learning rate
        augment=True          # Uses Ultralytics' internal mosaics/flips
        # cache=False,
    )

    # 3. Validate the model
    # This evaluates the model on the 'val' set defined in data.yaml
    metrics = model.val()
    print(f"Mean Average Precision (mAP50-95): {metrics.box.map}")

    # 4. Export the model
    # Export to .onnx or .engine (TensorRT) for maximum performance in an overlay
    model.export(format="onnx", imgsz=640)
    print("Model exported to ONNX format.")

if __name__ == "__main__":
    train_vampire_detector()