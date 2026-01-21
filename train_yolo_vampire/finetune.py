import os
from clearml import Task
from ultralytics import YOLO

def train_vampire_detector():
    # 1. LOAD YOUR SYNTHETIC WEIGHTS (Not the generic yolo26n.pt)
    # We want the model to start with the 'vocabulary' it learned from your synthetic data.
    # Replace 'best.pt' with the actual path to your synthetic model's best weights.
    model = YOLO("VampireTraining/yolo26_swarm_detector/weights/best.pt") 

    # 2. Set up the Fine-Tuning parameters
    results = model.train(
        data="dataset/synthetic_dataset_real/data.yaml",
        epochs=50,             # Fine-tuning needs fewer epochs (30-50 is usually plenty)
        imgsz=640,
        batch=16,              # Slightly lower batch can help with stability
        device=0,
        workers=4,
        project="VampireTraining", 
        name="yolo26_finetune",
        exist_ok=True,
        
        # --- THE FINE-TUNING SPECIAL SAUCE ---
        optimizer="AdamW",     
        lr0=0.0005,            # MUCH LOWER learning rate (don't overwrite the synthetic brain)
        freeze=10,             # FREEZE the first 10 layers (Backbone) to protect general features
        
        # --- DOMAIN ADAPTATION AUGMENTATIONS ---
        augment=True,
        mosaic=0.0,            # TURN OFF MOSAIC: Prevents the model from getting confused by 
                               # multiple "real" frames stitched together.
        hsv_h=0.015,           # Color jittering to handle different map tints
        hsv_v=0.4,             # Brightness jittering
        fliplr=0.5,            # Standard flips
    )

    # 3. Validate (The mAP here will now reflect real gameplay performance!)
    metrics = model.val()
    print(f"Real-World mAP50-95: {metrics.box.map}")

    # 4. Export
    model.export(format="onnx", imgsz=640)
    print("Fine-tuned model exported to ONNX.")

if __name__ == "__main__":
    train_vampire_detector()