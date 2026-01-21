import cv2
from ultralytics import YOLO
import numpy as np

def run_real_inference():
    # 1. Load your best weights
    model_path = "VampireTraining/yolo26_finetune/weights/best.pt"
    model = YOLO(model_path)

    # 2. Source: Can be a single image, a folder, or 0 for your webcam/live feed
    # For now, put a real game screenshot in your root folder
    source = "results/swarm.png" 

    # 3. Run Inference
    # We use a slightly lower conf (0.20) for real images to catch 
    # sprites that might be partially obscured by damage numbers.
    results = model.predict(
        source=source,
        conf=0.40,      
        iou=0.5,        # Lower IOU helps separate tightly packed swarms
        save=True,      
        imgsz=640,
        show=True       # This will open a window showing the result
    )

    # 4. Process Results
    for r in results:
        boxes = r.boxes
        print(f"Detected {len(boxes)} enemies in the real screenshot.")
        
        #save the image with the boxes
        r.save(filename="results/out/swarm.png")

    print("Inference complete. Check 'results/out' for the saved image.")

if __name__ == "__main__":
    run_real_inference()