from ultralytics import YOLO
import os

def evaluate_test_set():
    # 1. Load your best trained model
    model_path = "VampireTraining/yolo26_swarm_detector/weights/best.pt"
    model = YOLO(model_path)

    # 2. Run validation specifically on the 'test' split
    # 'split=test' tells YOLO to use the test/images and test/labels folders 
    # defined in your data.yaml
    metrics = model.val(
        data="synthetic_dataset/data.yaml",
        split='test',        # Force evaluation on the test folder
        imgsz=640,
        batch=16,
        conf=0.25,           # Confidence threshold for evaluation
        iou=0.6,             # NMS IoU threshold
        project="TestEvaluation",
        name="test_metrics_v1",
        plots=True           # Generates PR, F1, and Confusion Matrix plots
    )

    # 3. Print out the key results
    print("\n--- Synthetic Test Results ---")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    
    # Calculate an approximate F1 score if needed manually, 
    # though it's usually in the curves
    f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-9)
    print(f"F1 Score:   {f1:.4f}")

if __name__ == "__main__":
    evaluate_test_set()