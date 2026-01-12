from ultralytics.models.sam import SAM3SemanticPredictor
import os
import time

# Constants
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
# Assuming the user placed sam3.pt in the model directory or root.
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "sam3.pt")
# Using image.png as requested for the FPS test
SCREENSHOT_PATH = os.path.join(TEST_DATA_DIR, "image.png")

def test_sam3_fps():
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found at {MODEL_PATH}. Trying 'sam3.pt' in current directory...")
        model_to_load = "sam3.pt"
    else:
        model_to_load = MODEL_PATH

    print(f"Loading SAM 3 model: {model_to_load}")
    
    # Initialize predictor
    overrides = dict(conf=0.25, task="segment", mode="predict", model=model_to_load, half=True)
    try:
        predictor = SAM3SemanticPredictor(overrides=overrides)
    except Exception as e:
        print(f"[ERROR] Failed to initialize SAM3SemanticPredictor: {e}")
        return

    # Check image
    if not os.path.exists(SCREENSHOT_PATH):
        print(f"[ERROR] Image not found: {SCREENSHOT_PATH}")
        return

    print(f"Target Image: {SCREENSHOT_PATH}")
    
    # Define prompts
    text_prompts = ["enemy"]
    print(f"Prompts: {text_prompts}")

    # Benchmark Configuration
    ITERATIONS = 50
    WARMUP = 5

    print(f"\n--- Starting FPS Benchmark ({ITERATIONS} iterations, {WARMUP} warmup) ---")

    # Warmup
    print("Warming up...")
    for i in range(WARMUP):
        # We assume for a fair "video processing" test, we interpret the image each time
        # although checking predictor.set_image caches features, 
        # for a bot we'd set a NEW image each frame.
        # Since we only have static image, we'll call set_image repeatedly
        # to simulate the overhead of encoding a new frame.
        predictor.set_image(SCREENSHOT_PATH)
        predictor(text=text_prompts)

    print("Benchmarking...")
    start_time = time.time()
    
    for i in range(ITERATIONS):
        predictor.set_image(SCREENSHOT_PATH)
        predictor(text=text_prompts)
        
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / ITERATIONS
    fps = 1.0 / avg_time

    print(f"\nTotal Time: {total_time:.4f}s")
    print(f"Average Time per Frame: {avg_time*1000:.2f}ms")
    print(f"Estimated FPS: {fps:.2f}")

if __name__ == "__main__":
    test_sam3_fps()
