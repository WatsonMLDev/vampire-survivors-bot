import cv2
import numpy as np
import os
import glob

# Constants for Paths
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
RESULT_IMAGE_PATH = os.path.join(TEST_DATA_DIR, "vision_result.png")

# --- Configuration ---

# 1. Background (Grass) Color Range
BACKGROUND_GREEN = ((35, 20, 20), (85, 255, 255))

# 2. Enemy Blob Colors (HSV Ranges)
ENEMY_COLOR_RANGES = {
    "bats": [((110, 0, 0), (160, 255, 70))],       
    "skeletons": [((0, 0, 180), (180, 30, 255))]   
}

# 3. Rune Templates (Treating 'rune_blue' as the Universal Shape Template)
RUNE_FILENAMES = ["rune_blue.png"] 

# 4. Detection Thresholds
TEMPLATE_MATCH_THRESHOLD = 0.6 
EDGE_MATCH_THRESHOLD = 0.5     
MIN_BLOB_AREA = 100            
DILATION_ITERATIONS = 5        


def get_spatial_mask(shape, player_pos):
    """
    Creates a static mask to ignore UI and Player Dead-Zone.
    """
    height, width = shape[:2]
    mask = np.ones((height, width), dtype="uint8") * 255

    # 1. UI Masking (Ignore top 10% and bottom 5%)
    mask[0:int(height*0.1), :] = 0
    mask[int(height*0.95):, :] = 0

    # 2. Player Dead-Zone (Ignore self and close weapon effects)
    # Assumes player is always at center of screen
    px, py = player_pos
    pad = 50
    # Ensure bounds
    y1 = max(0, py - pad)
    y2 = min(height, py + pad)
    x1 = max(0, px - pad)
    x2 = min(width, px + pad)
    
    mask[y1:y2, x1:x2] = 0

    return mask

def match_items_robust(frame, templates):
    """
    Finds specific items using Multi-Scale Template Matching.
    Acting as "Universal Item Matcher" by converting to grayscale/edges.
    """
    matches = []
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for label, template in templates.items():
        if template is None: continue
            
        # Convert template to grayscale
        if len(template.shape) == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template
            
        h, w = gray_template.shape[:2]
        found_at_scale = None
        best_scale_score = -1.0
        
        # Wide scale searching to match asset to screen size
        scales = np.linspace(0.3, 1.0, 20) 
        
        for scale in scales:
            resized_w = int(w * scale)
            resized_h = int(h * scale)
            
            if resized_h > gray_frame.shape[0] or resized_w > gray_frame.shape[1] or resized_w < 10 or resized_h < 10:
                continue
                
            resized_tpl = cv2.resize(gray_template, (resized_w, resized_h))
            
            res = cv2.matchTemplate(gray_frame, resized_tpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_scale_score:
                best_scale_score = max_val
                found_at_scale = (scale, max_loc, (max_loc[0] + resized_w, max_loc[1] + resized_h))
        
        print(f"  [{label}] Best Multi-Scale Score: {best_scale_score:.4f} at scale {found_at_scale[0]:.2f}")
        
        if best_scale_score >= TEMPLATE_MATCH_THRESHOLD:
            scale, pt1, pt2 = found_at_scale
            matches.append((pt1, pt2, label))
            
    return matches

def detect_blobs_with_pipeline(frame, item_matches):
    """
    Detects enemy blobs using the full pipeline:
    1. Background Subtraction
    2. Enemy Color Masking
    3. Item Subtraction
    4. Spatial Exclusion (UI/Player)
    """
    height, width = frame.shape[:2]
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # --- Step 1: Background & Enemy Color ---
    lower_green = np.array(BACKGROUND_GREEN[0], dtype="uint8")
    upper_green = np.array(BACKGROUND_GREEN[1], dtype="uint8")
    grass_mask = cv2.inRange(frame_hsv, lower_green, upper_green)
    not_grass_mask = cv2.bitwise_not(grass_mask)

    enemy_mask = np.zeros((height, width), dtype="uint8")
    for enemy_type, ranges in ENEMY_COLOR_RANGES.items():
        for (lower, upper) in ranges:
            mask = cv2.inRange(frame_hsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
            enemy_mask = cv2.bitwise_or(enemy_mask, mask)

    # Combine: Enemy Colors AND Not Grass
    combined_mask = cv2.bitwise_and(enemy_mask, not_grass_mask)

    # --- Step 2: Item Subtraction ---
    for (pt1, pt2, _) in item_matches:
        pad = 5
        cv2.rectangle(combined_mask, 
                      (max(0, pt1[0]-pad), max(0, pt1[1]-pad)), 
                      (min(width, pt2[0]+pad), min(height, pt2[1]+pad)), 
                      0, -1) 

    # --- Step 3: Spatial Exclusion ---
    player_pos = (width // 2, height // 2)
    spatial_mask = get_spatial_mask(frame.shape, player_pos)
    
    final_mask = cv2.bitwise_and(combined_mask, spatial_mask)

    # --- Step 4: Morphology & Contours ---
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(final_mask, kernel, iterations=DILATION_ITERATIONS)
    
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_blobs = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_BLOB_AREA]
    
    return significant_blobs, final_mask

def main():
    print("--- Starting Blob Vision Test (Refined Spatial) ---")
    
    # 1. Load Game Screenshot
    screenshot_path = os.path.join(TEST_DATA_DIR, "game_screenshot.png")
    if not os.path.exists(screenshot_path):
        print(f"[ERROR] Screenshot not found at: {screenshot_path}")
        return

    frame = cv2.imread(screenshot_path)
    if frame is None:
        print(f"[ERROR] Failed to load image: {screenshot_path}")
        return
        
    print(f"Loaded screenshot: {frame.shape}")
    
    # 2. Load Rune Assets (Universal)
    templates = {}
    for fname in RUNE_FILENAMES:
        path = os.path.join(ASSETS_DIR, fname)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            templates[fname] = img
        else:
            print(f"[WARNING] Template not found: {fname}")

    # 3. Match Items FIRST (Universal Grayscale Match)
    item_matches = match_items_robust(frame, templates)
    print(f"Detected {len(item_matches)} items.")

    # 4. Detect Enemy Blobs (Pipeline)
    contours, debug_mask = detect_blobs_with_pipeline(frame, item_matches)
    print(f"Detected {len(contours)} enemy blobs.")

    # 5. Visualization
    output_img = frame.copy()
    
    # Draw Blobs (Red, Filled)
    cv2.drawContours(output_img, contours, -1, (0, 0, 255), 2)
    
    # Draw Item Matches (Green Rectangles)
    for (pt1, pt2, label) in item_matches:
        cv2.rectangle(output_img, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(output_img, "ITEM", (pt1[0], pt1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw Dead Zones (Gray outlines for debug visualization)
    h, w = frame.shape[:2]
    # Top/Bottom UI
    cv2.line(output_img, (0, int(h*0.1)), (w, int(h*0.1)), (100, 100, 100), 1)
    cv2.line(output_img, (0, int(h*0.95)), (w, int(h*0.95)), (100, 100, 100), 1)
    # Player
    cx, cy = w // 2, h // 2
    pad = 50
    cv2.rectangle(output_img, (cx-pad, cy-pad), (cx+pad, cy+pad), (100, 100, 100), 1)

    # Save Result
    cv2.imwrite(RESULT_IMAGE_PATH, output_img)
    print(f"Result saved to: {RESULT_IMAGE_PATH}")

if __name__ == "__main__":
    main()
