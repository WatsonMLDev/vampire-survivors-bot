
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def segment_gems(target_image_path, sprite_path):
    print(f"Looking for target image at: {os.path.abspath(target_image_path)}")
    print(f"Looking for sprite at: {os.path.abspath(sprite_path)}")

    # Load the main image and the sprite
    # We load the target in color for segmentation and grayscale for matching
    img_rgb = cv2.imread(target_image_path)
    if img_rgb is None:
        print(f"Error: Could not load target image at {target_image_path}")
        return
    
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    template = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        print(f"Error: Could not load sprite at {sprite_path}")
        return

    # If the sprite has an alpha channel, use it for masking
    if len(template.shape) > 2 and template.shape[2] == 4:
        # Separate alpha channel
        template_rgb = template[:, :, :3]
        template_alpha = template[:, :, 3]
        template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
    else:
        template_rgb = template
        if len(template.shape) > 2:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
        template_alpha = None

    w, h = template_gray.shape[::-1]

    w, h = template_gray.shape[::-1]

    # --- Step 1: Multi-Scale Template Matching ---
    # The sprite might be a different scale than the game image. We'll try multiple scales.
    best_scale = 1.0
    best_max_val = -1
    best_res = None
    best_w, best_h = w, h
    
    # Range of scales to check. Visual inspection suggests the sprite is much larger (5-10x) than in-game.
    # Searching from 10% to 50% scale.
    scales = np.linspace(0.1, 0.5, 20)
    
    print("Searching for optimal template scale...")
    for scale in scales:
        resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))
        
        # Skip if resized template is larger than image (unlikely but safe)
        if resized_template.shape[0] > img_gray.shape[0] or resized_template.shape[1] > img_gray.shape[1]:
            continue

        res = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_max_val:
            best_max_val = max_val
            best_scale = scale
            best_res = res
            best_w, best_h = resized_template.shape[::-1]
            
    print(f"Best scale found: {best_scale:.2f} (Max Corr: {best_max_val:.2f})")
    
    # Use the best result
    res = best_res
    w, h = best_w, best_h
    
    # Threshold to find multiple matches
    # Detected projectiles (circles) likely have lower correlation than real gems.
    # Increasing threshold to 0.8 to filter out loose shape matches.
    threshold = 0.78
    
    # If the best match is too weak, we might not have found anything.
    if best_max_val < threshold:
        print(f"Warning: Best match {best_max_val:.2f} is below threshold {threshold}.")
        
    # Debug: Check distribution of top scores to verify separation
    if best_res is not None:
        top_scores = np.sort(best_res.flatten())[-10:][::-1]
        print(f"Top 10 correlation scores: {top_scores}")

    loc = np.where(res >= threshold)

    # --- Step 2: Create a Mask and Segment ---
    # We'll create a mask based on the matching locations
    # Use the alpha channel of the sprite for precise shape masking if available
    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    
    # Prepare the alpha mask at the correct scale
    resized_alpha = None
    if template_alpha is not None:
        resized_alpha = cv2.resize(template_alpha, (best_w, best_h))
        # Binarize alpha for clean masking
        _, resized_alpha = cv2.threshold(resized_alpha, 127, 255, cv2.THRESH_BINARY)
    
    # Track unique matches to avoid overlapping duplicates
    matches = []
    # zip(*loc[::-1]) gives (x, y) points
    pt_list = list(zip(*loc[::-1]))
    
    # Sort matches by correlation score would be ideal, but for now just greedy
    for pt in pt_list:
        # Simple overlap check: if we already have a match nearby, skip
        is_duplicate = False
        for m in matches:
            if abs(pt[0] - m[0]) < w/2 and abs(pt[1] - m[1]) < h/2:
                is_duplicate = True
                break
        
        if not is_duplicate:
            matches.append(pt)
            
            # Draw on the mask
            # If we have an alpha mask, use it to stamp the shape
            if resized_alpha is not None:
                # ROI on the main mask
                y1, y2 = pt[1], pt[1] + best_h
                x1, x2 = pt[0], pt[0] + best_w
                
                # Ensure we fit in the image
                y2 = min(y2, mask.shape[0])
                x2 = min(x2, mask.shape[1])
                w_curr = x2 - x1
                h_curr = y2 - y1
                
                if w_curr > 0 and h_curr > 0:
                    # Stamp the alpha
                    mask_roi = mask[y1:y2, x1:x2]
                    alpha_roi = resized_alpha[:h_curr, :w_curr]
                    mask[y1:y2, x1:x2] = cv2.bitwise_or(mask_roi, alpha_roi)
            else:
                # Fallback to rectangle
                cv2.rectangle(mask, pt, (pt[0] + best_w, pt[1] + best_h), 255, -1)

    # --- Step 3: Result Application ---
    # Combine structural mask (template match) - Color filtering REMOVED to find all gem colors
    final_mask = mask

    # Apply mask to original image to see results
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=final_mask)

    # --- Step 4: Visualization ---
    # Convert for matplotlib (BGR to RGB)
    # Check if we are in a headless environment?
    # User said "try this code out", assuming they might see the window or we save it.
    # Since I'm an agent, I should probably save the result to a file so I can show it/user can see it.
    # But I will keep the plt.show() if they run it locally. 
    # However, I cannot see plt.show(). I will add saving.
    
    output_filename = "gem_segmentation_result.png"
    cv2.imwrite(output_filename, result)
    print(f"Saved result to {output_filename}")

    # plt code kept but might block if untended. 
    # I'll comment out plt.show() to prevent blocking and just save the figure.
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 2, 2)
    plt.title('Sprite Template')
    if len(template.shape) > 2 and template.shape[2] == 4:
        plt.imshow(cv2.cvtColor(template_rgb, cv2.COLOR_BGR2RGB))
    else:
        # handle grayscale or BGR
        if len(template.shape) == 2:
            plt.imshow(template, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
        
    plt.subplot(2, 2, 3)
    plt.title('Final Segmentation Mask')
    plt.imshow(final_mask, cmap='gray')
    
    plt.subplot(2, 2, 4)
    plt.title('Segmented Output')
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.savefig("gem_segmentation_plot.png")
    # plt.show() 

    print(f"Detected {len(matches)} potential gem locations.")

# Run the segmentation
if __name__ == "__main__":
    # Using relative paths for portability, assuming running from project root
    target = 'tests/test_data/image.png'
    sprite = 'assets/rune_blue.png'
    
    # Ensure paths exist
    if not os.path.exists(target):
         # Try absolute path based on known layout if relative fails
         target = r'c:\sudo_desktop_windows\vampire-survivors-bot\tests\test_data\image.png'
    if not os.path.exists(sprite):
         sprite = r'c:\sudo_desktop_windows\vampire-survivors-bot\assets\rune_blue.png'
         
    segment_gems(target, sprite)
