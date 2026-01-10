import cv2
import numpy as np
import os
import sys
import time

# Ensure we can import from bot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bot.computer_vision.screenshot import screenshot

def match_template(frame, template, threshold=0.8):
    """
    Checks if a template exists in the frame.
    Returns (found: bool, max_val: float, location: tuple)
    """
    if frame is None or template is None:
        return False, 0.0, (0, 0)

    # Convert to grayscale for template matching
    # Ensure connections are roughly compatible (dropping alpha if present)
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if template.shape[2] == 4:
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        return True, max_val, max_loc
    return False, max_val, max_loc

def detect_game_state(frame, templates):
    """
    Detects the current game state based on provided templates.
    Checks all provided templates and returns the one with the highest match (if > threshold).
    """
    best_match = None
    best_score = 0.0

    print("Checking templates...")
    for name, template in templates.items():
        found, score, _ = match_template(frame, template)
        print(f"  - {name}: {score:.4f} {'(MATCH)' if found else ''}")
        
        if found and score > best_score:
            best_score = score
            best_match = name

    if best_match:
        return best_match.upper()

    return "GAMEPLAY"

def main():
    print("--- UI Detection Test (Live) ---")
    
    pause_image = os.path.join(os.path.dirname(__file__), '..', 'assets', 'pause.png')
    levelup_image = os.path.join(os.path.dirname(__file__), '..', 'assets', 'level_up.png')
    treasure_start_image = os.path.join(os.path.dirname(__file__), '..', 'assets', 'treasure_start.png')
    treasure_done_image = os.path.join(os.path.dirname(__file__), '..', 'assets', 'treasure_done.png')
    
    templates = {'pause': cv2.imread(pause_image, cv2.IMREAD_UNCHANGED), 'levelup': cv2.imread(levelup_image, cv2.IMREAD_UNCHANGED), 'treasure_start': cv2.imread(treasure_start_image, cv2.IMREAD_UNCHANGED), 'treasure_done': cv2.imread(treasure_done_image, cv2.IMREAD_UNCHANGED)}

    frame = screenshot((0, 18, 1280, 740))
    cv2.imwrite("pause.png", frame)


    result = detect_game_state(frame, templates)
    print(f"\nFinal Result: {result}")

if __name__ == "__main__":
    main()
