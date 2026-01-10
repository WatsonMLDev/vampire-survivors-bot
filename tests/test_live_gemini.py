import sys
import os
import cv2
from PIL import Image
import dotenv

dotenv.load_dotenv()

# Add project root to sys.path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.gemini_client import GeminiClient
from bot.game_state import GameState
from bot.computer_vision.screenshot import screenshot
from bot.computer_vision.level_up import LevelUpDetector

def test_live_capture_and_decision():
    print("Initializing Gemini Client and Game State...")
    gemini_client = GeminiClient()
    game_state = GameState()
    
    # Pre-populate state for testing context
    game_state.add_weapon("Holy Wand")
    
    # Define game area (same as main.py)
    # Note: If game window is moved, this might need dynamic adjustment logic from main.py's check_and_update_view_position
    # For this test, we assume standard position or that the user has set it up similar to main default.
    # Default from main.py:
    game_dimensions = (1245, 768)
    game_area = {"top": 0, "left": 0, "width": game_dimensions[0], "height": game_dimensions[1]}
    
    IMAGE_SIZE = (960, 608)

    print("Capturing screenshot from defined game area...")
    try:
        frame = screenshot(game_area)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame_resized = cv2.resize(frame, IMAGE_SIZE)
        
        # Check if it looks like a level up screen (just for info)
        detector = LevelUpDetector()
        is_level_up = detector.is_level_up_screen(frame_resized)
        print(f"Level Up Detector says: {is_level_up}")

        # Convert to PIL
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        print("Sending to Gemini for decision...")
        decision = gemini_client.get_decision(pil_image, game_state.to_json())
        
        if decision:
             print("\n--- GEMINI DECISION ---")
             import json
             print(json.dumps(decision, indent=2))
             print("-----------------------")
        else:
            print("Gemini returned No Decision (check API key or logs).")
            
    except Exception as e:
        print(f"Error during live test: {e}")

if __name__ == "__main__":
    test_live_capture_and_decision()
