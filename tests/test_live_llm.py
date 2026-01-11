import sys
import os
import cv2
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.llm_client import LLMClient
from bot.game_state import GameState
from bot.computer_vision.screenshot import screenshot


def test_live_capture_and_decision():
    print("Initializing LLM Client and Game State...")
    try:
        llm_client = LLMClient()
    except Exception as e:
        print(f"Failed to initialize LLMClient: {e}")
        return

    game_state = GameState()
    
    # Pre-populate state for testing context
    game_state.add_weapon("Holy Wand")
    
    # Define game area (same as main.py)
    game_dimensions = (1245, 768)
    game_area = {"top": 0, "left": 0, "width": game_dimensions[0], "height": game_dimensions[1]}
    
    IMAGE_SIZE = (960, 608)

    print("Capturing screenshot from defined game area...")
    try:
        frame = screenshot(game_area)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame_resized = cv2.resize(frame, IMAGE_SIZE)
        
        # Convert to PIL
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        print(f"Sending to LLM ({llm_client.model_name}) for decision...")
        decision = llm_client.get_decision(pil_image, game_state.to_json())
        
        if decision:
             print("\n--- LLM DECISION ---")
             import json
             print(json.dumps(decision, indent=2))
             print("-----------------------")
        else:
            print("LLM returned No Decision (check API key, logs, or model availability).")
            
    except Exception as e:
        print(f"Error during live test: {e}")

if __name__ == "__main__":
    test_live_capture_and_decision()
