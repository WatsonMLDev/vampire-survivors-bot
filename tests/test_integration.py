from unittest.mock import MagicMock
from PIL import Image
import numpy as np

from bot.gemini_client import GeminiClient
from bot.game_state import GameState
from main import execute_decision

def test_integration():
    print("Starting integration test...")

    # 1. Mock PathManager
    mock_bot = MagicMock()
    mock_bot.press_dpad_down = MagicMock(side_effect=lambda: print("  > D-pad Down pressed"))
    mock_bot.press_a = MagicMock(side_effect=lambda: print("  > A button pressed"))

    # 2. Mock GameState
    game_state = GameState()
    game_state.add_weapon("Whip")
    print(f"Initial State: {game_state.to_json()}")

    # 3. Simulate a Gemini decision (Mocking the API call to avoid cost/delay)
    # We pretend Gemini decided to pick the item in Slot 2
    mock_decision = {
        "action": "select",
        "slot": 2,
        "item_name": "Garlic",
        "item_type": "weapon",
        "reasoning": "Garlic is good for early game AOE."
    }
    print(f"Simulated Decision: {mock_decision}")

    # 4. Log decision
    game_state.log_decision(mock_decision)
    print(f"Updated State: {game_state.to_json()}")
    
    # 5. Execute decision
    print("Executing decision...")
    execute_decision(mock_bot, mock_decision)
    
    # Verification
    print("Verifying inputs...")
    # Slot 2 means 1 down press, then A
    mock_bot.press_dpad_down.assert_called_once()
    mock_bot.press_a.assert_called_once()
    print("SUCCESS: Logic executed correctly.")

if __name__ == "__main__":
    test_integration()
