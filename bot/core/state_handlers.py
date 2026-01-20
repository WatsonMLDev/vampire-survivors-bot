import time
import cv2
from PIL import Image

from bot.system.config import config
from bot.system.logger import logger
from bot.vision.screenshot import screenshot
from bot.input.input_controller import InputController

# --- Decision Execution Logic ---

def execute_decision(bot: InputController, decision: dict):
    if not decision:
        return

    action = decision.get("action")
    slot = decision.get("slot", 1)
    
    logger.info(f"Executing decision: {action} on slot {slot}")
    
    # Navigation Logic (Start Point: Slot 1 / Top-Left Option)
    
    if action == "select":
        moves_down = slot - 1
        for _ in range(moves_down):
            bot.press_dpad_down()
        bot.press_a()
        
    elif action == "reroll":
        bot.press_dpad_right()
        bot.press_a()
        
    elif action == "skip":
        bot.press_dpad_right()
        bot.press_dpad_down()
        bot.press_a()
        
    elif action == "banish":
        bot.press_dpad_right()
        bot.press_dpad_down()
        bot.press_dpad_down()
        bot.press_a() # Enter Banish Mode
        
        bot.press_dpad_left()
        for _ in range(6):
            bot.press_dpad_up()
        
        moves_down = slot - 1
        for _ in range(moves_down):
            bot.press_dpad_down()
            
        bot.press_a() # Confirm Banish on target slot

# --- State Handlers ---

def handle_revive(bot):
    logger.info("Revive Detected! Opening...")
    time.sleep(1) # Wait for animation start
    bot.press_a()
    time.sleep(.5)

def handle_treasure_opening(bot, ui_detector, game_state, game_area):
    logger.info("Treasure Detected! Opening...")
    time.sleep(1) # Wait for animation start
    bot.press_a()
    
    # Wait for Treasure Done
    start_wait = time.time()
    while True:
        if time.time() - start_wait > config.get("timeouts.treasure_open", 30): # Timeout 15s
            logger.warning("Treasure opening timed out.")
            break
            
        # Capture current raw frame for checking 'Done' state
        curr_raw = screenshot(game_area)
        curr_frame_raw = cv2.cvtColor(curr_raw, cv2.COLOR_BGRA2BGR)
        
        if ui_detector.detect_state(curr_frame_raw) == 'TREASURE_DONE':
            logger.info("Treasure Open Complete.")
            
            # Process Result
            frame_rgb = cv2.cvtColor(curr_frame_raw, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            game_state.update_from_treasure(pil_image)
            
            time.sleep(1)
            # Close Chest
            bot.press_a()
            time.sleep(1) # Wait for close animation
            
            break
        
        time.sleep(0.5)

def handle_level_up(bot, llm_client, game_state, frame_raw):
    logger.info("Level Up detected! Pausing and consulting LLM...")
    bot.stop_movement()
    
    # Convert frame to PIL for Gemini
    # OpenCV is BGR, PIL needs RGB
    frame_rgb = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    decision = llm_client.get_decision(pil_image, game_state.to_json())
    
    if decision:
        time.sleep(3)
        game_state.log_decision(decision)
        execute_decision(bot, decision)
    else:
        logger.error("LLM failed to decide. Resuming manually (or stuck).")
