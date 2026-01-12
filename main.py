import cv2
import torch
import pyautogui
import threading
from typing import List, Tuple
import time
import os

from bot.computer_vision.annotations import AnnotationDrawer
from bot.computer_vision.object_detection import ObjectDetector, Detection
from bot.computer_vision.screenshot import screenshot

from bot.pilot import Pilot
from bot.computer_vision.ui_detector import UIDetector
from bot.llm_client import LLMClient
from bot.game_state import GameState
from bot.input_controller import InputController
from bot.recorder import Recorder
from PIL import Image
import dotenv

dotenv.load_dotenv()

from bot.config import config

KEY_ESC = config.get("keybindings.esc", 27)
KEY_Q = config.get("keybindings.q", 113)
KEY_P = config.get("keybindings.p", 112)
IMAGE_SIZE = tuple(config.get("game.image_size", (960, 608)))


def execute_decision(bot: InputController, decision: dict):
    if not decision:
        return

    action = decision.get("action")
    slot = decision.get("slot", 1)
    
    # Navigation Logic (Start Point: Slot 1 / Top-Left Option)
    
    if action == "select":
        # Slot 1 is current position.
        # Slot 2 is Down 1.
        # Slot 3 is Down 2.
        # Slot 4 is Down 3.
        moves_down = slot - 1
        for _ in range(moves_down):
            bot.press_dpad_down()
        
        bot.press_a()
        
    elif action == "reroll":
        # Reroll is Right 1 from Slot 1
        bot.press_dpad_right()
        bot.press_a()
        
        # After reroll, cursor usually resets to Slot 1 or stays there?
        # Assuming we need to just confirm the reroll. If reroll happens, 
        # the screen refreshes. We are done with this decision cycle.
        
    elif action == "skip":
        # Skip is Right 1, Down 1 from Slot 1
        bot.press_dpad_right()
        bot.press_dpad_down()
        bot.press_a()
        
    elif action == "banish":
        # Banish is Right 1, Down 2 from Slot 1
        bot.press_dpad_right()
        bot.press_dpad_down()
        bot.press_dpad_down()
        bot.press_a() # Enter Banish Mode
        
        # Return to Slot 1 to select target
        # User Logic: Left once, then Up 6 times to ensure top
        bot.press_dpad_left()
        for _ in range(6):
            bot.press_dpad_up()
        
        # Now at Slot 1, navigate to target slot
        moves_down = slot - 1
        for _ in range(moves_down):
            bot.press_dpad_down()
            
        bot.press_a() # Confirm Banish on target slot

    # print(f"Executed: {decision}") # Redundant, logged by llm_client


def main():
    print("Initializing...")
    
    # Initialize variables for cleanup safety
    bot = None
    recorder = None
    debug_writer = None
    
    try:
        try:
            torch.cuda.set_device(0) # Allows PyTorch to use a CUDA GPU for inference.
            print("CUDA device set.")
        except Exception as e:
            print(f"CUDA Error (ignoring if CPU only): {e}")

        print("Initializing ObjectDetector...")
        try:
            drawer = AnnotationDrawer()
            inference_model = ObjectDetector(config.get("paths.model", "model/monster_class.pt"))
        except Exception as e:
            print(f"Failed to load model: {e}")
            return

        print("Initializing UIDetector...")
        ui_detector = UIDetector()
        
        print("Initializing InputController...")

        bot = InputController()
        
        # LLM Integration
        llm_client = LLMClient()
        game_state = GameState()

        # Data Recorder
        recorder = Recorder()
        recorder.start()
        
        # [NEW] Initial Game State Loading (Config Only)
        initial_state_config = config.get("initial_state", {})
        if initial_state_config.get("enabled", False):
            print("Loading initial state from config...")
            for w in initial_state_config.get("weapons", []):
                game_state.add_weapon(w.strip())
                print(f"Added starting weapon: {w.strip()}")
            for p in initial_state_config.get("passives", []):
                game_state.add_passive(p.strip())
                print(f"Added starting passive: {p.strip()}")

        game_dimensions = tuple(config.get("game.dimensions", (1245, 768)))
        game_area = {"top": 0, "left": 0, "width": game_dimensions[0], "height": game_dimensions[1]}

        stop_event = threading.Event()
        pause_event = threading.Event()
        
        pilot = Pilot((IMAGE_SIZE[0]//2, IMAGE_SIZE[1]//2))
        
        # [NEW] Initialize Debug Recorder
        if config.get("debug_recording.enabled", False):
            output_dir = config.get("debug_recording.output_dir", "debug_visualizations")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"debug_viz_{timestamp}.mp4")
            
            fps = config.get("debug_recording.fps", 30)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            debug_writer = cv2.VideoWriter(filename, fourcc, fps, IMAGE_SIZE)
            print(f"Debug recording enabled: {filename}")

        while (key_press := cv2.waitKey(1)) != KEY_ESC:
            try:
                # Capture Raw Frame (for UI Detection)
                # get_frame_from_game resizes, so we inline the capture + convert logic here to keep raw res
                raw_screen = screenshot(game_area)
                frame_raw = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2BGR)

                # [modified] UI State Detection (Uses Raw Frame)
                ui_state = ui_detector.detect_state(frame_raw) # Returns 'LEVEL_UP', 'PAUSE', 'TREASURE_START', 'TREASURE_DONE', or 'GAMEPLAY'

                if ui_state == 'PAUSE':
                    continue # Skip frame
                elif ui_state == 'QUIT':
                    break
                elif ui_state == 'REVIVE':
                    print("Revive Detected! Opening...")
                    time.sleep(1) # Wait for animation start
                    bot.press_a()
                    time.sleep(.5)
                elif ui_state == 'TREASURE_START':
                    print("Treasure Detected! Opening...")
                    time.sleep(1) # Wait for animation start
                    bot.press_a()
                    
                    
                    # Wait for Treasure Done
                    start_wait = time.time()
                    while True:
                        if time.time() - start_wait > config.get("timeouts.treasure_open", 30): # Timeout 15s
                            print("Treasure opening timed out.")
                            break
                            
                        # Capture current raw frame for checking 'Done' state
                        curr_raw = screenshot(game_area)
                        curr_frame_raw = cv2.cvtColor(curr_raw, cv2.COLOR_BGRA2BGR)
                        
                        if ui_detector.detect_state(curr_frame_raw) == 'TREASURE_DONE':
                            print("Treasure Open Complete.")
                            
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
                    continue
                elif ui_state == 'LEVEL_UP':
                    print("Level Up detected! Pausing and consulting LLM...")
                    bot.stop_movement()
                    
                    # Convert frame to PIL for Gemini
                    # OpenCV is BGR, PIL needs RGB
                    frame_rgb = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    decision = llm_client.get_decision(pil_image, game_state.to_json())
                    
                    if decision:
                        # Log decision via GameState
                        # print(f"LLM Decision: {decision}") # Redundant, logged by llm_client
                        
                        time.sleep(3)
                        game_state.log_decision(decision)
                        execute_decision(bot, decision)
                    
                    else:
                        print("LLM failed to decide. Resuming manually (or stuck).")
                    
                    continue
                elif ui_state == 'GAMEPLAY':
                    # --- GAMEPLAY LOGIC ---
                    # Resize frame for Object Detection and Pilot (Model expects IMAGE_SIZE)
                    frame = cv2.resize(frame_raw, IMAGE_SIZE)
                
                    detections, class_names = inference_model.get_detections(frame, 0.6)
                    
                    # [NEW] Filter out detections in the center (Player Self-Detection)
                    # Screen center is approximately IMAGE_SIZE / 2
                    center_x, center_y = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
                    filtered_detections = []
                    for d in detections:
                        x1, y1, x2, y2 = d.position
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        dist_sq = (cx - center_x)**2 + (cy - center_y)**2
                        
                        # Ignorance Radius: 50 pixels (squared = 2500)
                        if dist_sq > config.get("pilot.center_exclusion_radius_sq", 2500): 
                            filtered_detections.append(d)
                    
                    detections = filtered_detections

                    # Update Pilot State and Calculate Force
                    pilot.update(detections, class_names)
                    fx, fy = pilot.get_force_vector(detections, class_names)
                    
                    # Normalize vector to ensure magnitude <= 1.0 (clamped)
                    magnitude = (fx**2 + fy**2)**0.5
                    if magnitude > 1.0:
                        fx /= magnitude
                        fy /= magnitude
                    
                    bot.update_movement(fx, fy)
                    
                    draw_debug_boxes(frame, drawer, detections, class_names)

                    # Draw Force Vector
                    center = pilot.center
                    end_point = (int(center[0] + fx * 50), int(center[1] + fy * 50)) # Scale visualization
                    cv2.arrowedLine(frame, (int(center[0]), int(center[1])), end_point, (0, 255, 0), 3)

                    # Draw Clustering Debug Info
                    debug_info = pilot.get_debug_info()
                    rows = debug_info['rows']
                    cols = debug_info['cols']
                    w = debug_info['width']
                    h = debug_info['height']
                    target = debug_info['target_bin']
                    active_target = debug_info['target_centroid']
                    
                    
                    scale_x = IMAGE_SIZE[0] / w
                    scale_y = IMAGE_SIZE[1] / h
                    
                    if active_target:
                        cv2.circle(frame, (int(active_target[0]), int(active_target[1])), 10, (0, 0, 255), -1)
                        # print(f"Target: {active_target}")

                    cv2.imshow("Model Vision", frame)
                    
                    # [NEW] Debug Recording
                    if debug_writer is not None:
                            debug_writer.write(frame)

                    check_and_update_view_position(key_press, game_area)
                    handle_pause(key_press, pause_event)
                else:
                    print("Unknown UI State: ", ui_state)
            except KeyboardInterrupt:
                print("\n[Main] Interrupted by user.")
                break
            except Exception as e:
                print(f"[Main] Error in loop: {e}")
                bot.stop_movement()
                raise e
    finally:
        # Cleanup
        print("[Main] Cleanup initiated...")
        if bot:
            bot.stop_movement()
        if recorder:
            recorder.stop()
        
        if debug_writer is not None:
            debug_writer.release()
            print("[Main] Debug recording saved.")
            
        cv2.destroyAllWindows()
        print("[Main] Cleanup complete.")
    return 0


def draw_debug_boxes(frame, drawer: AnnotationDrawer,
                     detections: List[Detection], class_names: List[str]):
    """Draws the red and blue boxes depending on the type of entity detected."""
    for detection in detections:
        x1, y1, x2, y2 = detection.position
        label = class_names[detection.label]
        debug_info = f"{label}: {detection.confidence:.2f}"
        color = (0, 0, 255) if label == "monster" else (255, 0, 0) # BGR
        
        drawer.draw_rectangle(frame, color, (x1, y1), (x2, y2))
        drawer.draw_text_with_background(frame, debug_info, (x1, y1))


def get_frame_from_game(bounding_box: Tuple[int, int, int, int]):
    frame = screenshot(bounding_box)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame = cv2.resize(frame, IMAGE_SIZE)
    return frame


def check_and_update_view_position(key_press, game_area):
    """Updates the position at which screenshots will be captured from when the
    letter Q is pressed.
    """
    if key_press == KEY_Q:
        x, y = pyautogui.position()
        game_area["top"] = y
        game_area["left"] = x
        

def handle_pause(key_press, pause: threading.Event):
    """Updates the position at which screenshots will be captured from when the
    letter Q is pressed.
    """    
    if key_press == KEY_P:       
        if pause.is_set():
            pause.clear()
        else:
            pause.set()


if __name__ == "__main__":
    print("Starting main execution...")
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR in main: {e}")
        import traceback
        traceback.print_exc()
