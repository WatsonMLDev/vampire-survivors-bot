import cv2
import torch
import pyautogui
import threading
from typing import List, Tuple

from bot.computer_vision.annotations import AnnotationDrawer
from bot.computer_vision.object_detection import ObjectDetector, Detection
from bot.computer_vision.screenshot import screenshot

from bot.game_ai.path_manager import PathManager, edge_list_to_direction_list
from bot.game_ai.position_evaluator import PositionEvaluator
from bot.game_ai.vector_pilot import VectorPilot
from bot.computer_vision.level_up import LevelUpDetector


KEY_ESC = 27
KEY_Q = 113
KEY_P = 112
IMAGE_SIZE = (960, 608)


def main():
    torch.cuda.set_device(0) # Allows PyTorch to use a CUDA GPU for inference.
    drawer = AnnotationDrawer()
    inference_model = ObjectDetector("model/monster_class.pt")
    level_up_detector = LevelUpDetector()
    bot = PathManager()
    
    game_dimensions = (1245, 768)
    game_area = {"top": 0, "left": 0, "width": game_dimensions[0], "height": game_dimensions[1]}

    stop_event = threading.Event()
    pause_event = threading.Event()
    bot_thread = threading.Thread(target=bot.follow_pathing_queue, args=[stop_event, pause_event])
    bot_thread.start()
    
    active_target = None
    pilot = VectorPilot((IMAGE_SIZE[0]//2, IMAGE_SIZE[1]//2))
    
    while (key_press := cv2.waitKey(1)) != KEY_ESC:
        try:
            frame = get_frame_from_game(game_area)

            if level_up_detector.is_level_up_screen(frame):
                print("Level Up detected! Pausing...")
                bot.stop_movement()
                continue
            
            detections, class_names = inference_model.get_detections(frame, 0.6)
            evaluator = PositionEvaluator(detections, class_names, 1, active_target)
            
            # Update target for next frame
            if evaluator.cluster_centroid:
                 active_target = evaluator.cluster_centroid
            else:
                 active_target = None

            # Vector Pilot Logic
            fx, fy = pilot.calculate_force(detections, class_names, active_target)
            keys = pilot.get_input_from_force(fx, fy)
            bot.add_to_pathing_queue(keys)
            
            draw_debug_boxes(frame, drawer, detections, class_names)

            # Draw Force Vector
            center = pilot.center
            end_point = (int(center[0] + fx * 2), int(center[1] + fy * 2)) # Scale visualization
            cv2.arrowedLine(frame, (int(center[0]), int(center[1])), end_point, (0, 255, 0), 3)

            # Draw Clustering Debug Info
            cluster_info = evaluator.get_clustering_info()
            rows = cluster_info['rows']
            cols = cluster_info['cols']
            w = cluster_info['width']
            h = cluster_info['height']
            target = cluster_info['target_bin']
            
            
            scale_x = IMAGE_SIZE[0] / w
            scale_y = IMAGE_SIZE[1] / h
            
            for r in range(rows):
                for c in range(cols):
                    x1 = int(c * (w / cols) * scale_x)
                    y1 = int(r * (h / rows) * scale_y)
                    x2 = int((c+1) * (w / cols) * scale_x)
                    y2 = int((r+1) * (h / rows) * scale_y)
                    
                    color = (255, 255, 255) # White grid
                    thickness = 1
                    
                    if target and target == (r, c):
                        color = (0, 255, 0) # Green for target
                        thickness = 3
                        
                    drawer.draw_rectangle(frame, color, (x1, y1), (x2, y2))
            
            if active_target:
                # Scale active_target from width/height (960x608) to display? 
                # active_target is in 960x608 space already.
                cv2.circle(frame, (int(active_target[0]), int(active_target[1])), 10, (0, 0, 255), -1)
                print(f"Target: {active_target}")

            cv2.imshow("Model Vision", frame)
            check_and_update_view_position(key_press, game_area)
            handle_pause(key_press, pause_event)
        except Exception:
            stop_event.set()
            raise Exception

    stop_event.set()
    cv2.destroyAllWindows()
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
    main()
