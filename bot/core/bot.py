import cv2
import torch
import threading
import time
import os
from PIL import Image

from bot.vision.object_detection import ObjectDetector
from bot.vision.screenshot import screenshot
from bot.core.pilot import Pilot
from bot.vision.ui_detector import UIDetector
from bot.system.llm_client import LLMClient
from bot.core.game_state import GameState
from bot.input.input_controller import InputController
from bot.recording.recorder import Recorder
from bot.system.config import config
from bot.system.logger import logger
from bot.recording.visualizer import Visualizer
from bot.core.state_handlers import handle_revive, handle_treasure_opening, handle_level_up
from bot.core.gameplay_loop import process_gameplay_frame

class VampireSurvivorsBot:
    def __init__(self):
        logger.info("Initializing VampireSurvivorsBot...")
        
        # 1. Device Setup
        self.device = 'cpu'
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(0) 
                self.device = 0
                logger.info("CUDA device set: 0")
            else:
                logger.info("CUDA not available. Using CPU.")
        except Exception as e:
            logger.error(f"CUDA Error (ignoring, falling back to CPU): {e}")

        # 2. Vision Models
        logger.debug("Initializing ObjectDetector...")
        self.inference_model = ObjectDetector(
            enemy_model_path=config.get("paths.enemy_model", "model/enemy.pt"),
            gem_model_path=config.get("paths.gem_model", "model/gem.pt"),
            enemy_conf=config.get("detection.enemy.confidence", 0.4),
            enemy_iou=config.get("detection.enemy.iou", 0.5),
            gem_conf=config.get("detection.gem.confidence", 0.6),
            gem_iou=config.get("detection.gem.iou", 0.5),
            device=self.device
        )
        
        logger.debug("Initializing UIDetector...")
        self.ui_detector = UIDetector()

        # 3. Input & Control
        logger.debug("Initializing InputController...")
        self.input_controller = InputController() # 'bot' in main.py

        # 4. State & AI
        self.llm_client = LLMClient()
        self.game_state = GameState()
        
        self.image_size = tuple(config.get("game.image_size", (960, 608)))
        self.pilot = Pilot((self.image_size[0]//2, self.image_size[1]//2))

        # 5. Recording & Visuals
        self.recorder = Recorder()
        self.visualizer = Visualizer()
        
        # 6. Game Environment
        game_dimensions = tuple(config.get("game.dimensions", (1245, 768)))
        self.game_area = {"top": 0, "left": 0, "width": game_dimensions[0], "height": game_dimensions[1]}
        
        # 7. Control Events
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        # 8. Load Initial State
        self._load_initial_state()
        
        # Constants
        self.KEY_ESC = config.get("keybindings.esc", 27)

    def _load_initial_state(self):
        initial_state_config = config.get("initial_state", {})
        if initial_state_config.get("enabled", False):
            logger.info("Loading initial state from config...")
            for w in initial_state_config.get("weapons", []):
                self.game_state.add_weapon(w.strip())
                logger.debug(f"Added starting weapon: {w.strip()}")
            for p in initial_state_config.get("passives", []):
                self.game_state.add_passive(p.strip())
                logger.debug(f"Added starting passive: {p.strip()}")

    def start(self):
        logger.info("Starting bot services...")
        self.recorder.start()
        self.visualizer.start()

    def stop(self):
        logger.info("Stopping bot services...")
        if self.input_controller:
            self.input_controller.stop_movement()
        if self.recorder:
            self.recorder.stop()
        if self.visualizer:
            self.visualizer.stop()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete.")

    def run(self):
        self.start()
        logger.info("Entering main game loop...")
        
        try:
            while (key_press := cv2.waitKey(1)) != self.KEY_ESC:
                try:
                    # Capture Raw Frame
                    raw_screen = screenshot(self.game_area)
                    frame_raw = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2BGR)

                    # UI Detection
                    ui_state = self.ui_detector.detect_state(frame_raw)
                    
                    if ui_state == 'PAUSE':
                        continue
                    elif ui_state == 'QUIT':
                        break
                    elif ui_state == 'REVIVE':
                        handle_revive(self.input_controller)
                    elif ui_state == 'TREASURE_START':
                        handle_treasure_opening(self.input_controller, self.ui_detector, self.game_state, self.game_area)
                        continue
                    elif ui_state == 'LEVEL_UP':
                        handle_level_up(self.input_controller, self.llm_client, self.game_state, frame_raw)
                        continue
                    elif ui_state == 'GAMEPLAY':
                        process_gameplay_frame(
                            frame_raw, 
                            self.inference_model, 
                            self.pilot, 
                            self.input_controller, 
                            self.visualizer, 
                            self.pause_event, 
                            key_press, 
                            self.game_area
                        )
                    else:
                        logger.warning(f"Unknown UI State: {ui_state}")
                        
                except KeyboardInterrupt:
                    logger.info("Interrupted by user.")
                    break
                except Exception as e:
                    logger.error(f"Error in game loop: {e}")
                    self.input_controller.stop_movement()
                    raise e
        finally:
            self.stop()
