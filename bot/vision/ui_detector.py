import cv2
import numpy as np
import os
from typing import Dict, Tuple, Optional

from bot.system.config import config
from bot.system.logger import logger

class UIDetector:
    def __init__(self, assets_dir: str = "", threshold: float = 0.8):
        self.threshold = threshold
        self.templates: Dict[str, np.ndarray] = {}
        # Allow assets_dir override, but default to config path
        if not assets_dir:
            assets_dir = config.get("paths.assets", "assets")
        self._load_templates(assets_dir)

    def _load_templates(self, assets_dir: str):
        """Loads template images from the assets directory."""
        # Map of State Name -> Filename loaded from config
        template_files = config.get("ui_templates")

        base_path = os.path.abspath(assets_dir)
        
        for name, filename in template_files.items():
            path = os.path.join(base_path, filename)
            if os.path.exists(path):
                # Load unchanged to handle potential alpha
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Convert to BGR if BGRA (drop alpha for simple matching)
                    if img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    
                    self.templates[name] = img
                    logger.debug(f"[UIDetector] Loaded template: {name} from {path}")
                else:
                    logger.warning(f"[UIDetector] Failed to load image: {path}")
            else:
                logger.warning(f"[UIDetector] Template not found: {path}")

    def _match_template(self, frame: np.ndarray, template: np.ndarray) -> Tuple[bool, float]:
        """
        Internal matching logic.
        """
        if frame is None or template is None:
            return False, 0.0

        # Ensure frame is compatible
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= self.threshold:
            return True, max_val
        return False, max_val

    def detect_state(self, frame: np.ndarray) -> str:
        """
        Detects the current game state.
        Returns: 'LEVEL_UP', 'PAUSE', 'TREASURE_START', 'TREASURE_DONE', or 'GAMEPLAY'
        """
        best_match = None
        best_score = 0.0

        for name, template in self.templates.items():
            found, score = self._match_template(frame, template)
            if found and score > best_score:
                best_score = score
                best_match = name
        
        if best_match:
            return best_match.upper()
        
        return "GAMEPLAY"
