import cv2
import numpy as np
from typing import Optional

class LevelUpDetector:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def is_level_up_screen(self, frame: np.ndarray) -> bool:
        """
        Checks if the 'Level Up!' screen is present in the current frame 
        using color signature.
        """
        if frame is None:
            return False
            
        return self.check_color_signature(frame)
        
    def check_color_signature(self, frame: np.ndarray) -> bool:
        """
        Checks for massive solid background colors:
        1. Level Up Screen: Slate Blue (116, 79, 75)
        2. Title Screen: Bright Blue (205, 64, 39)
        """
        # 1. Level Up Check
        lower_lvl = np.array([100, 64, 60])
        upper_lvl = np.array([132, 94, 90])
        
        mask_lvl = cv2.inRange(frame, lower_lvl, upper_lvl)
        if cv2.countNonZero(mask_lvl) > 15000:
            return True

        # 2. Title Screen Check
        lower_title = np.array([190, 50, 25])
        upper_title = np.array([220, 80, 55])
        
        mask_title = cv2.inRange(frame, lower_title, upper_title)
        if cv2.countNonZero(mask_title) > 15000:
            return True
            
        return False
