from ultralytics import YOLO
from typing import Tuple, List, Dict
from bot.vision.types import Detection
from bot.system.logger import logger

class ObjectDetector:
    def __init__(self, enemy_model_path: str, gem_model_path: str, 
                 enemy_conf: float = 0.4, enemy_iou: float = 0.5,
                 gem_conf: float = 0.6, gem_iou: float = 0.5,
                 device: str = 'cpu'):
        
        logger.debug(f"Loading Enemy Model: {enemy_model_path} (Conf: {enemy_conf}, IoU: {enemy_iou}, Device: {device})")
        self.enemy_model = YOLO(enemy_model_path)
        self.enemy_model.to(device)
        self.enemy_conf = enemy_conf
        self.enemy_iou = enemy_iou
        
        logger.debug(f"Loading Gem Model: {gem_model_path} (Conf: {gem_conf}, IoU: {gem_iou}, Device: {device})")
        self.gem_model = YOLO(gem_model_path)
        self.gem_model.to(device)
        self.gem_conf = gem_conf
        self.gem_iou = gem_iou
        
        # Define class names mapping
        # 0: monster (from Enemy Model Class 0)
        # 1: rune (from Gem Model Class 3)
        self.class_names = {0: "monster", 1: "rune"}

    def get_detections(self, frame) -> Tuple[List[Detection], Dict[int, str]]:
        detections = []
        
        # --- 1. Enemy Detection ---
        # Enemy Model: Class 0 is 'Enemy'
        enemy_results = self.enemy_model(frame, verbose=False, conf=self.enemy_conf, iou=self.enemy_iou)[0]
        
        if enemy_results.boxes:
             for box in enemy_results.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0: # Only interested in 'Enemy'
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    
                    detections.append(Detection(
                        position=xyxy,
                        label=0, # Bot Label: Monster
                        confidence=conf
                    ))

        # --- 2. Gem Detection ---
        # Gem Model: Class 3 is 'rune'
        gem_results = self.gem_model(frame, verbose=False, conf=self.gem_conf, iou=self.gem_iou)[0]
        
        if gem_results.boxes:
            for box in gem_results.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 3: # Only interested in 'rune'
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    
                    detections.append(Detection(
                        position=xyxy,
                        label=1, # Bot Label: Rune
                        confidence=conf
                    ))

        return detections, self.class_names
