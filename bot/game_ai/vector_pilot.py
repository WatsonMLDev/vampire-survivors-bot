import math
import numpy as np
from typing import List, Tuple, Optional
from bot.utilities import Point, distance_to_point
from bot.computer_vision.object_detection import Detection

class VectorPilot:
    def __init__(self, screen_center: Point):
        self.center = screen_center
        # Weights
        self.k_attract_target = 150.0 # Strong pull to target cluster (Boosted)
        self.k_attract_rune = 1.0    # Weak pull to individual runes
        self.k_repel_monster = 80.0  # Strong push from monsters
        
        self.repulsion_range = 100 # Pixels (only repel if closer than this)

    def calculate_force(self, detections: List[Detection], class_names: dict, target_cluster: Optional[Point]) -> Tuple[float, float]:
        """
        Calculates the total force vector (fx, fy) acting on the player at self.center.
        """
        fx, fy = 0.0, 0.0
        
        # 1. Attraction: Target Cluster
        if target_cluster:
            dx = target_cluster[0] - self.center[0]
            dy = target_cluster[1] - self.center[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                # Normalize and apply weight
                fx += (dx / dist) * self.k_attract_target
                fy += (dy / dist) * self.k_attract_target
                
        # 2. Repulsion: Monsters
        monsters = [x.position for x in detections if class_names[x.label] == "monster"]
        for monster_rect in monsters:
            # Monster center
            mx = (monster_rect[0] + monster_rect[2]) / 2
            my = (monster_rect[1] + monster_rect[3]) / 2
            
            dx = self.center[0] - mx
            dy = self.center[1] - my
            dist = math.sqrt(dx*dx + dy*dy)
            
            if 0 < dist < self.repulsion_range:
                # Repel force: Inverse square law or linear linear falloff?
                # Linear push-away is simpler and often safer.
                force = self.k_repel_monster * (1 - (dist / self.repulsion_range))
                fx += (dx / dist) * force
                fy += (dy / dist) * force
                
        return fx, fy
        
    def get_input_from_force(self, fx: float, fy: float) -> List[str]:
        keys = []
        threshold = 5.0 # Deadzone
        
        if abs(fx) > threshold:
            if fx > 0: keys.append('d')
            else: keys.append('a')
            
        if abs(fy) > threshold:
            if fy > 0: keys.append('s')
            else: keys.append('w')
            
        return keys
