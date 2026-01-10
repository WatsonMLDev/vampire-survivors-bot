import math
import random
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
        self.tick_counter = 0

    def calculate_force(self, detections: List[Detection], class_names: dict, target_cluster: Optional[Point]) -> Tuple[float, float]:
        """
        Calculates the total force vector (fx, fy) acting on the player at self.center.
        """
        fx, fy = 0.0, 0.0
        
        # 1. Attraction: Target Cluster (Main Goal)
        if target_cluster:
            dx = target_cluster[0] - self.center[0]
            dy = target_cluster[1] - self.center[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                fx += (dx / dist) * self.k_attract_target
                fy += (dy / dist) * self.k_attract_target

        # 2. Attraction: Individual Runes (Scavenger Logic)
        # Allows player to pick up stray gems even if they aren't the main cluster
        runes = [x.position for x in detections if class_names[x.label] == "rune"]
        for r_rect in runes:
            rx = (r_rect[0] + r_rect[2]) / 2
            ry = (r_rect[1] + r_rect[3]) / 2
            dx = rx - self.center[0]
            dy = ry - self.center[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Repulsion range is ~100, so we need attraction to work at least that close
            if dist > 0:
                # Stronger pull for nearby gems ( Inverse distance? )
                # k_attract_rune was 1.0, boosting to 10.0 to compete with monsters
                pull = 10.0 
                fx += (dx / dist) * pull
                fy += (dy / dist) * pull
                
        # 3. Repulsion: Monsters
        monsters = [x.position for x in detections if class_names[x.label] == "monster"]
        
        repel_fx, repel_fy = 0.0, 0.0
        
        for monster_rect in monsters:
            mx = (monster_rect[0] + monster_rect[2]) / 2
            my = (monster_rect[1] + monster_rect[3]) / 2
            
            dx = self.center[0] - mx
            dy = self.center[1] - my
            dist = math.sqrt(dx*dx + dy*dy)
            
            if 0 < dist < self.repulsion_range:
                force = self.k_repel_monster * (1 - (dist / self.repulsion_range))
                repel_fx += (dx / dist) * force
                repel_fy += (dy / dist) * force

        # CAP Repulsion (Fear Factor Cap)
        # Prevents 50 weak monsters from making the bot flee at light speed across the map
        # If total repulsion > 150 (approx magnitude of cluster pull), clamp it.
        repel_mag = math.sqrt(repel_fx**2 + repel_fy**2)
        if repel_mag > 150.0:
            scale = 150.0 / repel_mag
            repel_fx *= scale
            repel_fy *= scale
            
        fx += repel_fx
        fy += repel_fy
        
        # 4. Stochastic Noise (Stuck Recovery)
        # Prevents getting stuck in local minima (equilibrium where attraction = repulsion)
        total_mag = math.sqrt(fx**2 + fy**2)
        
        # Base noise (always wobble a bit to seem organic)
        jitter_strength = 5.0
        
        # If stuck (force is near zero), massive kick
        if total_mag < 10.0:
             jitter_strength = 50.0
             
        
        fx += random.uniform(-1, 1) * jitter_strength
        fy += random.uniform(-1, 1) * jitter_strength

        # 5. Periodic Chaos (Obstacle Recovery)
        # Even if we have strong drive, we might be hitting a wall.
        # Every ~150 frames (approx 5s), add a large random impulse to "slide" off geometry.
        self.tick_counter += 1
        if self.tick_counter % 150 == 0:
             chaos_strength = 200.0 # Aggressive kick
             fx += random.uniform(-1, 1) * chaos_strength
             fy += random.uniform(-1, 1) * chaos_strength

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
