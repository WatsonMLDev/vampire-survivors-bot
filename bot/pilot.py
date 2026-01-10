import math
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from bot.utilities import Point
from bot.computer_vision.object_detection import Detection

from bot.config import config

class Pilot:
    def __init__(self, screen_center: Point):
        self.center = screen_center
        
        # Grid settings for Clustering
        self.grid_cols = config.get("pilot.grid.cols", 4)
        self.grid_rows = config.get("pilot.grid.rows", 3)
        self.width = config.get("game.image_size")[0]
        self.height = config.get("game.image_size")[1]
        
        # State
        self.target_cluster_centroid: Optional[Point] = None
        self.target_bin: Optional[Tuple[int, int]] = None
        self.tick_counter = 0

        # Weights
        self.k_attract_target = config.get("pilot.forces.attract_target", 150.0)
        self.k_attract_rune = config.get("pilot.forces.attract_rune", 10.0)
        self.k_repel_monster = config.get("pilot.forces.repel_monster", 500.0)
        self.repulsion_range = config.get("pilot.forces.repulsion_range", 100)
        self.repulsion_cap = config.get("pilot.forces.repulsion_cap", 500.0)

    def update(self, detections: List[Detection], class_names: Dict[int, str]):
        """
        Updates internal state based on new frame detections.
        """
        self._update_target_cluster(detections, class_names)

    def get_force_vector(self, detections: List[Detection], class_names: Dict[int, str]) -> Tuple[float, float]:
        """
        Calculates the force vector for movement.
        """
        fx, fy = 0.0, 0.0
        
        # 1. Attraction: Target Cluster
        if self.target_cluster_centroid:
            dx = self.target_cluster_centroid[0] - self.center[0]
            dy = self.target_cluster_centroid[1] - self.center[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                fx += (dx / dist) * self.k_attract_target
                fy += (dy / dist) * self.k_attract_target

        # 2. Attraction: Individual Runes
        runes = [x.position for x in detections if class_names[x.label] == "rune"]
        for r_rect in runes:
            rx = (r_rect[0] + r_rect[2]) / 2
            ry = (r_rect[1] + r_rect[3]) / 2
            dx = rx - self.center[0]
            dy = ry - self.center[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                fx += (dx / dist) * self.k_attract_rune
                fy += (dy / dist) * self.k_attract_rune
                
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

        # Cap Repulsion
        repel_mag = math.sqrt(repel_fx**2 + repel_fy**2)
        if repel_mag > self.repulsion_cap:
            scale = self.repulsion_cap / repel_mag
            repel_fx *= scale
            repel_fy *= scale
            
        fx += repel_fx
        fy += repel_fy
        
        #  # 4. Stochastic Noise (Stuck Recovery)
        # total_mag = math.sqrt(fx**2 + fy**2)
        # jitter_strength = 5.0
        # if total_mag < 10.0:
        #      jitter_strength = 50.0 # Kick if stuck
        
        # fx += random.uniform(-1, 1) * jitter_strength
        # fy += random.uniform(-1, 1) * jitter_strength

        # # 5. Periodic Chaos
        # self.tick_counter += 1
        # if self.tick_counter % 150 == 0:
        #      chaos_strength = 200.0
        #      fx += random.uniform(-1, 1) * chaos_strength
        #      fy += random.uniform(-1, 1) * chaos_strength

        return fx, fy

    def _update_target_cluster(self, detections: List[Detection], class_names: Dict[int, str]):
        """
        Internal logic to determine the "Best" cluster of gems.
        Migrated from PositionEvaluator.
        """
        runes = [x.position for x in detections if class_names[x.label] == "rune"]
        
        if not runes:
            # Keep previous target if possible? Or reset?
            # If no runes visible, we probably shouldn't blindly chase old target.
            # But let's be safe and clear checks.
            # Usually we might want to "remember" for a few frames, but for simplicity:
            pass 
        
        # Binning
        bins = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        for rune in runes:
            # center of rune
            rx = (rune[0] + rune[2]) / 2
            ry = (rune[1] + rune[3]) / 2
            c = min(int(rx / (self.width / self.grid_cols)), self.grid_cols - 1)
            r = min(int(ry / (self.height / self.grid_rows)), self.grid_rows - 1)
            bins[r, c] += 1
            
        max_runes = np.max(bins)
        
        target_r, target_c = None, None
        use_previous = False
        
        # Sticky Logic: Check if previous bin is still good
        if self.target_bin:
             p_r, p_c = self.target_bin
             current_target_count = bins[p_r, p_c]
             
             min_runes = config.get("pilot.sticky_target.min_runes", 2)
             multiplier = config.get("pilot.sticky_target.better_cluster_multiplier", 1.5)
             
             if current_target_count > min_runes:
                 if max_runes > current_target_count * multiplier:
                     use_previous = False # Switch to better
                 else:
                     target_r, target_c = p_r, p_c
                     use_previous = True
        
        # If not sticking, find new max
        if not use_previous and max_runes > 2:
            target_r, target_c = np.unravel_index(np.argmax(bins), bins.shape)
            
        if target_r is not None:
             self.target_bin = (target_r, target_c)
             
             # Calculate Centroid of that specific bin
             cluster_points = []
             for rune in runes:
                rx = (rune[0] + rune[2]) / 2
                ry = (rune[1] + rune[3]) / 2
                c = min(int(rx / (self.width / self.grid_cols)), self.grid_cols - 1)
                r = min(int(ry / (self.height / self.grid_rows)), self.grid_rows - 1)
                
                if (r, c) == (target_r, target_c):
                    cluster_points.append((rx, ry))
            
             if cluster_points:
                 cx = np.mean([p[0] for p in cluster_points])
                 cy = np.mean([p[1] for p in cluster_points])
                 self.target_cluster_centroid = (cx, cy)
        else:
            self.target_bin = None
            self.target_cluster_centroid = None

    def get_debug_info(self):
        return {
            "rows": self.grid_rows,
            "cols": self.grid_cols,
            "width": self.width,
            "height": self.height,
            "target_bin": self.target_bin,
            "target_centroid": self.target_cluster_centroid
        }
