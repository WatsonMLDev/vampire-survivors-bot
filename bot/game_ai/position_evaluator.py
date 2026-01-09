from typing import List, Dict
from random import sample
import numpy as np

from bot.computer_vision.object_detection import Detection
from bot.utilities import Point, distance_to_point


MAGNET_RANGE = 40
RUNE_VALUE = 0.3
MAX_RISK_DISTANCE = 25


class PositionEvaluator():
    def __init__(self, detections: List[Detection], class_names: Dict[int, str], sampling_rate: float, 
                 previous_target: Point = None):
        monsters = [(np.mean([x.position[0], x.position[2]]), np.mean([x.position[1], x.position[3]]))
                    for x in detections
                    if class_names[x.label] == "monster"]
        runes = [x.position for x in detections if class_names[x.label] == "rune"]
        
        self.monsters = sample(monsters, int(sampling_rate * len(monsters)))
        self.runes = sample(runes, int(sampling_rate * len(runes)))

        # Density Gain: Cluster targeting (Initialize here)
        self.grid_cols = 4
        self.grid_rows = 3
        self.width = 960
        self.height = 608
        
        bins = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        
        for rune in self.runes:
            c = min(int(rune[0] / (self.width / self.grid_cols)), self.grid_cols - 1)
            r = min(int(rune[1] / (self.height / self.grid_rows)), self.grid_rows - 1)
            bins[r, c] += 1

        self.target_bin = None
        self.cluster_centroid = None
        max_runes = np.max(bins)
        
        # Sticky Logic
        target_r, target_c = None, None
        use_previous = False
        
        if previous_target:
             # Check if previous target (centroid) still has significant runes
             p_c = min(int(previous_target[0] / (self.width / self.grid_cols)), self.grid_cols - 1)
             p_r = min(int(previous_target[1] / (self.height / self.grid_rows)), self.grid_rows - 1)
             
             # If the bin of the previous target still has > 2 runes, keep it
             if bins[p_r, p_c] > 2:
                 target_r, target_c = p_r, p_c
                 use_previous = True

        if not use_previous and max_runes > 2:
            target_r, target_c = np.unravel_index(np.argmax(bins), bins.shape)

        if target_r is not None:
            self.target_bin = (target_r, target_c)
            
            # Re-find items in this specific bin for centroid
            cluster_runes = [rune for rune in self.runes 
                             if min(int(rune[0] / (self.width / self.grid_cols)), self.grid_cols - 1) == target_c
                             and min(int(rune[1] / (self.height / self.grid_rows)), self.grid_rows - 1) == target_r]

            if cluster_runes:
                centroid_x = np.mean([p[0] for p in cluster_runes])
                centroid_y = np.mean([p[1] for p in cluster_runes])
                self.cluster_centroid = (centroid_x, centroid_y)

    def value(self, position: Point) -> float:
        value = 0.5 # base value
        value += self.__gain(position) - self.__risk(position)
        return np.clip(value, 0, 1)
    
    def __gain(self, position: Point):
        if len(self.runes) == 0:
            return 0
        
        runes_collected = [1 - min(0.01*distance_to_point(position, rune), 1)
                           for rune in self.runes]
        base_gain = sum(runes_collected) * RUNE_VALUE

        if self.cluster_centroid:
             dist_to_cluster = distance_to_point(position, self.cluster_centroid)
             # Fix: reduced falloff so it pulls from across the screen (range ~1250px)
             # Boosted to 1000.0 to ensure it breaks local optima (ignoring nearby crumbs for the main feast)
             cluster_bonus = 1000.0 * (1 - min(0.0008 * dist_to_cluster, 1))
             return base_gain + cluster_bonus
        
        return base_gain

    def get_clustering_info(self):
        return {
            "rows": self.grid_rows,
            "cols": self.grid_cols,
            "width": self.width,
            "height": self.height,
            "target_bin": self.target_bin
        }

    #TODO: experiment more with gain and risk functions
    def __risk(self, position: Point):
        if len(self.monsters) == 0:
            return 0
        
        closest = np.min([distance_to_point(position, monster)
                          for monster in self.monsters])
        if closest == 0:
            return 99
        
        return MAX_RISK_DISTANCE / closest
