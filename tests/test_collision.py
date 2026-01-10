
import unittest
import sys
import os
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.pilot import Pilot
from bot.computer_vision.object_detection import Detection
from bot.utilities import Point

class TestCollisionAvoidance(unittest.TestCase):
    def test_dont_ram_monster(self):
        center = (480, 304)
        pilot = Pilot(center)
        class_names = {0: "monster", 1: "rune"}
        
        # Scenario:
        # Target (Rune) is directly to the RIGHT (East) at distance 200
        # Monster is directly to the RIGHT (East) at distance 50 (Blocking path)
        
        rune_pos_1 = (center[0] + 190, center[1] - 10, center[0] + 210, center[1] + 10)
        rune_pos_2 = (center[0] + 195, center[1] - 15, center[0] + 215, center[1] + 5)
        rune_pos_3 = (center[0] + 200, center[1], center[0] + 220, center[1] + 20)
        monster_pos = (center[0] + 40, center[1] - 10, center[0] + 60, center[1] + 10) # ~50px away
        
        detections = [
            Detection(rune_pos_1, 1, 1.0),
            Detection(rune_pos_2, 1, 1.0),
            Detection(rune_pos_3, 1, 1.0),
            Detection(monster_pos, 0, 1.0)
        ]
        
        pilot.update(detections, class_names)
        fx, fy = pilot.get_force_vector(detections, class_names)
        
        print(f"Force Vector: ({fx:.2f}, {fy:.2f})")
        
        # If fx is positive, we are moving RIGHT -> INTO THE MONSTER
        # If fx is negative, we are moving LEFT -> AWAY FROM MONSTER (Good)
        
        if fx > 0:
            print("FAIL: Bot is ramming into the monster to get to the rune.")
        else:
            print("PASS: Bot is backing away from the monster.")
            
        self.assertLess(fx, 0, "Bot should be repelled by the close monster, but is moving towards it.")

if __name__ == "__main__":
    unittest.main()
