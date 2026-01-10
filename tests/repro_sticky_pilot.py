
import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.pilot import Pilot
from bot.computer_vision.object_detection import Detection
from bot.utilities import Point

class TestPilotStickyLogic(unittest.TestCase):
    def test_sticky_target(self):
        screen_center = (480, 304)
        pilot = Pilot(screen_center)
        class_names = {0: "rune"}
        
        # 1. Establish a small cluster in Bin A (Top-Left)
        # Pilot grid is 3 rows, 4 cols.
        # Width 960, Height 608.
        # Bin (0,0) is top left.
        
        # Create 3 runes in top-left
        cluster_a = [
            Detection((10, 10, 30, 30), 0, 1.0),
            Detection((40, 40, 60, 60), 0, 1.0),
            Detection((20, 50, 40, 70), 0, 1.0) # 3rd rune
        ]
        
        print("\nStep 1: Feeding small cluster (3 runes) in top-left...")
        pilot.update(cluster_a, class_names)
        
        info = pilot.get_debug_info()
        print(f"Target Bin: {info['target_bin']}")
        
        self.assertEqual(info['target_bin'], (0, 0), "Should target top-left bin initially")
        
        # 2. Introduce a massive cluster in Bin B (Bottom-Right)
        # Bin (2, 3) is bottom right.
        cluster_b = []
        for i in range(10): # 10 runes!
            cluster_b.append(Detection((800+i, 500+i, 820+i, 520+i), 0, 1.0))
            
        print("\nStep 2: Feeding small cluster (3 runes) AND massive cluster (10 runes) bottom-right...")
        combined = cluster_a + cluster_b
        pilot.update(combined, class_names)
        
        info = pilot.get_debug_info()
        print(f"Target Bin: {info['target_bin']}")
        
        # EXPECTED FAIL: If logic is too sticky, it will stay at (0,0) despite (2,3) having 10 runes.
        if info['target_bin'] == (0, 0):
             print("ISSUE REPRODUCED: Pilot stuck on small cluster (3) ignoring huge cluster (10).")
        else:
             print("Pilot switched to better cluster.")

if __name__ == "__main__":
    unittest.main()
