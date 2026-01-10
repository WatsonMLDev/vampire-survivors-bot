import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.pilot import Pilot
from bot.computer_vision.object_detection import Detection
from bot.utilities import Point

class TestPilot(unittest.TestCase):
    def test_initialization(self):
        screen_center = (480, 304)
        pilot = Pilot(screen_center)
        self.assertIsNotNone(pilot)
        self.assertEqual(pilot.center, screen_center)

    def test_update_no_detections(self):
        pilot = Pilot((480, 304))
        detections = []
        class_names = {}
        
        # Should not crash
        pilot.update(detections, class_names)
        fx, fy = pilot.get_force_vector(detections, class_names)
        
        # Should be some random noise (not exact zero) due to stochastic noise
        # But relatively small
        self.assertTrue(isinstance(fx, float))
        self.assertTrue(isinstance(fy, float))

    def test_update_with_rune(self):
        pilot = Pilot((480, 304))
        # Rune at top left (0,0) -> should pull left/up
        rune_pos = (0, 0, 10, 10)
        detections = [Detection(rune_pos, 0, 1.0)]
        class_names = {0: "rune"}
        
        pilot.update(detections, class_names)
        fx, fy = pilot.get_force_vector(detections, class_names)
        
        # Rune attraction
        self.assertLess(fx, 0) # Pull left
        self.assertLess(fy, 0) # Pull up

if __name__ == "__main__":
    unittest.main()
