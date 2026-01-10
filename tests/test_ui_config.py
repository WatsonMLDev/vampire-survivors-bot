
import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.computer_vision.ui_detector import UIDetector
from bot.config import config

class TestUIDetectorConfig(unittest.TestCase):
    def test_load_from_config(self):
        # Ensure config has templates
        templates = config.get("ui_templates")
        self.assertIsNotNone(templates)
        self.assertIn("level_up", templates)
        
        # Initialize Detector
        detector = UIDetector()
        
        # Check if templates loaded (won't be empty if assets exist)
        # We can't guarantee assets exist in this test env, but we can check logic
        print(f"Loaded templates: {detector.templates.keys()}")
        
if __name__ == "__main__":
    unittest.main()
