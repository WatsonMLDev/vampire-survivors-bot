import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.input_controller import InputController

def test_simple_input():
    print("---------------------------------------------------------")
    print("ISOLATED CONTROLLER TEST")
    print("---------------------------------------------------------")
    print("Initializing controller... (Wait 3s)")
    try:
        bot = InputController()
    except Exception as e:
        print(f"FAILED to initialize controller: {e}")
        return

    print("Controller Ready.")
    print("Please switch focus to the Vampire Survivors game window IMMEDIATELY.")
    print("You have 5 seconds...")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("\nSTARTING MOVEMENT LOOP")
    print("Moving in a circle...")
    print("Press Ctrl+C to stop.")
    print("---------------------------------------------------------")

    import math
    start_time = time.time()

    try:
        while True:
            t = time.time() - start_time
            # Circle math
            fx = math.cos(t * 2)
            fy = math.sin(t * 2)
            
            print(f"Moving: x={fx:.2f}, y={fy:.2f}   ", end='\r')
            bot.update_movement(fx, fy)
            time.sleep(0.016)
            
    except KeyboardInterrupt:
        print("\nStopping controller...")
        bot.stop_movement()
        print("Test stopped.")

if __name__ == "__main__":
    test_simple_input()
