import time
import math
from bot.game_ai.path_manager import PathManager

def verify_controller():
    print("Initializing PathManager (creating virtual controller)...")
    bot = PathManager()
    
    print("Starting circle test...")
    print("Open 'HTML5 Gamepad Tester' or 'Joy.cpl' to visualize inputs.")
    
    try:
        start_time = time.time()
        while True:
            t = time.time() - start_time
            # Spin in a circle
            fx = math.cos(t * 2) 
            fy = math.sin(t * 2)
            
            bot.update_movement(fx, fy)
            print(f"Input: fx={fx:.2f}, fy={fy:.2f}    ", end='\r')
            time.sleep(0.016) # ~60 FPS

    except KeyboardInterrupt:
        print("\nStopping controller...")
        bot.stop_movement()
        print("Done.")

if __name__ == "__main__":
    verify_controller()
