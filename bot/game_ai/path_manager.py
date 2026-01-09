import time
import vgamepad as vg

class PathManager:
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
    
    def update_movement(self, fx: float, fy: float):
        """
        Updates the virtual controller stick position based on a normalized force vector.
        fx, fy: floats roughly in range [-1.0, 1.0] (clamped to 1.0 magnitude outside if needed, 
                but we clamp to integer range here).
        """
        # NitroGen / vgamepad scaling logic
        # Max integer value for joystick
        MAX_VAL = 32767.0
        
        # Scale to integer range
        x_val = int(fx * MAX_VAL)
        y_val = int(fy * MAX_VAL)
        
        # Apply integer clamping strictly
        x_val = max(-32768, min(32767, x_val))
        y_val = max(-32768, min(32767, y_val))
        
        # NitroGen's specific Windows inversion logic: value = -value - 1
        # This seems to be because Windows Y-axis is inverted relative to standard cartesian or something similar
        # found in NitroGen/nitrogen/game_env.py
        y_val_converted = -y_val - 1
        
        # Clamp again after inversion just to be safe, though math should hold
        y_val_converted = max(-32768, min(32767, y_val_converted))

        self.gamepad.left_joystick(x_value=x_val, y_value=y_val_converted)
        self.gamepad.update()

    def stop_movement(self):
        self.gamepad.reset()
        self.gamepad.update()
