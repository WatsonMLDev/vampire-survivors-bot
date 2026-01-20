import time
import vgamepad as vg
from bot.system.logger import logger

class InputController:
    def __init__(self):
        logger.info("Creating virtual gamepad... (Drivers initializing)")
        self.gamepad = vg.VX360Gamepad()
        time.sleep(3) # Wait for driver to connect
        logger.info("Virtual gamepad ready.")
    
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

    def press_a(self):
        """Presses and releases the A button."""
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.3)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.3)

    def press_dpad_up(self):
        """Presses and releases D-pad Up."""
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        self.gamepad.update()
        time.sleep(0.3)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        self.gamepad.update()
        time.sleep(0.3)

    def press_dpad_down(self):
        """Presses and releases D-pad Down."""
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        self.gamepad.update()
        time.sleep(0.3)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        self.gamepad.update()
        time.sleep(0.3)

    def press_dpad_left(self):
        """Presses and releases D-pad Left."""
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self.gamepad.update()
        time.sleep(0.3)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self.gamepad.update()
        time.sleep(0.3)

    def press_dpad_right(self):
        """Presses and releases D-pad Right."""
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.update()
        time.sleep(0.3)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.update()
        time.sleep(0.3)
