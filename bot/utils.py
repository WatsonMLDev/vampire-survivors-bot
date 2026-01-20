import pyautogui
import threading
import math
from typing import Tuple, TypeAlias
from bot.system.config import config

KEY_Q = config.get("keybindings.q", 113)
KEY_P = config.get("keybindings.p", 112)

# --- Type Aliases (from utilities.py) ---
Point: TypeAlias = Tuple[int, int]
Color: TypeAlias = Tuple[int, int, int]
Rect: TypeAlias = Tuple[int, int, int, int]

# --- Geometry Utilities (from utilities.py) ---

def distance_to_point(point_a: Point, point_b: Point) -> float:
    x = abs(int(point_a[0]) - int(point_b[0]))
    y = abs(int(point_a[1]) - int(point_b[1]))    
    return math.sqrt((x*x) + (y*y))


def middle_point(point_a: Point, point_b: Point) -> Point:
    x = (point_a[0] + point_b[0]) / 2
    y = (point_a[1] + point_b[1]) / 2
    return (x, y)


def point_convert_to_int(point: Point) -> Point:
    return int(point[0]), int(point[1])

# --- System/Game Interaction Utilities (from utils.py) ---

def check_and_update_view_position(key_press, game_area):
    """Updates the position at which screenshots will be captured from when the
    letter Q is pressed.
    """
    if key_press == KEY_Q:
        x, y = pyautogui.position()
        game_area["top"] = y
        game_area["left"] = x
        

def handle_pause(key_press, pause: threading.Event):
    """Updates the position at which screenshots will be captured from when the
    letter P is pressed.
    """    
    if key_press == KEY_P:       
        if pause.is_set():
            pause.clear()
        else:
            pause.set()
