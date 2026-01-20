import os
import time
import json
import cv2
import multiprocessing
import pygame
import psutil
import win32gui
import win32process
import dxcam
import dxcam
import warnings
from bot.system.config import config
from bot.system.logger import logger, setup_logger

def _normalize_trigger(value):
    val = max(-1.0, min(1.0, value))
    return int((val + 1.0) / 2.0 * 255.0)

def _normalize_stick(value):
    val = max(-1.0, min(1.0, value))
    return int(val * 32767.0)

def _get_input_state(joystick, step_index):
    pygame.event.pump() 
    
    state = {
        "step": step_index,
        "SOUTH": joystick.get_button(0),
        "EAST": joystick.get_button(1),
        "WEST": joystick.get_button(2),
        "NORTH": joystick.get_button(3),
        "LEFT_SHOULDER": joystick.get_button(4),
        "RIGHT_SHOULDER": joystick.get_button(5),
        "BACK": joystick.get_button(6),
        "START": joystick.get_button(7),
        "LEFT_THUMB": joystick.get_button(8),
        "RIGHT_THUMB": joystick.get_button(9),
        "GUIDE": 0, 
        "DPAD_UP": 0, "DPAD_DOWN": 0, "DPAD_LEFT": 0, "DPAD_RIGHT": 0,
        "LEFT_TRIGGER": 0, "RIGHT_TRIGGER": 0,
        "AXIS_LEFTX": 0, "AXIS_LEFTY": 0, "AXIS_RIGHTX": 0, "AXIS_RIGHTY": 0
    }

    if joystick.get_numhats() > 0:
        hat_x, hat_y = joystick.get_hat(0)
        state["DPAD_LEFT"] = 1 if hat_x == -1 else 0
        state["DPAD_RIGHT"] = 1 if hat_x == 1 else 0
        state["DPAD_UP"] = 1 if hat_y == 1 else 0
        state["DPAD_DOWN"] = 1 if hat_y == -1 else 0
    
    # Axis mapping
    state["AXIS_LEFTX"] = [_normalize_stick(joystick.get_axis(0))]
    state["AXIS_LEFTY"] = [_normalize_stick(joystick.get_axis(1))]
    
    if joystick.get_numaxes() >= 4:
            state["AXIS_RIGHTX"] = [_normalize_stick(joystick.get_axis(2))]
            state["AXIS_RIGHTY"] = [_normalize_stick(joystick.get_axis(3))]
    else:
            state["AXIS_RIGHTX"] = [0]
            state["AXIS_RIGHTY"] = [0]
    
    if joystick.get_numaxes() >= 6:
        state["LEFT_TRIGGER"] = [_normalize_trigger(joystick.get_axis(4))]
        state["RIGHT_TRIGGER"] = [_normalize_trigger(joystick.get_axis(5))]
    else:
        state["LEFT_TRIGGER"] = [0]
        state["RIGHT_TRIGGER"] = [0]
        
    return state

def _get_game_window(process_name_fragment="Vampire"):

    l = setup_logger("VS_Bot") 
    
    l.info(f"[Recorder] Searching for process matching '{process_name_fragment}'...")
    pid = None
    
    for proc in psutil.process_iter(['pid', 'name']):
        if process_name_fragment.lower() in proc.info['name'].lower():
            pid = proc.info['pid']
            l.info(f"[Recorder] Found Process: {proc.info['name']} (PID: {pid})")
            break
            
    if pid is None:
        l.warning(f"[Recorder] WARNING: Process matching '{process_name_fragment}' not found.")
        return None, None

    windows = []
    def enum_window_callback(hwnd, pid_to_find):
        try:
            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
            if found_pid == pid_to_find and win32gui.IsWindowVisible(hwnd):
                windows.append(hwnd)
        except:
            pass
        return True

    win32gui.EnumWindows(enum_window_callback, pid)
    
    if not windows:
        l.warning(f"[Recorder] WARNING: No visible windows for PID {pid}")
        return None, None
        
    hwnd = windows[0] # Assume main window
    
    # Get Client Area
    rect = win32gui.GetClientRect(hwnd)
    pt_tl = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
    pt_br = win32gui.ClientToScreen(hwnd, (rect[2], rect[3]))
    bbox = (pt_tl[0], pt_tl[1], pt_br[0], pt_br[1])
    
    return hwnd, bbox

def _capture_process(stop_event, video_filename, action_filename, fps):
    # This runs in a separate process
    # Re-initialize logger for this process to ensure consistent formatting
    proc_logger = setup_logger("VS_Bot") # Recycle same name for consistency
    
    # Suppress pkg_resources warning from pygame
    warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

    try:
        pygame.init()
        pygame.joystick.init()
        
        # Wait for controller
        retry_count = 0
        while pygame.joystick.get_count() == 0 and not stop_event.is_set():
            if retry_count % 5 == 0:
                proc_logger.info("[Recorder] Waiting for controller...")
            time.sleep(1)
            pygame.joystick.quit()
            pygame.joystick.init()
            retry_count += 1
        
        if stop_event.is_set():
            return

        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        proc_logger.info(f"[Recorder] Connected to joystick: {joystick.get_name()}")

        # Find Window
        hwnd, bbox = _get_game_window("Vampire")
        if not bbox:
            proc_logger.error("[Recorder] Could not find game window. Aborting capture.")
            return

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        proc_logger.info(f"[Recorder] Capture Region: {bbox} ({width}x{height})")
        
        if width % 2 != 0: width -= 1
        if height % 2 != 0: height -= 1

        # Init Camera
        # Dxcam needs to be created in the same process it is used
        camera = dxcam.create(output_idx=0, output_color="BGR")
        if camera is None:
            proc_logger.error("[Recorder] Failed to create DXCAM.")
            return
            
        # Start the camera in target_fps mode to enforce simple rate limiting at source
        # or grab repeatedly. If we want exact sync with loop, just grab.
        camera.start(target_fps=fps, video_mode=True, region=bbox)

        # Init Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
        
        jsonl_file = open(action_filename, 'w')
        
        frame_duration = 1.0 / fps
        step_index = 0
        
        proc_logger.info("[Recorder] Recording started (Process).")
        
        while not stop_event.is_set():
            start_time = time.perf_counter()
            
            # Get latest frame from dxcam buffer
            # With target_fps=60 and video_mode=True, .get_latest_frame() returns the most recent frame
            # This is better than grab() blocking?
            # Actually, dxcam doc says: .start() -> background thread updates buffer.
            # .get_latest_frame() returns the last frame.
            frame = camera.get_latest_frame()
            
            if frame is None:
                continue

            # Poll inputs
            inputs = _get_input_state(joystick, step_index)
            
            # Save
            # Resize if needed (dxcam region might be slightly off if we adjusted width/height for evenness)
            if frame.shape[1] != width or frame.shape[0] != height:
                 frame = frame[:height, :width]

            video_writer.write(frame)
            json.dump(inputs, jsonl_file)
            jsonl_file.write('\n')
            
            step_index += 1
            
            # Sleep to maintain FPS
            elapsed = time.perf_counter() - start_time
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except Exception as e:
        proc_logger.error(f"[Recorder] Error in capture process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        proc_logger.info("[Recorder] Cleaning up process...")
        if 'camera' in locals() and camera:
            try: camera.stop() 
            except: pass
        if 'video_writer' in locals():
            video_writer.release()
        if 'jsonl_file' in locals():
            jsonl_file.close()
        pygame.quit()

class Recorder:
    def __init__(self):
        self._process = None
        self._stop_event = multiprocessing.Event()
        self.enabled = config.get('capture.enabled', False)
        
        if not self.enabled:
            return

        self.output_dir = config.get('capture.output_dir', 'training_data')
        self.fps = config.get('capture.fps', 60)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.video_filename = os.path.join(self.output_dir, f"capture_{timestamp}.mp4")
        self.action_filename = os.path.join(self.output_dir, f"capture_{timestamp}.jsonl")
        
    def start(self):
        if not self.enabled:
            return
        
        logger.info(f"[Recorder] Starting capture process...")
        self._stop_event.clear()
        self._process = multiprocessing.Process(
            target=_capture_process,
            args=(self._stop_event, self.video_filename, self.action_filename, self.fps)
        )
        self._process.start()

    def stop(self):
        if not self.enabled or not self._process:
            return
            
        logger.info("[Recorder] Signaling stop to process...")
        self._stop_event.set()
        self._process.join(timeout=5.0)
        if self._process.is_alive():
             logger.warning("[Recorder] Process did not exit, terminating...")
             self._process.terminate()
        logger.info("[Recorder] Capture stopped.")
