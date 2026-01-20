import cv2
import threading
import queue
import time
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from bot.system.config import config
from bot.system.logger import logger
from bot.vision.types import Detection

class Visualizer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue(maxsize=1) # Keep it fresh, drop old frames if backing up
        self.stop_event = threading.Event()
        self.daemon = True
        
        # Configuration
        self.target_fps = 60
        self.frame_duration = 1.0 / self.target_fps
        
        # Recording Setup
        self.recording_enabled = config.get("debug_recording.enabled", False)
        self.writer = None
        self.output_dir = config.get("debug_recording.output_dir", "debug_visualizations")
        self.fps = config.get("debug_recording.fps", 30) # Use config FPS or match target? Plan said 60fps stable. 
        # User requested 60fps recording, so let's override config if needed or respect it.
        # "i do want to save this visualization at 60fps" -> Enforce 60
        self.record_fps = 60 
        
        # Output Resolution
        # "make it teh same size as the input (1280x720)"
        # Game dimensions are usually input, but let's use the explicit request or config
        input_dims = config.get("game.dimensions", (1280, 720))
        self.output_size = tuple(input_dims) # (width, height)
        
        # Internal State
        self.last_frame = None
        self.last_detections = []
        self.last_pilot_state = None
        self.class_names = {}
        
        # Drawing Config (from AnnotationDrawer)
        self.thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_color = (0, 0, 0) # Black text
        
        if self.recording_enabled:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"debug_viz_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Ensure output size (width, height) matches writer
            self.writer = cv2.VideoWriter(filename, fourcc, self.record_fps, self.output_size)
            logger.info(f"Visualizer recording enabled: {filename} @ {self.record_fps} FPS")

    def start(self):
        logger.info("Visualizer thread starting...")
        super().start()

    def stop(self):
        logger.info("Visualizer stopping...")
        self.stop_event.set()
        self.join()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

    def update(self, frame, detections, pilot_state, class_names):
        """
        Push new state to the visualizer.
        frame: raw frame (BGR)
        detections: list of Detection objects (coordinates in model resolution? No, they should be scaled by visualizer if needed, or assumed raw)
        pilot_state: dict or object with force vectors, etc.
        """
        if self.stop_event.is_set():
            return
            
        try:
            # Non-blocking put, remove old item if full to ensure freshness
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put((frame, detections, pilot_state, class_names), block=False)
        except queue.Full:
            pass # Should not happen with get_nowait

    def _run(self):
        """
        Main loop: 60 FPS continuous update.
        """
        while not self.stop_event.is_set():
            start_time = time.perf_counter()
            
            # 1. Update State from Queue
            try:
                # Try to get new data
                data = self.queue.get_nowait()
                self.last_frame, self.last_detections, self.last_pilot_state, self.class_names = data
            except queue.Empty:
                # No new data, persist last state
                pass

            # 2. Draw and Show (if we have a frame)
            if self.last_frame is not None:
                # Create a copy to draw on, so we don't modify the shared original if referenced elsewhere
                # Resize to output size if needed
                viz_frame = cv2.resize(self.last_frame, self.output_size)
                
                # Scale detections to output size
                # Detections are likely in Model Resolution (960x608 specified in config/main)
                # We need to map them to self.output_size (1280x720)
                # Get model resolution from config for scaling ratio
                model_size = tuple(config.get("game.image_size", (960, 608)))
                
                scale_x = self.output_size[0] / model_size[0]
                scale_y = self.output_size[1] / model_size[1]
                
                self._draw_state(viz_frame, self.last_detections, self.last_pilot_state, scale_x, scale_y)
                
                cv2.imshow("Bot Vision", viz_frame)
                cv2.waitKey(1) # maintain window responsivness
                
                # 3. Record
                if self.writer:
                    self.writer.write(viz_frame)

            # 4. Sleep to maintain FPS
            elapsed = time.perf_counter() - start_time
            sleep_time = self.frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def run(self):
        # Wrapper for _run to catch exceptions
        try:
            self._run()
        except Exception as e:
            logger.error(f"Visualizer thread crashed: {e}")
            import traceback
            traceback.print_exc()

    # --- Drawing Helpers (Consolidated from annotations.py and debug.py) ---

    def _draw_state(self, frame, detections, pilot_state, scale_x, scale_y):
        # Draw Detections
        if detections:
            for detection in detections:
                x1, y1, x2, y2 = detection.position
                # Scale
                sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
                sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
                
                label = self.class_names.get(detection.label, detection.label)
                if isinstance(label, int): category = "unknown" # Fallback
                else: category = label
                
                # Color logic from debug.py
                color = (0, 0, 255) if "monster" in str(category).lower() else (255, 0, 0) # Red for monsters, Blue for others
                
                # Draw Box
                self.draw_rectangle(frame, color, (sx1, sy1), (sx2, sy2))
                
                # Draw Label
                debug_text = f"{category}: {detection.confidence:.2f}"
                self.draw_text_with_background(frame, debug_text, (sx1, sy1))

        # Draw Pilot State (Force Vector)
        if pilot_state:
             # pilot_state is expected to be a dict with keys like 'fx', 'fy', 'center'
             # or passed as separate args. Let's assume it's the `pilot` object or a dict.
             # In gameplay_loop, we passed `pilot` object, but passing a snapshot dict is safer for threading.
             # Let's adjust gameplay_loop to pass a dict: {'fx': fx, 'fy': fy, 'center': pilot.center}
             
             # If pilot_state is a dict:
             if isinstance(pilot_state, dict):
                 fx = pilot_state.get('fx', 0)
                 fy = pilot_state.get('fy', 0)
                 cx, cy = pilot_state.get('center', (0,0))
                 
                 # Scale Center
                 scx, scy = int(cx * scale_x), int(cy * scale_y)
                 
                 # Draw Vector
                 end_point = (int(scx + fx * 50), int(scy + fy * 50))
                 cv2.arrowedLine(frame, (scx, scy), end_point, (0, 255, 0), 3)
                 
                 # Draw Target Centroid
                 target = pilot_state.get('target_centroid')
                 if target:
                     tx, ty = target
                     stx, sty = int(tx * scale_x), int(ty * scale_y)
                     cv2.circle(frame, (stx, sty), 10, (0, 0, 255), -1)


    def draw_rectangle(self, frame, color: Tuple[int, int, int], point_a: Tuple[int, int], point_b: Tuple[int, int]):
        cv2.rectangle(frame, point_a, point_b, color, self.thickness)

    def draw_text_with_background(self, frame, text: str, point: Tuple[int, int]):
        label_size, base_line = cv2.getTextSize(text, self.font, self.font_scale, 1)
        # Ensure text doesn't go off top screen
        y = point[1]
        if y < label_size[1] + 5:
            y = point[1] + label_size[1] + 10
            
        background_begin = (point[0], y - label_size[1])
        background_end = (point[0] + label_size[0], y + base_line)
        
        cv2.rectangle(frame, background_begin, background_end, (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (point[0], y), self.font, self.font_scale, self.font_color)
