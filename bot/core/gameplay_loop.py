import cv2
from bot.system.config import config
from bot.system.logger import logger

from bot.utils import check_and_update_view_position, handle_pause

def process_gameplay_frame(frame_raw, inference_model, pilot, bot, visualizer, 
                           pause_event, key_press, game_area):
    
    IMAGE_SIZE = tuple(config.get("game.image_size", (960, 608)))
    
    # Resize frame for Object Detection and Pilot (Model expects IMAGE_SIZE)
    frame = cv2.resize(frame_raw, IMAGE_SIZE)

    detections, class_names = inference_model.get_detections(frame)
    
    # [NEW] Filter out detections in the center (Player Self-Detection)
    # Screen center is approximately IMAGE_SIZE / 2
    center_x, center_y = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
    filtered_detections = []
    for d in detections:
        x1, y1, x2, y2 = d.position
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist_sq = (cx - center_x)**2 + (cy - center_y)**2
        
        # Ignorance Radius: 50 pixels (squared = 2500)
        if dist_sq > config.get("pilot.center_exclusion_radius_sq", 2500): 
            filtered_detections.append(d)
    
    detections = filtered_detections

    # Update Pilot State and Calculate Force
    pilot.update(detections, class_names)
    fx, fy = pilot.get_force_vector(detections, class_names)
    
    # Normalize vector to ensure magnitude <= 1.0 (clamped)
    magnitude = (fx**2 + fy**2)**0.5
    if magnitude > 1.0:
        fx /= magnitude
        fy /= magnitude
    
    bot.update_movement(fx, fy)
    
    # Send Data to Visualizer
    pilot_state = {
        'fx': fx,
        'fy': fy,
        'center': pilot.center,
        'target_centroid': pilot.get_debug_info().get('target_centroid')
    }
    
    if visualizer:
        visualizer.update(frame_raw, detections, pilot_state, class_names)

    check_and_update_view_position(key_press, game_area)
    handle_pause(key_press, pause_event)

    check_and_update_view_position(key_press, game_area)
    handle_pause(key_press, pause_event)
