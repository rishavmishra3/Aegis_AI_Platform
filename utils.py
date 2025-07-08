import numpy as np
import cv2

# --- Constants ---
N_FRAMES_FOR_ANALYSIS = 30
FALL_ASPECT_RATIO_THRESHOLD = 1.6
FALL_TIME_THRESHOLD_SECS = 0.5
PANIC_SPEED_THRESHOLD_KPH = 15
PIXELS_PER_METER = 20  # Heuristic, needs calibration per camera

def calculate_fall_score(history):
    """Calculates a score from 0 to 1 indicating if a person has fallen."""
    if len(history) < 2:
        return 0.0

    current_box = history[-1]['box']
    prev_box = history[-2]['box']

    w_curr, h_curr = current_box[2] - current_box[0], current_box[3] - current_box[1]
    
    if h_curr == 0: return 0.0
    aspect_ratio = w_curr / h_curr

    if aspect_ratio > FALL_ASPECT_RATIO_THRESHOLD:
        # Check if the fall was sudden
        cy_curr = (current_box[1] + current_box[3]) / 2
        cy_prev = (prev_box[1] + prev_box[3]) / 2
        
        # A significant drop in the vertical center indicates a fall, not just crouching
        if (cy_curr - cy_prev) > (h_curr * 0.2): # Dropped by more than 20% of their new height
            return 1.0
    return 0.0


def calculate_panic_score(history):
    """Calculates a panic score based on speed."""
    if len(history) < 2:
        return 0.0, 0.0

    current_box = history[-1]['box']
    prev_box = history[-2]['box']
    
    cx_curr = (current_box[0] + current_box[2]) / 2
    cy_curr = (current_box[1] + current_box[3]) / 2
    cx_prev = (prev_box[0] + prev_box[2]) / 2
    cy_prev = (prev_box[1] + prev_box[3]) / 2

    pixel_dist = np.sqrt((cx_curr - cx_prev)**2 + (cy_curr - cy_prev)**2)
    meter_dist = pixel_dist / PIXELS_PER_METER
    
    # Assuming video is 30 FPS for speed calculation
    speed_mps = meter_dist * 30 
    speed_kph = speed_mps * 3.6

    panic_score = min(speed_kph / PANIC_SPEED_THRESHOLD_KPH, 1.0)
    return panic_score, speed_kph


def calculate_conflict_score(persons_data):
    """
    Calculates a score based on how many high-speed individuals are close to each other.
    This is a proxy for potential physical conflict.
    """
    high_panic_persons = [p for p in persons_data if p['panic_score'] > 0.7]
    if len(high_panic_persons) < 2:
        return 0.0
    
    conflict_pairs = 0
    for i in range(len(high_panic_persons)):
        for j in range(i + 1, len(high_panic_persons)):
            p1_box = high_panic_persons[i]['box']
            p2_box = high_panic_persons[j]['box']

            # Simple IoU (Intersection over Union) to check for proximity/overlap
            x1 = max(p1_box[0], p2_box[0])
            y1 = max(p1_box[1], p2_box[1])
            x2 = min(p1_box[2], p2_box[2])
            y2 = min(p1_box[3], p2_box[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            if intersection > 0:
                conflict_pairs += 1

    # Normalize score by the number of high-panic individuals
    return min(conflict_pairs / len(high_panic_persons), 1.0)


# utils.py

# ... (all other functions remain the same) ...

def draw_visuals(frame, persons_list, overall_panic, conflict_score):
    """Draws all visualizations on the frame."""
    # --- THIS IS THE FIX ---
    for data in persons_list:
        track_id = data.get('id', -1)
        box = np.array(data['box']).astype(int)
        kpts = np.array(data['keypoints']).astype(int)
        
        color = (0, 255, 0)
        if data.get('fall_score', 0) > 0.9:
            color = (0, 0, 255)
        elif data.get('panic_score', 0) > 0.7:
            color = (0, 165, 255)
        
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        for k in kpts:
            cv2.circle(frame, tuple(k), 2, color, -1)
        
        speed_kph = data.get('speed_kph', 0)
        info_text = f"ID:{track_id} | S:{speed_kph:.1f}kph"
        cv2.putText(frame, info_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # --- Draw Overall Stats ---
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 60), (0,0,0), -1)
    panic_color = (0, 255, 0)
    if overall_panic > 0.5: panic_color = (0, 0, 255)
    cv2.putText(frame, f"OVERALL PANIC: {overall_panic:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, panic_color, 2)
    conflict_color = (0, 255, 0)
    if conflict_score > 0.4: conflict_color = (0, 165, 255)
    cv2.putText(frame, f"CONFLICT SCORE: {conflict_score:.2f}", (350, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, conflict_color, 2)
    cv2.putText(frame, f"PERSON COUNT: {len(persons_list)}", (w - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    return frame