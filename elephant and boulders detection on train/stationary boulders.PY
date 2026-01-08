import cv2
from ultralytics import YOLO
import numpy as np
import math

# --- 1. CONFIGURATION ---
model_path = r'bestb2.pt' 

try:
    model = YOLO(model_path)
except:
    print(f"Could not find custom model at {model_path}. Loading standard YOLO...")
    model = YOLO('yolov8m-seg.pt')

VIDEO_SOURCE = r'C:\Users\Neeraj\Desktop\GRAY\stones.mp4'

# --- MOTION SETTINGS ---
# If a rock moves less than 3 pixels per frame, we consider it "Stationary"
MOVEMENT_THRESHOLD = 3.0 

# Memory now stores 'centroid' to calculate speed
# Structure: { track_id: {'box':..., 'centroid': (x,y), 'missed': 0, ...} }
object_memory = {}      

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Rock Stationary vs Falling Detector Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- GRAYSCALE PIPELINE ---
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    model_input = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # --- DETECTION & TRACKING ---
    results = model.track(
        model_input, 
        stream=True, 
        persist=True, 
        conf=0.35,      
        imgsz=1280,     
        classes=[0],    
        tracker="botsort.yaml", 
        verbose=False
    )

    current_frame_ids = [] 
    any_falling_alert = False

    for r in results:
        boxes = r.boxes
        masks = r.masks 
        
        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            if box.id is None:
                continue 
            
            track_id = int(box.id.item())
            current_frame_ids.append(track_id)

            # 1. Get Coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 2. Calculate Current Center (Centroid)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            current_centroid = (cx, cy)

            # 3. Get Mask
            segment_pts = None
            if masks is not None and len(masks) > i:
                 if hasattr(masks, 'xy') and len(masks.xy) > i and masks.xy[i].size > 0:
                    segment_pts = masks.xy[i].astype(int).reshape((-1, 1, 2))

            # --- 4. STATIONARY vs FALLING LOGIC ---
            
            # Retrieve previous data for this specific rock ID
            prev_data = object_memory.get(track_id)
            
            speed = 0.0
            if prev_data is not None:
                prev_cx, prev_cy = prev_data['centroid']
                # Calculate Euclidean distance (Movement in pixels)
                speed = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)

            # Determine State based on Speed
            if speed > MOVEMENT_THRESHOLD:
                # Moving fast -> FALLING
                status = "FALLING!"
                color = (0, 0, 255) # Red
                any_falling_alert = True
            else:
                # Moving little/none -> STATIONARY
                status = "Stationary"
                color = (0, 255, 0) # Green

            display_text = f"{status} [ID:{track_id}] Speed:{int(speed)}"

            # --- 5. DRAWING ---
            if segment_pts is not None:
                overlay = model_input.copy()
                cv2.fillPoly(overlay, [segment_pts], color=color)
                cv2.addWeighted(overlay, 0.4, model_input, 0.6, 0, model_input)
            
            cv2.rectangle(model_input, (x1, y1), (x2, y2), color, 2)
            cv2.putText(model_input, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- 6. UPDATE MEMORY ---
            # We overwrite the old data with new data for the next loop
            object_memory[track_id] = {
                'box': (x1, y1, x2, y2),
                'centroid': current_centroid, # <--- Save center for speed calc next frame
                'mask': segment_pts,
                'text': display_text,
                'color': color,
                'missed': 0 
            }

    # --- MEMORY HANDLER (GHOST BOXES) ---
    for track_id in list(object_memory.keys()):
        if track_id not in current_frame_ids:
            object_memory[track_id]['missed'] += 1
            if object_memory[track_id]['missed'] > 10: # MAX_MISSED_FRAMES
                del object_memory[track_id]
                continue

            # Draw Ghost (Last known state)
            data = object_memory[track_id]
            gx1, gy1, gx2, gy2 = data['box']
            g_mask = data['mask']
            
            # Use Grey/Yellow to indicate "Lost Signal"
            ghost_color = (0, 255, 255) 
            ghost_text = f"{data['text']} (?)"

            if g_mask is not None:
                overlay = model_input.copy()
                cv2.fillPoly(overlay, [g_mask], color=ghost_color)
                cv2.addWeighted(overlay, 0.2, model_input, 0.8, 0, model_input) 

            cv2.rectangle(model_input, (gx1, gy1), (gx2, gy2), ghost_color, 2)
            cv2.putText(model_input, ghost_text, (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ghost_color, 2)

    # --- ALERT FLASH (Only flash if something is actually FALLING) ---
    if any_falling_alert:
        cv2.rectangle(model_input, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10) 

    cv2.imshow('Rock Status Detector', model_input)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()