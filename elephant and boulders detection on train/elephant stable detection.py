import cv2
from ultralytics import YOLO
import numpy as np

# --- 1. CONFIGURATION ---
model = YOLO('yolov8s-seg.pt')  # Medium model for stability

VIDEO_SOURCE = r'C:\Users\Neeraj\Desktop\trains\vid2.mp4'
ALERT_AREA_THRESHOLD = 50000 

# MEMORY SETTINGS (The Fix for Blinking)
MAX_MISSED_FRAMES = 20  # Keep showing box for ~20 frames if detection is lost
object_memory = {}      # Stores data: { track_id: {'box':..., 'mask':..., 'missed': 0, 'color':...} }

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Continuous Elephant Detection (Anti-Flicker) Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- GRAYSCALE PIPELINE ---
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    model_input = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # --- DETECTION & TRACKING ---
    # We use 'botsort.yaml' which is often smoother than 'bytetrack' for large animals
    results = model.track(
        model_input, 
        stream=True, 
        persist=True, 
        conf=0.2,       # Lower confidence slightly to catch difficult angles
        imgsz=1280, 
        classes=[20],   # Elephant Only
        tracker="botsort.yaml", 
        verbose=False
    )

    current_frame_ids = [] # To track which elephants we actually SAW this frame
    any_alert = False

    for r in results:
        boxes = r.boxes
        masks = r.masks 
        
        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            # 1. Get ID (Critical for tracking)
            if box.id is None:
                continue # Skip objects with no ID yet (noise)
            
            track_id = int(box.id.item())
            current_frame_ids.append(track_id)

            # 2. Get Coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            width = x2 - x1
            height = y2 - y1
            area = width * height

            # 3. Get Mask
            segment_pts = None
            if masks is not None and len(masks) > i:
                 if masks.xy[i].size > 0:
                    segment_pts = masks.xy[i].astype(int).reshape((-1, 1, 2))

            # 4. Alert Logic (Live Object)
            if area > ALERT_AREA_THRESHOLD:
                color = (0, 0, 255) # Red
                text = f"ALERT: ELEPHANT"
                any_alert = True
            else:
                color = (0, 255, 0) # Green
                text = f"Elephant"

            # 5. DRAW LIVE OBJECT
            if segment_pts is not None:
                overlay = model_input.copy()
                cv2.fillPoly(overlay, [segment_pts], color=color)
                cv2.addWeighted(overlay, 0.4, model_input, 0.6, 0, model_input)
            
            cv2.rectangle(model_input, (x1, y1), (x2, y2), color, 2)
            cv2.putText(model_input, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 6. UPDATE MEMORY (Save this elephant's data)
            object_memory[track_id] = {
                'box': (x1, y1, x2, y2),
                'mask': segment_pts,
                'text': text,
                'color': color,
                'missed': 0 # Reset missed counter because we just saw it
            }

    # --- MEMORY HANDLER (THE ANTI-BLINK LOGIC) ---
    # Look at every elephant we have in memory
    # keys() must be converted to list to avoid runtime error during deletion
    for track_id in list(object_memory.keys()):
        
        # If this elephant was NOT seen in the current frame...
        if track_id not in current_frame_ids:
            
            # Increment 'missed' counter
            object_memory[track_id]['missed'] += 1
            missed_count = object_memory[track_id]['missed']

            # If it's been gone too long, forget it
            if missed_count > MAX_MISSED_FRAMES:
                del object_memory[track_id]
                continue

            # OTHERWISE: DRAW THE "GHOST" (Last known position)
            data = object_memory[track_id]
            gx1, gy1, gx2, gy2 = data['box']
            g_mask = data['mask']
            
            # Use Orange to indicate "Tracking Memory"
            ghost_color = (0, 165, 255) 
            ghost_text = f"{data['text']} (Holding...)"

            # Draw Ghost Mask
            if g_mask is not None:
                overlay = model_input.copy()
                cv2.fillPoly(overlay, [g_mask], color=ghost_color)
                cv2.addWeighted(overlay, 0.2, model_input, 0.8, 0, model_input) # Fainter

            # Draw Ghost Box
            cv2.rectangle(model_input, (gx1, gy1), (gx2, gy2), ghost_color, 2)
            cv2.putText(model_input, ghost_text, (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ghost_color, 2)

    # --- SCREEN FLASH ---
    if any_alert:
        overlay_alert = model_input.copy()
        cv2.rectangle(overlay_alert, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        cv2.addWeighted(overlay_alert, 0.1, model_input, 0.9, 0, model_input)

    cv2.imshow('Stable Elephant Monitor', model_input)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()