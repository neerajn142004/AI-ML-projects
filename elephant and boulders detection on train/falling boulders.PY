import cv2
from ultralytics import YOLO
import numpy as np

# --- 1. CONFIGURATION ---

# IMPORTANT: Point this to your CUSTOM TRAINED model
# Standard 'yolov8m.pt' will NOT detect rocks. You must train it.
model_path = r'bestb2.pt' 
# If you haven't trained one yet, use 'yolov8m-seg.pt' just to test the code (it won't detect rocks though)
try:
    model = YOLO(model_path)
except:
    print(f"Could not find custom model at {model_path}. Loading standard YOLO for testing code logic...")
    model = YOLO('yolov8m-seg.pt')

VIDEO_SOURCE = r'C:\Users\Neeraj\Desktop\GRAY\stone5.mp4' # Update your video path

# Rocks are smaller than elephants, so we lower the threshold.
# Any rock bigger than 5000 pixels is "CRITICAL".
ALERT_AREA_THRESHOLD = 5000 

# MEMORY SETTINGS (Crucial for fast falling rocks)
# Rocks might disappear for a split second while falling. We hold the box for 10 frames.
MAX_MISSED_FRAMES = 10 
object_memory = {}      

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Falling Rock Detection System (Grayscale) Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- GRAYSCALE PIPELINE ---
    # Good for rocks: Focuses on contrast/shape rather than color blending
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    model_input = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # --- DETECTION & TRACKING ---
    results = model.track(
        model_input, 
        stream=True, 
        persist=True, 
        conf=0.50,      # Higher confidence to avoid mistaking mountain shadows for rocks
        imgsz=1280,     # High Res is critical for small rocks
        classes=[0],    # <--- ASSUMPTION: Your custom model has 'Rock' as Class 0
        tracker="botsort.yaml", 
        verbose=False
    )

    current_frame_ids = [] 
    any_alert = False

    for r in results:
        boxes = r.boxes
        masks = r.masks 
        
        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            # 1. Get ID 
            if box.id is None:
                continue 
            
            track_id = int(box.id.item())
            current_frame_ids.append(track_id)

            # 2. Get Coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            width = x2 - x1
            height = y2 - y1
            area = width * height

            # 3. Get Mask (If your custom model supports segmentation)
            segment_pts = None
            if masks is not None and len(masks) > i:
                 if hasattr(masks, 'xy') and len(masks.xy) > i and masks.xy[i].size > 0:
                    segment_pts = masks.xy[i].astype(int).reshape((-1, 1, 2))

            # 4. Alert Logic
            if area > ALERT_AREA_THRESHOLD:
                color = (0, 0, 255) # Red
                text = f"CRITICAL: FALLING ROCK [ID:{track_id}]"
                any_alert = True
            else:
                color = (0, 165, 255) # Orange (Warning)
                text = f"Rock Detected [ID:{track_id}]"

            # 5. DRAW LIVE OBJECT
            if segment_pts is not None:
                overlay = model_input.copy()
                cv2.fillPoly(overlay, [segment_pts], color=color)
                cv2.addWeighted(overlay, 0.4, model_input, 0.6, 0, model_input)
            
            cv2.rectangle(model_input, (x1, y1), (x2, y2), color, 2)
            cv2.putText(model_input, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 6. UPDATE MEMORY
            object_memory[track_id] = {
                'box': (x1, y1, x2, y2),
                'mask': segment_pts,
                'text': text,
                'color': color,
                'missed': 0 
            }

    # --- MEMORY HANDLER (ANTI-FLICKER) ---
    for track_id in list(object_memory.keys()):
        if track_id not in current_frame_ids:
            object_memory[track_id]['missed'] += 1
            missed_count = object_memory[track_id]['missed']

            if missed_count > MAX_MISSED_FRAMES:
                del object_memory[track_id]
                continue

            # Draw Ghost Box (Prediction)
            data = object_memory[track_id]
            gx1, gy1, gx2, gy2 = data['box']
            g_mask = data['mask']
            
            ghost_color = (0, 255, 255) # Yellow for memory
            ghost_text = f"{data['text']} (?)"

            if g_mask is not None:
                overlay = model_input.copy()
                cv2.fillPoly(overlay, [g_mask], color=ghost_color)
                cv2.addWeighted(overlay, 0.2, model_input, 0.8, 0, model_input) 

            cv2.rectangle(model_input, (gx1, gy1), (gx2, gy2), ghost_color, 2)
            cv2.putText(model_input, ghost_text, (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ghost_color, 2)

    # --- VISUAL FLASH ON ALERT ---
    if any_alert:
        overlay_alert = model_input.copy()
        # Flash borders instead of full screen (less distracting for rocks)
        cv2.rectangle(overlay_alert, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 50) 
        cv2.addWeighted(overlay_alert, 0.3, model_input, 0.7, 0, model_input)

    cv2.imshow('Rockfall Detection System', model_input)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()