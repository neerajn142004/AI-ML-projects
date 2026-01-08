import cv2
from ultralytics import YOLO
import numpy as np

# --- 1. CONFIGURATION & SETUP ---

# Load the pre-trained YOLOv8 SEGMENTATION model
model = YOLO('yolov8n-seg.pt') 

# Define the source (0 for default webcam, or a video path)
VIDEO_SOURCE = r'C:\Users\Neeraj\Desktop\trains\train2.mp4' # Or 0 for live stream

# Frame Flash Threshold (Only used for flashing the screen, not for object color)
# Only flash the screen if ANY detected object's area is greater than this threshold.
FRAME_FLASH_THRESHOLD = 50000 

# Define the set of CRITICAL DANGER CLASSES (COCO IDs):
# If the object is ANY of these, it is IMMEDIATELY a RED ALERT, regardless of size/distance.
CRITICAL_DANGER_IDS = {
    # Vehicles & People (Primary Hazards)
    0, 2, 3, 5, 7,     # person, car, motorcycle, bus, truck
    
    # Large Animals (Immediate Road Hazards)
    17, 19, 20, 21, 22, 23, # horse, cow, elephant, bear, zebra, giraffe
    1, 6, 8,           # bicycle, train, boat
    
    # Large Moving/Stationary Obstructions (Road Blockage/Debris)
    57, 58, 60, 61,    # chair, couch, bed, dining table (common debris)
    69, 70, 71, 72, 73 # microwave, oven, toaster, sink, refrigerator (large debris)
}

# Load video stream
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Initialize CLAHE for better night vision
clahe_processor = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

print("Dashcam ADAS (Aggressive Alert Mode with CLAHE) Running... Press 'q' to quit.")

# --- 2. MAIN DETECTION LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- IMAGE PRE-PROCESSING (NEW: CLAHE for Night Vision) ---
    # Convert frame to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel (lightness)
    cl = clahe_processor.apply(l)
    
    # Merge the enhanced L-channel back and convert to BGR
    limg = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # --- END PRE-PROCESSING ---


    # Run detection on all 80 COCO classes
    results = model(
        frame, 
        stream=True, 
        conf=0.6, 
        verbose=False
    )

    frame_alert = False

    for r in results:
        boxes = r.boxes
        masks = r.masks 
        
        if masks is None:
            continue

        segment_polygons = masks.xy 

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0])
            
            # Calculate object area
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Get class name safely, defaulting to UNKNOWN if not found
            class_name = model.names.get(class_id, "UNKNOWN OBJECT")
            
            
            # --- AGGRESSIVE IDENTIFICATION/ALERT LOGIC ---
            
            if class_id in CRITICAL_DANGER_IDS:
                # --- A. KNOWN CRITICAL OBJECT (RED ALERT) ---
                color = (0, 0, 255) # Red (BGR format)
                alert_text = f"!!! DANGER: {class_name.upper()} !!!"
                frame_alert = True
                
            elif class_name == "UNKNOWN OBJECT":
                # --- B. UNKNOWN OBJECT (RED ALERT) ---
                color = (0, 0, 255) # Red (BGR format)
                alert_text = f"!!! DANGER: UNKNOWN OBJECT ({int(class_id)})!!!"
                frame_alert = True
                
            else:
                # --- C. NON-CRITICAL KNOWN OBJECT (GREEN) ---
                color = (0, 255, 0) # Green (BGR format)
                alert_text = class_name
            
            # Trigger full screen flash if a RED object is also large enough
            if color == (0, 0, 255) and area > FRAME_FLASH_THRESHOLD:
                 frame_alert = True


            # --- INSTANCE SEGMENTATION DRAWING ---
            polygon = segment_polygons[i].astype(int)
            segment_pts = polygon.reshape((-1, 1, 2))
            
            # Create a semi-transparent colored overlay for the segment
            overlay = frame.copy()
            cv2.fillPoly(overlay, [segment_pts], color=color)
            
            # Blend the overlay with the original frame (alpha=0.5 for transparency)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # --- BOUNDING BOX AND TEXT DRAWING ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, alert_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    # --- TRIGGER VISUAL ALERT (Frame Flash) ---
    if frame_alert:
        # Visual Alert: Flash the screen red
        overlay_alert = frame.copy()
        cv2.rectangle(overlay_alert, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        # Blend the red overlay with the frame to create a visual flash
        cv2.addWeighted(overlay_alert, 0.1, frame, 0.9, 0, frame)

    # Display the resulting frame
    cv2.imshow('Dashcam ADAS Segmentation (Aggressive Mode with CLAHE)', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 3. CLEANUP ---
cap.release()
cv2.destroyAllWindows()