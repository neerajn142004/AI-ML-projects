import cv2
from ultralytics import YOLO

# 1. Load the model
model = YOLO('bestA1.pt')

# 2. Set your "Strong" Confidence Threshold
# 0.75 means the model must be 75% confident to show a box.
MIN_CONFIDENCE = 0.75

# 3. Open Webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- INFERENCE ---
    # We feed the 'frame' directly. No color conversion needed.
    # YOLO automatically handles BGR (OpenCV format) internally.
    results = model(frame, conf=MIN_CONFIDENCE, stream=True)

    # --- VISUALIZE ---
    for result in results:
        # Plot the detections on the frame
        annotated_frame = result.plot()
        
        # Optional: Print the number of strong detections
        count = len(result.boxes)
        cv2.putText(annotated_frame, f"Strong Detections: {count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the result
    cv2.imshow("YOLOv8 Color - High Confidence", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()