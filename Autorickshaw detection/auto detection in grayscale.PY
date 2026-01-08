import cv2
from ultralytics import YOLO

# 1. Load the model
model = YOLO('bestA1.pt')

# 2. Set your "Strong" Confidence Threshold
MIN_CONFIDENCE = 0.75

# 3. Open Webcam
# TRY 1 FIRST. If the code crashes or opens the wrong camera, change this to 0.
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read from camera. Try changing cv2.VideoCapture(1) to 0.")
        break

    # --- STEP A: GRAYSCALE CONVERSION ---
    # Convert to single channel gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Duplicate the channel 3 times to make it "fake" BGR for YOLO
    # Input shape becomes (Height, Width, 3)
    input_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # --- STEP B: INFERENCE WITH HIGH CONFIDENCE ---
    results = model(input_frame, conf=MIN_CONFIDENCE, verbose=False)

    # --- STEP C: VISUALIZE ---
    # Initialize annotated_frame to input_frame in case no detections occur
    annotated_frame = input_frame

    # Plot results on the frame
    for result in results:
        annotated_frame = result.plot()

    # Show the result
    cv2.imshow("YOLOv8 Grayscale - USB Camera", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()