from ultralytics import YOLO
import cv2
import torch

# Load models
general_model = YOLO(r'yolov8m.pt')        # general COCO model
bisleri_model = YOLO(r'best9.pt')          # your trained model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
general_model.to(device)
bisleri_model.to(device)
print(f"✅ Using device: {device}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Detect general objects
    results_general = general_model(frame, conf=0.5, device=device, verbose=False)
    
    for r in results_general:
        boxes = r.boxes
        names = r.names
        
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Check only bottles
            if label == 'bottle':
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    # Step 2: Run Bisleri model on cropped bottle
                    results_bisleri = bisleri_model(roi, conf=0.65, device=device, verbose=False)
                    bisleri_found = False
                    for rb in results_bisleri:
                        for bb in rb.boxes:
                            cls2 = int(bb.cls[0])
                            label2 = rb.names[cls2].lower()
                            if label2 == 'bisleri':
                                bisleri_found = True

                    # Draw box
                    color = (0, 255, 0) if bisleri_found else (0, 0, 255)
                    text = "Bisleri" if bisleri_found else "Other Bottle"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # For other objects like person, draw normally
            elif label == 'person':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Bisleri Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
