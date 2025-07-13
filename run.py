import cv2
from ultralytics import YOLO

# Config
MODEL_PATH = "model.py" 
TARGET_CLASS_NAME = "dish"

# Load the trained model
model = YOLO(MODEL_PATH)
class_names = model.names # class ID -> name mapping

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    # Process each frame's detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = class_names[cls_id]

            # Draw boxes
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Check if class is detected
            if label.lower() == TARGET_CLASS_NAME.lower():
                print(f"[ALERT] Detected: {label} ({conf:.2f})")
        
        # Display the live frame
        cv2.imshow("Live Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
