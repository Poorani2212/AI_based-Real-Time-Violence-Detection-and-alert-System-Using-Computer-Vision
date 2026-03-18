from ultralytics import YOLO
import cv2
import time
import winsound   # Windows alert sound

model = YOLO("yolov5n.pt")

cap = cv2.VideoCapture(0)

prev_gray = None
save_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    labels = results[0].names
    person_count = 0

    # Count persons
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if labels[cls_id] == "person":
            person_count += 1

    # Motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion = 0

    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        motion = diff.mean()

    prev_gray = gray

    # 🚨 Violence condition
    if person_count >= 2 and motion > 20:
        cv2.putText(annotated, "🚨 VIOLENCE DETECTED!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # 🔊 Beep sound
        winsound.Beep(2500, 500)

        # 📸 Save image
        filename = f"violence_{save_count}.jpg"
        cv2.imwrite(filename, frame)
        save_count += 1

    elif person_count >= 2:
        cv2.putText(annotated, "Multiple People", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("AI Violence Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()