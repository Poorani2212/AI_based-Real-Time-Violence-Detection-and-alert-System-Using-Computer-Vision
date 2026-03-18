import cv2
import numpy as np
import os
import winsound

# Create folder for saving images
if not os.path.exists("images"):
    os.makedirs("images")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start camera
cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 👤 FACE DETECTION
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    # 🔊 PERSON ALERT
    if len(faces) > 0:
        cv2.putText(frame, "PERSON DETECTED", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        winsound.Beep(1000, 300)

    # 🚨 VIOLENCE DETECTION (2+ faces)
    if len(faces) >= 2:
        cv2.putText(frame, "VIOLENCE DETECTED!", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        winsound.Beep(2000, 500)

        # Save image
        cv2.imwrite(f"images/violence_{count}.jpg", frame)
        count += 1

    # 🔪 WEAPON DETECTION (edge logic)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = cv2.countNonZero(edges)

    if edge_count > 50000:  # adjust if needed
        cv2.putText(frame, "WEAPON DETECTED!", (50,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        winsound.Beep(2500, 500)

        # Save image
        cv2.imwrite(f"images/weapon_{count}.jpg", frame)
        count += 1

    # Show video
    cv2.imshow("AI SECURITY SYSTEM", frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()