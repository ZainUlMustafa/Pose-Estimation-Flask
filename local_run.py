import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use a small YOLOv8 model for faster inference

# Initialize MediaPipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)  # For webcam, change to video path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Use YOLOv8 to detect people in the frame
    results = model(frame)
    boxes = results[0].boxes.cpu().numpy()  # Extract bounding boxes

    for box in boxes:
        if box.cls == 0:  # YOLO class 0 corresponds to 'person'
            # Step 2: Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box on the original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Step 3: Crop the detected person from the frame
            person_frame = frame[y1:y2, x1:x2]

            # Step 4: Apply MediaPipe Pose Estimation on the cropped person
            person_rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(person_rgb)

            # If pose landmarks are detected
            if results_pose.pose_landmarks:
                # Step 5: Draw pose landmarks on the cropped person frame
                mp_drawing.draw_landmarks(
                    person_frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Step 6: Overlay the cropped pose back onto the original frame
            frame[y1:y2, x1:x2] = person_frame

    # Display the final frame
    cv2.imshow('YOLOv8 + MediaPipe Pose Estimation', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
