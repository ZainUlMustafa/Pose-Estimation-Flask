from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from PIL import Image
import io
from ultralytics import YOLO
import mediapipe as mp
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize YOLOv5 Model (ultralytics)
yolo_model = YOLO("yolov8n.pt")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# YOLO and Pose processing
def process_frame_with_yolo_and_pose(frame):
    # 1. YOLO Object Detection
    results = yolo_model.predict(frame, verbose=False)
    detected_frame = frame.copy()  # Copy the frame for drawing

    # 2. MediaPipe Pose Detection
    frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)

    # Process detections
    for result in results[0].boxes.data:
        cls = int(result[5].item())  # Get the class index
        if cls == 0:  # YOLO class 0 corresponds to 'person'
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, result[:4].tolist())

            # Draw bounding box around the detected person
            cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # Crop the detected person from the frame
            person_frame = detected_frame[y1:y2, x1:x2]

            # Convert cropped person frame to RGB for MediaPipe processing
            person_rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)

            # Apply MediaPipe Pose Estimation on the cropped person
            results_pose = pose.process(person_rgb)

            # Draw pose landmarks if detected
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    person_frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # Place the processed person frame back into the original frame
            detected_frame[y1:y2, x1:x2] = person_frame

    detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
    return detected_frame_rgb

# Handle incoming WebSocket frames
@socketio.on('frame')
def handle_frame(data):
    # Decode base64 image
    frame_data = data.split(',')[1]
    frame_bytes = base64.b64decode(frame_data)
    image = Image.open(io.BytesIO(frame_bytes))

    # Convert image to OpenCV format
    frame = np.array(image)

    # Process frame with YOLO and Pose
    processed_frame = process_frame_with_yolo_and_pose(frame)

    # Encode processed frame back to base64
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    # Send processed frame back to client
    emit('processed_frame', 'data:image/jpeg;base64,' + processed_frame_base64)

# Serve the client-side HTML
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, allow_unsafe_werkzeug=True)
