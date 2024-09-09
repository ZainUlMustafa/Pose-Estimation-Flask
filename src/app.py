from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from PIL import Image
import io
from ultralytics import YOLO
import mediapipe as mp

app = Flask(__name__)
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
    detected_frame = results[0].plot()  # Draw the YOLO bounding boxes on the frame

    # 2. MediaPipe Pose Detection
    frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(detected_frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return detected_frame

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
    socketio.run(app, host='0.0.0.0', port=8080)
