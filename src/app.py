from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
print("CORS enabled")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")  # Load YOLOv8 model for person detection

# Default camera index
camera_index = 0
print("Default camera index")

# Video stream generator function
def generate_frames():
    global camera_index
    cap = cv2.VideoCapture(camera_index)  # Use the selected camera
    print(f'Cam index: {camera_index}')

    while True:
        success, frame = cap.read()
        if not success:
            print('Not successful')
            break

        print('Frame grab successful')
        # Run YOLOv8 object detection
        results = model(frame)
        boxes = results[0].boxes.cpu().numpy()  # Extract bounding boxes

        # Filter for people (class 0) and sort by confidence, take top 3
        person_boxes = [box for box in boxes if box.cls == 0]
        person_boxes = sorted(person_boxes, key=lambda x: x.conf[0], reverse=True)[:3]

        # Iterate through top 3 detected people
        for box in person_boxes:
            print('Box proc')
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw bounding box on the original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop the detected person from the frame
            person_frame = frame[y1:y2, x1:x2]

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

            # Place the cropped and processed person frame back into the original frame
            frame[y1:y2, x1:x2] = person_frame

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in byte format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    print('Rendering template index')
    return 'Connected', 200

@app.route('/proc')
def proc():
    print('Rendering template index')
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_camera/<int:selected_camera_index>', methods=['GET'])
def set_camera(selected_camera_index):
    global camera_index
    camera_index = selected_camera_index  # Update the camera index based on user selection
    print(f'Selected: {camera_index}')
    return '', 200  # Return a 200 OK response to confirm

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
