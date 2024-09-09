from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load logo
logo = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)

# Function to overlay logo
def overlay_logo(frame, logo, position=(10, 10), scale=0.2):
    logo_h, logo_w, _ = logo.shape
    logo = cv2.resize(logo, (0, 0), fx=scale, fy=scale)
    logo_h, logo_w, _ = logo.shape
    x, y = position
    roi = frame[y:y + logo_h, x:x + logo_w]
    logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(logo_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    logo_fg = cv2.bitwise_and(logo, logo, mask=mask)
    dst = cv2.add(img_bg, logo_fg)
    frame[y:y + logo_h, x:x + logo_w] = dst
    return frame

# Video stream generator function
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose estimation
        results = pose.process(rgb_frame)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # Overlay company logo
        # frame = overlay_logo(frame, logo)

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in byte format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
