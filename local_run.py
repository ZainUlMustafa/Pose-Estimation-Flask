import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Load company logo (adjust path to your logo file)
logo = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)

# Function to overlay a transparent PNG logo on the frame
def overlay_logo(frame, logo, position=(10, 10), scale=0.2):
    logo_h, logo_w, _ = logo.shape

    # Resizing logo based on the scale
    logo = cv2.resize(logo, (0, 0), fx=scale, fy=scale)
    logo_h, logo_w, _ = logo.shape
    
    x, y = position
    
    # Extract regions where the logo will be placed
    roi = frame[y:y + logo_h, x:x + logo_w]
    
    # Create a mask for the logo
    logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(logo_gray, 1, 255, cv2.THRESH_BINARY)

    # Create an inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Black-out the area of the logo in the region of interest
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    
    # Take only the logo region
    logo_fg = cv2.bitwise_and(logo, logo, mask=mask)

    # Put logo on the frame
    # dst = cv2.add(img_bg, logo_fg)
    # frame[y:y + logo_h, x:x + logo_w] = dst
    return img_bg

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB before processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(rgb_frame)

    # Convert back to BGR for OpenCV
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )

    # Overlay the company logo in the top-left corner
    # frame = overlay_logo(frame, logo)

    # Display the result
    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Exit when 'ESC' is pressed
        break

cap.release()
cv2.destroyAllWindows()
