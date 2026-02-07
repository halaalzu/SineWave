import cv2
import mediapipe as mp
import numpy as np

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

def draw_landmarks_on_image(image, detection_result):
    """Draw hand landmarks on the image."""
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_landmark = hand_landmarks[start_idx]
                end_landmark = hand_landmarks[end_idx]
                
                start_point = (int(start_landmark.x * image.shape[1]), 
                             int(start_landmark.y * image.shape[0]))
                end_point = (int(end_landmark.x * image.shape[1]), 
                           int(end_landmark.y * image.shape[0]))
                
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    
    return image

# Setup MediaPipe Hands with new API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'hand_landmarker.task')

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# For webcam input:
cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect hand landmarks
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33  # Approximately 30 FPS
        
        # Draw the hand annotations on the image
        annotated_image = image.copy()
        annotated_image = draw_landmarks_on_image(annotated_image, detection_result)
        
        # Flip the image horizontally for a selfie-view display
        cv2.imshow('MediaPipe Hands', cv2.flip(annotated_image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:  # ESC key
            break

cap.release()
cv2.destroyAllWindows()

 