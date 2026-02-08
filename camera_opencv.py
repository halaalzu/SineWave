import cv2
import mediapipe as mp
import numpy as np
import os

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
            
            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 120, 255), -1)
                cv2.circle(image, (x, y), 3, (255, 255, 255), -1)
    
    return image

def draw_ui_overlay(image, fps):
    """Draw UI overlay with information."""
    # Semi-transparent overlay for text background
    overlay = image.copy()
    
    # Title background
    cv2.rectangle(overlay, (10, 10), (400, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    
    # Title text
    cv2.putText(image, "FlowState Hand Tracking", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    # FPS counter
    cv2.putText(image, f"FPS: {fps:.1f}", (image.shape[1] - 120, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Instructions at bottom
    instructions = [
        "ESC - Exit",
        "Q - Quit"
    ]
    
    y_offset = image.shape[0] - 60
    for instruction in instructions:
        cv2.putText(image, instruction, (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    return image

def main():
    # Setup MediaPipe
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'hand_landmarker.task')
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Create window
    cv2.namedWindow('FlowState', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('FlowState', 1280, 720)
    
    frame_timestamp_ms = 0
    fps = 0
    frame_count = 0
    start_time = cv2.getTickCount()
    
    print("Camera window opened. Press ESC or Q to quit.")
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            # Capture frame
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                continue
            
            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hand landmarks
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33  # ~30 FPS
            
            # Draw hand landmarks
            frame = draw_landmarks_on_image(frame, detection_result)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = cv2.getTickCount()
                time_diff = (end_time - start_time) / cv2.getTickFrequency()
                fps = 10 / time_diff
                start_time = end_time
            
            # Draw UI overlay
            frame = draw_ui_overlay(frame, fps)
            
            # Display the frame
            cv2.imshow('FlowState', frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

if __name__ == "__main__":
    main()
