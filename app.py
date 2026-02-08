from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

class HandTracker:
    def __init__(self):
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
        
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0
    
    def draw_landmarks(self, image, detection_result):
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
    
    def process_frame(self, frame):
        """Process frame with MediaPipe."""
        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        detection_result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        self.frame_timestamp_ms += 33  # ~30 FPS
        
        # Draw hand landmarks
        frame = self.draw_landmarks(frame, detection_result)
        
        return frame
    
    def close(self):
        self.landmarker.close()

# Initialize camera and tracker
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
tracker = HandTracker()

def generate_frames():
    """Generate frames for video streaming."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame with hand tracking
        frame = tracker.process_frame(frame)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎥 FlowState Hand Tracking Web App")
    print("="*60)
    print("\n📱 Open your browser and go to:")
    print("   http://localhost:5001")
    print("   or")
    print("   http://127.0.0.1:5001")
    print("\n⌨️  Press CTRL+C to stop the server\n")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    finally:
        camera.release()
        tracker.close()
        print("✅ Camera released. Goodbye!\n")
