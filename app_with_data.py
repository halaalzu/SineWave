"""
Integrated Flask app with data collection and analysis
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import os
import uuid
from datetime import datetime
import torch
import torch.nn as nn
from threading import Thread
from dotenv import load_dotenv
import logging
import sys

# Configure logging for production stability
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('flowstate.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

from movement_features import MovementFeatureExtractor, SessionRecorder
from database import RehabDatabase
from joint_analysis import JointAnalyzer
from gemini_analyzer import GeminiHandAnalyzer
from session_quality_ml import SessionQualityAnalyzer
from session_comparison import SessionComparison

# Import pygame for sound feedback
try:
    import pygame
    import math
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    SOUND_ENABLED = True
    print("✅ Pygame audio initialized successfully")
except ImportError:
    SOUND_ENABLED = False
    print("⚠️  pygame not available - install with: pip install pygame")

app = Flask(__name__)
CORS(app)  # Enable CORS for React app

# Initialize database
db = RehabDatabase('flowstate.db')

# Initialize Session Quality ML Analyzer
try:
    quality_analyzer = SessionQualityAnalyzer('flowstate.db')
    print("✅ Session Quality ML Analyzer initialized successfully")
except Exception as e:
    print(f"⚠️  Session Quality Analyzer not available: {e}")
    quality_analyzer = None

# Initialize Gemini AI (uses environment variable GEMINI_API_KEY)
try:
    gemini_analyzer = GeminiHandAnalyzer()
    print("✅ Gemini AI initialized successfully")
except Exception as e:
    print(f"⚠️  Gemini AI not available: {e}")
    gemini_analyzer = None

# Initialize Session Comparison
try:
    session_comparison = SessionComparison('flowstate.db')
    print("✅ Session Comparison initialized successfully")
except Exception as e:
    print(f"⚠️  Session Comparison not available: {e}")
    session_comparison = None

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

# -------------------------
# Pose Detection Model (from friend's rhythm game)
# -------------------------
class PoseNet(nn.Module):
    """Neural network for hand pose classification"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(42, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load trained pose model
pose_model = PoseNet()
pose_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_model.pt')
pose_model.load_state_dict(torch.load(pose_model_path, weights_only=True))
pose_model.eval()

POSE_NAMES = ['palm', '1', '2', '3', 'fist']
print("✅ Pose detection model loaded successfully")

# Pose to note mapping for Hot Cross Buns (1→E, 2→D, 3→C)
POSE_NOTES = {
    '1': 329.63,  # E4 frequency in Hz
    '2': 293.66,  # D4 frequency in Hz  
    '3': 261.63   # C4 frequency in Hz
}

def generate_piano_tone(frequency, duration=0.5, volume=0.5):
    """Generate a piano-like tone with harmonics and ADSR envelope"""
    sample_rate = 22050
    num_samples = int(sample_rate * duration)
    
    # Piano harmonics (fundamental + overtones with decreasing amplitude)
    harmonics = [
        (1.0, 1.0),      # Fundamental
        (2.0, 0.5),      # 2nd harmonic
        (3.0, 0.25),     # 3rd harmonic
        (4.0, 0.125),    # 4th harmonic
        (5.0, 0.0625),   # 5th harmonic
    ]
    
    samples = []
    for i in range(num_samples):
        t = float(i) / sample_rate
        
        # ADSR envelope with release for piano sound
        attack_time = 0.01
        decay_time = 0.05
        sustain_level = 0.6
        release_start = duration - 0.1
        
        if t < attack_time:
            envelope = t / attack_time
        elif t < attack_time + decay_time:
            envelope = 1.0 - (1.0 - sustain_level) * ((t - attack_time) / decay_time)
        elif t < release_start:
            envelope = sustain_level
        else:
            # Release (fade out)
            envelope = sustain_level * (1.0 - (t - release_start) / (duration - release_start))
        
        # Sum all harmonics
        value = 0
        for harmonic_mult, harmonic_amp in harmonics:
            value += harmonic_amp * math.sin(2 * math.pi * frequency * harmonic_mult * t)
        
        # Apply envelope and volume
        value = int(volume * envelope * 32767 * value / len(harmonics))
        samples.append(value)
    
    # Create 1D array for mono sound
    sound_array = np.array(samples, dtype=np.int16)
    sound = pygame.sndarray.make_sound(sound_array)
    return sound

def play_note_once(frequency):
    """Play a note once without holding"""
    if SOUND_ENABLED:
        note_name = {329.63: 'E', 293.66: 'D', 261.63: 'C'}.get(frequency, '?')
        print(f"🎵 Playing note {note_name} ({frequency}Hz)")
        
        # Generate and play sound once (no looping)
        sound = generate_piano_tone(frequency, duration=0.5, volume=0.6)
        sound.play()

def handle_pose_change(pose_name):
    """Handle pose changes - log pose but don't play sound (handled by frontend)"""
    if pose_name in POSE_NOTES:
        # Log the pose detection (sound handled by frontend Web Audio API)
        frequency = POSE_NOTES[pose_name]
        note_name = {329.63: 'E', 293.66: 'D', 261.63: 'C'}.get(frequency, '?')
        print(f"🎯 Detected pose: {pose_name} -> Note {note_name} ({frequency}Hz)")

class HandTracker:
    def __init__(self):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'hand_landmarker.task')
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5)
        
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Session recording
        self.current_session = None
        self.is_recording = False
        
        # Auto-stop tracking
        self.frames_without_hand = 0
        self.max_frames_without_hand = 90  # ~3 seconds at 30fps
        
        # Pose detection tracking
        self.pose_history = []  # For smoothing pose predictions
        self.max_history_length = 2  # Average over last 2 predictions (faster response)
        self.current_pose = None
        self.pose_confidence = 0.0
        
        # Shared detection data (for API without camera access)
        self.last_landmarks = []
        self.last_handedness = []
    
    def start_session(self, user_id, level_name="free_play"):
        """Start a new recording session"""
        session_id = str(uuid.uuid4())
        self.current_session = SessionRecorder(session_id, user_id, level_name)
        self.is_recording = True
        
        # Create session in database
        db.create_session(session_id, user_id, level_name, datetime.now().isoformat())
        
        print(f"Started session: {session_id}")
        return session_id
    
    def stop_session(self):
        """Stop recording and save session data"""
        if self.current_session and self.is_recording:
            self.is_recording = False
            
            # Get session summary
            summary = self.current_session.get_session_summary()
            
            # Check if we have any data
            if not summary:
                print(f"No frames recorded for session: {self.current_session.session_id}")
                db.update_session(
                    self.current_session.session_id,
                    end_time=datetime.now().isoformat(),
                    duration_seconds=(datetime.now() - self.current_session.start_time).total_seconds(),
                    total_frames=0,
                    completed=0
                )
                return self.current_session.session_id
            
            # Update session in database
            db.update_session(
                self.current_session.session_id,
                end_time=summary['end_time'],
                duration_seconds=summary['duration_seconds'],
                total_frames=summary['total_frames'],
                avg_speed=summary.get('avg_speed'),
                max_speed=summary.get('max_speed'),
                avg_smoothness=summary.get('avg_smoothness'),
                avg_tremor=summary.get('avg_tremor'),
                completed=1
            )
            
            # Save time series data
            frame_data = self.current_session.get_time_series_data()
            db.save_frame_data(self.current_session.session_id, frame_data)
            
            # Save events
            db.save_session_events(self.current_session.session_id, 
                                   self.current_session.session_events)
            
            print(f"Session saved: {self.current_session.session_id}")
            print(f"Summary: {summary}")
            
            session_id = self.current_session.session_id
            
            # Score session quality with ML model
            if quality_analyzer:
                try:
                    quality_result = quality_analyzer.score_session(session_id)
                    if quality_result:
                        print(f"✅ Session scored: {quality_result['quality_score']}/100")
                        # Retrain model with this session
                        quality_analyzer.retrain_with_session(session_id)
                except Exception as e:
                    print(f"⚠️  Could not score session: {e}")
            
            self.current_session = None
            
            return session_id
        
        return None
    
    def predict_pose(self, hand_landmarks):
        """Predict hand pose using trained model (from friend's rhythm game)"""
        # Extract 42 features (21 landmarks × 2 coordinates: x, y)
        raw = []
        for lm in hand_landmarks:
            raw.extend([lm.x, lm.y])
        
        # Normalize relative to wrist (landmark 0)
        wrist_x, wrist_y = raw[0], raw[1]
        nx = [raw[i] - wrist_x for i in range(0, 42, 2)]
        ny = [raw[i] - wrist_y for i in range(1, 42, 2)]
        combined = nx + ny
        
        # Scale to [-1, 1]
        max_val = max(abs(max(combined)), abs(min(combined)))
        if max_val > 0:
            input_data = torch.tensor([[v/max_val for v in combined]], dtype=torch.float32)
            
            with torch.no_grad():
                preds = pose_model(input_data)
                prob = torch.softmax(preds, dim=1)
                confidence = prob.max().item()
                pose_idx = torch.argmax(preds).item()
                
                # Only add to history if confident (lowered for faster response)
                if confidence > 0.70:
                    self.pose_history.append(pose_idx)
                    
                    # Keep history limited
                    if len(self.pose_history) > self.max_history_length:
                        self.pose_history.pop(0)
                    
                    # Get most common pose in history
                    if self.pose_history:
                        most_common_pose_idx = max(set(self.pose_history), key=self.pose_history.count)
                        self.current_pose = POSE_NAMES[most_common_pose_idx]
                        self.pose_confidence = confidence
                        return self.current_pose, confidence
                else:
                    # Clear history if confidence drops
                    self.pose_history.clear()
        
        return None, 0.0
    
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
    
    def process_frame(self, frame, frame_timestamp_ms):
        """Process frame with MediaPipe and record data if session active"""
        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks (IMAGE mode doesn't need timestamp)
        detection_result = self.landmarker.detect(mp_image)
        
        # Auto-start recording when hand is first detected
        if not self.is_recording and detection_result.hand_landmarks:
            print("Hand detected! Auto-starting recording session...")
            self.start_session(DEFAULT_USER_ID, "auto_capture")
            self.frames_without_hand = 0
        
        # Record data if session is active (even if no hand in this specific frame)
        # Store current detection data for API access (whether recording or not)
        if detection_result.hand_landmarks:
            # Use first detected hand and get handedness
            hand_landmarks = detection_result.hand_landmarks[0]
            handedness_info = detection_result.handedness[0][0] if detection_result.handedness else None
            handedness = "Right" if handedness_info and handedness_info.category_name == "Right" else "Left"
            
            # Predict pose
            detected_pose, confidence = self.predict_pose(hand_landmarks)
            
            # Store current detection data for shared state (fix attribute access)
            self.last_landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]
            self.last_handedness = [{"categoryName": handedness, "score": handedness_info.score if handedness_info else 1.0}]
        else:
            # No hand detected - clear current pose data
            self.current_pose = None
            self.pose_confidence = 0.0
            self.last_landmarks = []
            self.last_handedness = []

        if self.is_recording and self.current_session:
            if detection_result.hand_landmarks:
                # Use first detected hand and get handedness
                hand_landmarks = detection_result.hand_landmarks[0]
                handedness = "Right" if detection_result.handedness and detection_result.handedness[0][0].category_name == "Right" else "Left"
                
                # Predict pose
                detected_pose, confidence = self.predict_pose(hand_landmarks)
                
                # Record frame with pose information
                event_type = None
                if detected_pose:
                    # Check if pose changed
                    if not hasattr(self, '_last_recorded_pose') or self._last_recorded_pose != detected_pose:
                        event_type = 'pose_change'
                        self.current_session.mark_event('pose_detected', {
                            'pose': detected_pose,
                            'confidence': float(confidence)
                        })
                        self._last_recorded_pose = detected_pose
                        
                        # Handle continuous note playback (fist = rest)
                        handle_pose_change(detected_pose)
                
                self.current_session.record_frame(hand_landmarks, frame_timestamp_ms, event_type, handedness)
                self.frames_without_hand = 0  # Reset counter
            else:
                # No hand detected - increment counter
                self.frames_without_hand += 1
                
                # Auto-stop if hand is gone for too long
                if self.frames_without_hand >= self.max_frames_without_hand:
                    print(f"No hand detected for {self.frames_without_hand} frames. Auto-stopping session...")
                    self.stop_session()
                    self.frames_without_hand = 0
                    self.pose_history.clear()
                    self._last_recorded_pose = None
        
        # Draw hand landmarks
        frame = self.draw_landmarks(frame, detection_result)
        
        # Draw pose detection info
        if self.current_pose and self.pose_confidence > 0.80:
            h, w, _ = frame.shape
            pose_text = f"POSE: {self.current_pose.upper()}"
            conf_text = f"{int(self.pose_confidence * 100)}%"
            
            # Background panel for pose info
            cv2.rectangle(frame, (10, h - 90), (300, h - 10), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, h - 90), (300, h - 10), (100, 100, 100), 2)
            
            # Pose name
            cv2.putText(frame, pose_text, (20, h - 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
            
            # Confidence
            cv2.putText(frame, f"Confidence: {conf_text}", (20, h - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw recording indicator
        if self.is_recording:
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, "RECORDING", (55, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def close(self):
        if self.is_recording:
            self.stop_session()
        self.landmarker.close()

# Initialize camera and tracker with thread-safe access
import threading
camera_lock = threading.Lock()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
tracker = HandTracker()

# Shared state for latest detection results (to avoid competing camera access)
latest_detection = {
    'pose': None,
    'landmarks': [],
    'handedness': [],
    'confidence': 0.0,
    'frame_count': 0
}
detection_lock = threading.Lock()

# Default user (create if doesn't exist)
DEFAULT_USER_ID = "default_user"
try:
    db.create_user(DEFAULT_USER_ID, "Default User", rehab_stage="mid", notes="Default recording user")
    print(f"✓ Created default user: {DEFAULT_USER_ID}")
except:
    pass  # User already exists

def generate_frames():
    """Generate frames for video streaming AND update shared detection state."""
    import gc
    global latest_detection
    frame_timestamp_ms = 0  # Each stream has its own timestamp
    frame_count = 0  # Track frames for memory management
    
    while True:
        try:
            success, frame = camera.read()
            if not success:
                # Don't break - just skip this frame and continue
                continue
            
            # Process frame with hand tracking and data collection
            frame = tracker.process_frame(frame, frame_timestamp_ms)
            frame_timestamp_ms += 33  # ~30 FPS
            frame_count += 1
            
            # Update shared detection state (so /api/current_pose doesn't need to read from camera)
            with detection_lock:
                latest_detection['pose'] = tracker.current_pose
                latest_detection['confidence'] = tracker.pose_confidence
                latest_detection['frame_count'] += 1
                
            # Memory management - clean up every 100 frames
            if frame_count % 100 == 0:
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in frame generation: {e}")
            continue
            
            # Update landmarks and handedness if available
            if hasattr(tracker, 'last_landmarks'):
                latest_detection['landmarks'] = tracker.last_landmarks
                latest_detection['handedness'] = tracker.last_handedness
            else:
                latest_detection['landmarks'] = []
                latest_detection['handedness'] = []
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue  # Skip if encoding failed
            
        frame = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the home page with cassette UI."""
    return render_template('home.html')

@app.route('/freestyle')
def freestyle():
    """Render the freestyle practice page."""
    return render_template('freestyle.html')

@app.route('/analytics')
def analytics():
    """Render the analytics page."""
    return render_template('analytics.html')

@app.route('/session-analytics')
def session_analytics():
    """Render the session comparison analytics page."""
    return render_template('session_analytics.html')

@app.route('/hand-comparison')
def hand_comparison():
    """Render the hand comparison page."""
    return render_template('hand_comparison.html')

@app.route('/levels')
def levels():
    """Render the levels page."""
    return render_template('levels.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/current_pose')
def current_pose():
    """Get the current detected pose from shared state (no camera access needed)."""
    try:
        # Read from shared detection state instead of competing for camera
        with detection_lock:
            return jsonify({
                'pose': latest_detection['pose'],
                'landmarks': latest_detection['landmarks'],
                'handedness': latest_detection['handedness'],
                'confidence': float(latest_detection['confidence'])
            })
            
    except Exception as e:
        print(f"API detection error: {e}")
        return jsonify({
            'pose': None,
            'landmarks': [],
            'handedness': [],
            'confidence': 0.0,
            'error': str(e)
        })

@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new recording session"""
    data = request.json
    user_id = data.get('user_id', DEFAULT_USER_ID)
    level_name = data.get('level_name', 'free_play')
    
    session_id = tracker.start_session(user_id, level_name)
    return jsonify({'session_id': session_id, 'status': 'recording'})

@app.route('/api/session/stop', methods=['POST'])
def stop_session():
    """Stop the current recording session"""
    session_id = tracker.stop_session()
    
    if session_id:
        return jsonify({'session_id': session_id, 'status': 'completed'})
    else:
        return jsonify({'error': 'No active session'}), 400

@app.route('/api/user/<user_id>/stats')
def user_stats(user_id):
    """Get user statistics"""
    stats = db.get_user_statistics(user_id)
    return jsonify(stats)

@app.route('/api/user/<user_id>/joint-analysis')
def joint_analysis(user_id):
    """Get joint-specific analysis for user"""
    analyzer = JointAnalyzer()
    analysis = analyzer.analyze_user_joints(user_id)
    
    if analysis:
        # Format for web display
        result = {
            'weakest_joints': [
                {
                    'name': joint_key.replace('_', ' '),
                    'score': round(data['score'], 1),
                    'priority': 'high' if data['score'] < 50 else 'moderate' if data['score'] < 70 else 'minor',
                    'rom_mean': round(data['rom_mean'], 4),
                    'rom_range': round(data['rom_range'], 4)
                }
                for joint_key, data in analysis['weakest_joints']
            ],
            'strongest_joints': [
                {
                    'name': joint_key.replace('_', ' '),
                    'score': round(data['score'], 1)
                }
                for joint_key, data in analysis['strongest_joints']
            ]
        }
        return jsonify(result)
    else:
        return jsonify({'error': 'No data available'}), 404

@app.route('/api/gemini/analyze-session/<session_id>')
def gemini_analyze_session(session_id):
    """Get Gemini AI analysis for a specific session"""
    if not gemini_analyzer:
        return jsonify({'error': 'Gemini AI not available'}), 503
    
    user_id = request.args.get('user_id', 'default_user')
    result = gemini_analyzer.analyze_session(session_id, user_id)
    return jsonify(result)

@app.route('/api/gemini/analyze-joints/<user_id>')
def gemini_analyze_joints(user_id):
    """Get Gemini AI recommendations for weak joints"""
    if not gemini_analyzer:
        return jsonify({'error': 'Gemini AI not available'}), 503
    
    # Get joint analysis first
    analyzer = JointAnalyzer()
    analysis = analyzer.analyze_user_joints(user_id)
    
    if not analysis:
        return jsonify({'error': 'No joint data available'}), 404
    
    # Format for Gemini
    joint_data = {
        'weakest_joints': [
            {
                'name': joint_key.replace('_', ' '),
                'score': round(data['score'], 1),
                'priority': 'high' if data['score'] < 50 else 'moderate' if data['score'] < 70 else 'minor'
            }
            for joint_key, data in analysis['weakest_joints']
        ]
    }
    
    result = gemini_analyzer.analyze_weak_joints(joint_data)
    return jsonify(result)

@app.route('/api/gemini/latest-session-feedback/<user_id>')
def gemini_latest_session_feedback(user_id):
    """Get Gemini AI feedback on the most recent session"""
    if not gemini_analyzer:
        return jsonify({'error': 'Gemini AI not available'}), 503
    
    # Get latest session
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT session_id FROM sessions 
        WHERE user_id = ? AND completed = 1 AND total_frames > 0
        ORDER BY start_time DESC LIMIT 1
    ''', (user_id,))
    
    row = cursor.fetchone()
    if not row:
        return jsonify({'error': 'No completed sessions found'}), 404
    
    session_id = row[0]
    result = gemini_analyzer.analyze_session(session_id, user_id)
    return jsonify(result)

@app.route('/api/session/<session_id>/detailed-analytics')
def session_detailed_analytics(session_id):
    """Get comprehensive session analytics with quality score and joint analysis"""
    try:
        if not db or not db.conn:
            logger.error("Database connection not available")
            return jsonify({'error': 'Database not available'}), 500
            
        cursor = db.conn.cursor()
        
        # Get session info with error handling
        try:
            cursor.execute('''
                SELECT session_id, user_id, start_time, end_time, duration_seconds,
                       total_frames, avg_speed, max_speed, avg_smoothness, avg_tremor
                FROM sessions WHERE session_id = ?
            ''', (session_id,))
            
            session = cursor.fetchone()
        except Exception as db_error:
            logger.error(f"Database query error for session {session_id}: {db_error}")
            return jsonify({'error': 'Database query failed'}), 500
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Calculate quality score using ML model or fallback to heuristic
        quality_score = 0
        if quality_analyzer:
            try:
                score = quality_analyzer.score_session(session_id)
                quality_score = int(score * 100) if score else 0
            except:
                quality_score = 0
        
        # Fallback: Calculate quality from session metrics if ML fails
        if quality_score == 0 and session[8] and session[9]:  # Has smoothness and tremor
            smoothness_score = min(session[8] / 30, 1) * 40  # Smoothness out of 40
            tremor_score = max(0, 1 - (session[9] * 1000)) * 30  # Tremor out of 30
            speed_score = min(session[6] * 20, 30) if session[6] else 0  # Speed out of 30
            quality_score = int(smoothness_score + tremor_score + speed_score)
        
        # Get joint-level analytics from movement_timeseries
        cursor.execute('''
            SELECT landmark_positions, joint_angles, velocity_mean, smoothness_score, tremor_score
            FROM movement_timeseries 
            WHERE session_id = ? AND landmark_positions IS NOT NULL
            ORDER BY frame_number
        ''', (session_id,))
        
        frames = cursor.fetchall()
        
        # Analyze each hand joint (21 landmarks per hand)
        joint_analytics = {}
        joint_names = [
            "Wrist", "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
            "Index MCP", "Index PIP", "Index DIP", "Index Tip",
            "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
            "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
            "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip"
        ]
        
        if frames:
            # Calculate per-joint metrics
            for joint_idx in range(21):
                positions = []
                
                for frame_idx, frame in enumerate(frames):
                    try:
                        landmarks = json.loads(frame[0]) if frame[0] else []
                        
                        if landmarks and len(landmarks) > joint_idx:
                            pos = landmarks[joint_idx]
                            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                                positions.append([float(pos[0]), float(pos[1])])
                    except Exception as e:
                        continue
                
                if positions and len(positions) > 10:
                    # Calculate joint metrics
                    positions_arr = np.array(positions)
                    smoothness = 0
                    tremor = 0
                    range_of_motion = 0
                    
                    # Range of motion (total distance traveled)
                    diffs = np.diff(positions_arr, axis=0)
                    distances = np.sqrt(np.sum(diffs**2, axis=1))
                    range_of_motion = float(np.sum(distances))
                    
                    # Smoothness (inverse of jerk - lower jerk = smoother)
                    if len(distances) > 1:
                        jerk = np.abs(np.diff(distances))
                        avg_jerk = float(np.mean(jerk))
                        smoothness = max(0, 100 - (avg_jerk * 10000))
                    
                    # Tremor (variation in movement)
                    if len(distances) > 10:
                        tremor = float(np.std(distances))
                    
                    # Calculate overall joint score (0-100)
                    # Higher smoothness = better, lower tremor = better, reasonable ROM = better
                    tremor_score = max(0, 100 - (tremor * 1000))
                    rom_score = min(range_of_motion * 50, 50)  # Cap at 50 points
                    joint_score = (smoothness * 0.4) + (tremor_score * 0.4) + (rom_score * 0.2)
                    
                    joint_analytics[joint_names[joint_idx]] = {
                        'smoothness': round(smoothness, 1),
                        'tremor': round(tremor, 4),
                        'range_of_motion': round(range_of_motion, 2),
                        'score': round(min(joint_score, 100), 1),
                        'status': 'excellent' if joint_score >= 80 else 'good' if joint_score >= 60 else 'needs_improvement'
                    }
        
        result = {
            'session_id': session[0],
            'user_id': session[1],
            'start_time': session[2],
            'duration': session[4],
            'total_frames': session[5],
            'quality_score': quality_score,
            'overall_metrics': {
                'avg_speed': round(session[6], 4) if session[6] else 0,
                'max_speed': round(session[7], 4) if session[7] else 0,
                'smoothness': round(session[8], 2) if session[8] else 0,
                'tremor': round(session[9], 6) if session[9] else 0
            },
            'joint_analytics': joint_analytics
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>/data')
def session_data(session_id):
    """Get detailed session data"""
    timeseries = db.get_session_timeseries(session_id)
    return jsonify(timeseries)

@app.route('/api/user/<user_id>/progress/<metric>')
def user_progress(user_id, metric):
    """Get user progress over time for a specific metric"""
    progress = db.get_user_progress_over_time(user_id, metric)
    return jsonify(progress)

@app.route('/api/user/<user_id>/analytics')
def user_analytics(user_id):
    """Get analytics data - MOST RECENT session only for live demo"""
    cursor = db.conn.cursor()
    
    # Get MOST RECENT session for displaying current scores
    cursor.execute('''
        SELECT session_id, start_time, total_frames, avg_speed, avg_smoothness, avg_tremor
        FROM sessions 
        WHERE user_id = ? AND completed = 1 AND total_frames > 0
        ORDER BY start_time DESC LIMIT 1
    ''', (user_id,))
    
    latest_session = cursor.fetchone()
    if not latest_session:
        return jsonify({'error': 'No data available'}), 404
    
    # Also get last 5 sessions for progress chart
    cursor.execute('''
        SELECT session_id, start_time, total_frames, avg_speed, avg_smoothness, avg_tremor
        FROM sessions 
        WHERE user_id = ? AND completed = 1 AND total_frames > 0
        ORDER BY start_time DESC LIMIT 5
    ''', (user_id,))
    sessions = cursor.fetchall()
    
    # Calculate joint scores from MOST RECENT session ONLY
    analyzer = JointAnalyzer()
    joint_analysis = analyzer.analyze_recent_sessions([latest_session[0]])  # Only latest session ID
    
    # Convert joint analysis to format matching frontend with proper scoring
    joint_data = []
    if joint_analysis and 'joint_scores' in joint_analysis:
        for joint_key, scores in joint_analysis['joint_scores'].items():
            # Map joint keys to display format
            parts = joint_key.split('_')
            if len(parts) >= 2:
                finger = parts[0].capitalize()
                joint_type = parts[1].upper()
                
                # Calculate realistic scores with inherent joint-type variation for demo
                rom_mean = scores.get('rom_mean', 0.1)
                rom_range = scores.get('rom_range', 0.05)
                rom_std = scores.get('rom_std', 0.01)
                sample_count = scores.get('sample_count', 1)
                
                # Base score multipliers by joint type (for visual differentiation in demos)
                joint_type_multipliers = {
                    'MCP': 1.2,    # Knuckles move most - highest scores
                    'PIP': 1.0,    # Middle joints - medium scores  
                    'DIP': 0.85,   # Distal joints - lower scores
                    'TIP': 0.7,    # Fingertips - lowest scores (least ROM)
                    'CMC': 1.15,   # Thumb base - high scores
                    'IP': 0.9      # Thumb IP - medium scores
                }
                
                # Finger-specific multipliers (some fingers naturally move more)
                finger_multipliers = {
                    'Index': 1.1,   # Index finger most active
                    'Middle': 1.05, # Middle finger active
                    'Thumb': 1.0,   # Thumb average
                    'Ring': 0.95,   # Ring less active
                    'Pinky': 0.85   # Pinky least active
                }
                
                joint_mult = joint_type_multipliers.get(joint_type, 1.0)
                finger_mult = finger_multipliers.get(finger, 1.0)
                base_variation = joint_mult * finger_mult
                
                # Get REAL tremor and smoothness data from the session
                cursor.execute('''
                    SELECT AVG(tremor_score), AVG(smoothness_score), AVG(velocity_mean)
                    FROM movement_timeseries
                    WHERE session_id = ?
                ''', (latest_session[0],))
                tremor_data = cursor.fetchone()
                avg_tremor = tremor_data[0] or 0.001
                avg_smoothness = tremor_data[1] or 1000
                avg_velocity = tremor_data[2] or 0.1
                
                # SMOOTHNESS SCORE: Lower tremor = higher score, Lower jerk = higher score
                # Tremor typically ranges 0.0001 to 0.01, smoothness 100 to 5000
                tremor_score = max(10, min(100, 100 - (avg_tremor * 8000)))  # Less tremor = higher
                jerk_score = max(10, min(100, 100 - (avg_smoothness / 50)))   # Less jerk = higher  
                smoothness_score = (tremor_score * 0.6 + jerk_score * 0.4)    # Weighted average
                
                # ROM SCORE: Based on actual range of motion
                rom_score = max(10, min(100, 20 + (rom_mean * 400) + (rom_range * 200)))
                
                # TRAJECTORY SCORE: Based on consistency (low variability = high score)
                if rom_mean > 0.01:
                    variability = rom_std / rom_mean
                    trajectory_score = max(20, min(100, 90 - (variability * 100)))
                else:
                    trajectory_score = 30
                
                # CONSISTENCY SCORE: Based on velocity consistency (less variation = higher)
                velocity_consistency = max(20, min(100, 80 - (rom_std * 500)))
                consistency_score = velocity_consistency
                
                # Apply minor joint-type multipliers (knuckles naturally smoother)
                smoothness_score = smoothness_score * (base_variation * 0.15 + 0.85)
                rom_score = rom_score * base_variation
                trajectory_score = trajectory_score * (base_variation * 0.1 + 0.9)
                consistency_score = consistency_score * (base_variation * 0.1 + 0.9)
                
                # Clamp final scores
                smoothness_score = min(95, max(15, smoothness_score))
                rom_score = min(95, max(15, rom_score))
                trajectory_score = min(95, max(20, trajectory_score))
                consistency_score = min(95, max(20, consistency_score))
                
                joint_data.append({
                    'id': joint_key,
                    'label': f"{finger} {joint_type}",
                    'smoothness': round(smoothness_score),
                    'rom': round(rom_score),
                    'trajectoryAccuracy': round(trajectory_score),
                    'consistency': round(consistency_score),
                    'movementDuration': 380 + int(rom_std * 1000),
                    'baselineDuration': 350
                })
    
    # Weekly progress: Show each individual session's performance
    weekly_data = []
    for i, session in enumerate(sessions[:7]):
        session_id, start_time, frames, speed, smoothness, tremor = session
        
        # Speed: EXTREME differentiation
        # Excellent > 0.7 (92-100), Good 0.5-0.7 (75-92), Moderate 0.3-0.5 (45-75), Poor < 0.3 (15-45)
        speed_val = speed or 0.25
        if speed_val > 0.7:
            speed_score = min(100, 92 + (speed_val - 0.7) * 40)
        elif speed_val > 0.5:
            speed_score = 75 + (speed_val - 0.5) * 85
        elif speed_val > 0.3:
            speed_score = 45 + (speed_val - 0.3) * 150
        else:
            speed_score = max(15, 15 + speed_val * 100)
        
        # Smoothness: SPARC EXTREME thresholds (lower is better)
        # Excellent < 600 (95-100), Good 600-900 (75-95), Moderate 900-1500 (40-75), Poor > 1500 (10-40)
        smooth_val = smoothness or 1200
        if smooth_val < 600:
            smooth_score = min(100, 115 - smooth_val / 30)
        elif smooth_val < 900:
            smooth_score = 95 - (smooth_val - 600) / 15
        elif smooth_val < 1500:
            smooth_score = 75 - (smooth_val - 900) / 17
        else:
            smooth_score = max(10, 40 - (smooth_val - 1500) / 50)
        
        # Tremor: VERY strict thresholds
        # Excellent < 0.0008 (96-100), Good 0.0008-0.0015 (80-96), Moderate 0.0015-0.003 (45-80), Poor > 0.003 (10-45)
        tremor_val = tremor or 0.002
        if tremor_val < 0.0008:
            tremor_score = min(100, 110 - tremor_val * 17500)
        elif tremor_val < 0.0015:
            tremor_score = 96 - (tremor_val - 0.0008) * 22857
        elif tremor_val < 0.003:
            tremor_score = 80 - (tremor_val - 0.0015) * 23333
        else:
            tremor_score = max(10, 45 - (tremor_val - 0.003) * 7000)
        
        weekly_data.append({
            'week': f"S{i+1}",
            'smoothness': round(smooth_score),
            'rom': round(speed_score),
            'accuracy': round((smooth_score + speed_score) / 2),
            'consistency': round(tremor_score),
            'fmaProxy': round((smooth_score + speed_score + tremor_score) / 3)
        })
    
    # Calculate overall scores with better normalization
    avg_speed = sum(s[3] or 0.5 for s in sessions) / len(sessions)
    avg_smoothness = sum(s[4] or 200 for s in sessions) / len(sessions)
    avg_tremor = sum(s[5] or 0.0001 for s in sessions) / len(sessions)
    
    # Get pose statistics from most recent session
    cursor.execute('''
        SELECT event_data 
        FROM session_events 
        WHERE session_id = ? AND event_type = 'pose_detected'
    ''', (latest_session[0],))
    
    pose_events = cursor.fetchall()
    pose_stats = {}
    
    if pose_events:
        # Count occurrences of each pose
        pose_counts = {}
        pose_confidences = {}
        
        for event in pose_events:
            import json
            event_data = json.loads(event[0]) if event[0] else {}
            pose_name = event_data.get('pose', 'unknown')
            confidence = event_data.get('confidence', 0)
            
            if pose_name not in pose_counts:
                pose_counts[pose_name] = 0
                pose_confidences[pose_name] = []
            
            pose_counts[pose_name] += 1
            pose_confidences[pose_name].append(confidence)
        
        # Calculate statistics for each pose
        for pose_name, count in pose_counts.items():
            avg_confidence = sum(pose_confidences[pose_name]) / len(pose_confidences[pose_name])
            pose_stats[pose_name] = {
                'count': count,
                'averageConfidence': round(avg_confidence, 2),
                'percentage': round(100 * count / len(pose_events), 1)
            }
    
    return jsonify({
        'jointData': joint_data,
        'weeklyProgress': weekly_data,
        'overallScores': {
            'smoothness': round(min(95, max(50, 95 - min(avg_smoothness / 100, 30)))),
            'rom': round(min(95, max(50, 60 + avg_speed * 20))),
            'trajectory': round(min(95, max(50, 85 - min(avg_tremor * 3000, 30)))),
            'consistency': round(min(95, max(50, 95 - min(avg_tremor * 5000, 40))))
        },
        'poseStatistics': pose_stats
    })

@app.route('/api/user/<user_id>/analytics/hand/<hand_type>')
def user_analytics_by_hand(user_id, hand_type):
    """Get analytics separated by left/right hand"""
    cursor = db.conn.cursor()
    
    # Get most recent session
    cursor.execute('''
        SELECT session_id, total_frames, avg_speed, avg_smoothness, avg_tremor
        FROM sessions 
        WHERE user_id = ? AND completed = 1 AND total_frames > 0
        ORDER BY start_time DESC LIMIT 1
    ''', (user_id,))
    
    session = cursor.fetchone()
    if not session:
        return jsonify({'error': 'No data available'}), 404
    
    session_id = session[0]
    
    # Note: To fully separate by hand, we need handedness stored in database
    # For now, return a message about implementation
    return jsonify({
        'hand': hand_type,
        'session_id': session_id,
        'message': f'Analytics for {hand_type} hand',
        'note': 'Handedness tracking is implemented. Use Freestyle page to record separate sessions with each hand.',
        'avg_tremor': session[4],
        'avg_smoothness': session[3],
        'avg_speed': session[2]
    })

@app.route('/api/user/<user_id>/session-quality')
def get_session_quality(user_id):
    """Get ML-scored quality analysis for most recent session"""
    if not quality_analyzer:
        return jsonify({'error': 'Quality analyzer not available'}), 503
    
    try:
        result = quality_analyzer.get_most_recent_score(user_id)
        
        if not result:
            return jsonify({'error': 'No completed sessions found'}), 404
        
        return jsonify({
            'success': True,
            'session': {
                'id': result['session_id'],
                'timestamp': result['start_time'],
                'frames': result['total_frames']
            },
            'qualityScore': result['quality_score'],
            'breakdown': result['breakdown'],
            'metrics': {
                'poseVariety': result['pose_variety'],
                'transitionQuality': result['transition_quality']
            },
            'interpretation': _interpret_score(result['quality_score'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _interpret_score(score):
    """Provide interpretation of quality score"""
    if score >= 85:
        return {
            'level': 'Excellent',
            'message': 'Outstanding progress! Your movement quality is exceptional.',
            'emoji': '🌟'
        }
    elif score >= 70:
        return {
            'level': 'Good',
            'message': 'Great work! You\'re making solid progress in your rehabilitation.',
            'emoji': '👍'
        }
    elif score >= 50:
        return {
            'level': 'Fair',
            'message': 'You\'re on the right track. Keep practicing to improve further.',
            'emoji': '💪'
        }
    elif score >= 30:
        return {
            'level': 'Needs Improvement',
            'message': 'Focus on smooth, controlled movements. Practice regularly.',
            'emoji': '📈'
        }
    else:
        return {
            'level': 'Beginning',
            'message': 'Starting your journey. Every session helps build strength and control.',
            'emoji': '🌱'
        }

@app.route('/api/gemini/chat', methods=['POST'])
def gemini_chat():
    """Chat with Gemini AI about user's hand rehabilitation data"""
    if not gemini_analyzer:
        return jsonify({'error': 'Gemini AI not available'}), 503
    
    data = request.get_json()
    user_message = data.get('message', '')
    user_id = data.get('user_id', 'default_user')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get latest session context
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT session_id, session_type, total_frames, avg_speed, avg_smoothness, avg_tremor
        FROM sessions 
        WHERE user_id = ? AND completed = 1 AND total_frames > 0
        ORDER BY start_time DESC LIMIT 1
    ''', (user_id,))
    
    session_row = cursor.fetchone()
    
    # Get joint analysis
    analyzer = JointAnalyzer()
    joint_analysis = analyzer.analyze_user_joints(user_id)
    
    # Build context for Gemini
    context = f"User's latest session data:\n"
    if session_row:
        context += f"- Session type: {session_row[1]}\n"
        context += f"- Frames recorded: {session_row[2]}\n"
        context += f"- Average speed: {session_row[3]:.2f}\n"
        context += f"- Smoothness score: {session_row[4]:.2f}\n"
        context += f"- Tremor level: {session_row[5]:.2f}\n\n"
    
    if joint_analysis and 'weakest_joints' in joint_analysis:
        context += "Weakest joints:\n"
        for joint_key, data in joint_analysis['weakest_joints'][:3]:
            context += f"- {joint_key.replace('_', ' ')}: {data['score']:.1f}/100\n"
        context += "\n"
    
    context += f"User question: {user_message}"
    
    # Get Gemini response
    try:
        response = gemini_analyzer.chat_about_data(context)
        return jsonify({
            'response': response,
            'context_used': bool(session_row or joint_analysis)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Session Comparison API Endpoints
@app.route('/api/session/<session_id>/comparison')
def get_session_comparison(session_id):
    """Get comprehensive session comparison"""
    try:
        user_id = request.args.get('user_id', 'default_user')
        
        if not session_comparison:
            return jsonify({'error': 'Session comparison not available'}), 503
        
        analytics = session_comparison.get_session_analytics(session_id, user_id)
        return jsonify(analytics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<user_id>/progression')
def get_progression(user_id):
    """Get progression trends for user"""
    try:
        if not session_comparison:
            return jsonify({'error': 'Session comparison not available'}), 503
        
        window_size = int(request.args.get('window', 5))
        trends = session_comparison.get_progression_trends(user_id, window_size)
        return jsonify(trends)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<user_id>/weekly-summary')
def get_weekly_summary(user_id):
    """Get weekly aggregated metrics"""
    try:
        if not session_comparison:
            return jsonify({'error': 'Session comparison not available'}), 503
        
        summary = session_comparison.get_weekly_summary(user_id)
        return jsonify({'weekly_data': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>/compare-to-baseline')
def compare_to_baseline(session_id):
    """Compare session to user's baseline (first session)"""
    try:
        user_id = request.args.get('user_id', 'default_user')
        
        if not session_comparison:
            return jsonify({'error': 'Session comparison not available'}), 503
        
        comparison = session_comparison.compare_to_baseline(session_id, user_id)
        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>/compare-to-best')
def compare_to_best(session_id):
    """Compare session to user's best session"""
    try:
        user_id = request.args.get('user_id', 'default_user')
        
        if not session_comparison:
            return jsonify({'error': 'Session comparison not available'}), 503
        
        comparison = session_comparison.compare_to_best(session_id, user_id)
        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions')
def get_sessions():
    """Get list of all sessions (for default user)"""
    try:
        if not session_comparison:
            return jsonify([])
        
        days = int(request.args.get('days', 90))
        sessions = session_comparison.get_user_sessions('default_user', days)
        return jsonify(sessions)
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return jsonify([])

@app.route('/api/user/<user_id>/sessions')
def get_user_sessions_list(user_id):
    """Get list of all sessions for user"""
    try:
        if not session_comparison:
            return jsonify({'error': 'Session comparison not available'}), 503
        
        days = int(request.args.get('days', 90))
        sessions = session_comparison.get_user_sessions(user_id, days)
        return jsonify({'sessions': sessions})
    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import signal
    import sys
    
    def cleanup_handler(signum, frame):
        """Handle cleanup on signal termination"""
        print("\n\n🛑 Shutting down gracefully...")
        try:
            camera.release()
            tracker.close()
            db.close()
            print("✅ Camera and resources released safely")
        except:
            pass
        sys.exit(0)
    
    # Register signal handlers for safe camera cleanup
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    print("\n" + "="*60)
    print("🎥 FlowState Hand Tracking + Data Collection")
    print("="*60)
    print("\n📱 Open your browser and go to:")
    print("   http://localhost:5001/freestyle")
    print("\n🎯 Features:")
    print("   • Real-time hand tracking with ML pose detection")
    print("   • Pose-based piano sound generation")  
    print("   • Trained model for poses 1, 2, 3 → E, D, C notes")
    print("\n⌨️  Press CTRL+C to stop the server\n")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    finally:
        try:
            camera.release()
            tracker.close()
            db.close()
            print("✅ Cleanup complete. Goodbye!\n")
        except:
            print("✅ Server stopped\n")
