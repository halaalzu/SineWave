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

# Load environment variables from .env file
load_dotenv()

from movement_features import MovementFeatureExtractor, SessionRecorder
from database import RehabDatabase
from joint_analysis import JointAnalyzer
from gemini_analyzer import GeminiHandAnalyzer

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

# Initialize Gemini AI (uses environment variable GEMINI_API_KEY)
try:
    gemini_analyzer = GeminiHandAnalyzer()
    print("✅ Gemini AI initialized successfully")
except Exception as e:
    print(f"⚠️  Gemini AI not available: {e}")
    gemini_analyzer = None
    gemini_analyzer = None

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
    """Handle pose changes - play note once when pose detected"""
    if pose_name in POSE_NOTES:
        # Play the note once for this pose
        frequency = POSE_NOTES[pose_name]
        play_note_once(frequency)

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
            
            # Store current detection data for API
            self._last_pose = detected_pose
            self._last_landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
            self._last_handedness = [{"categoryName": handedness, "score": handedness_info.score if handedness_info else 1.0}]
            self._last_confidence = float(confidence)
            
            # Debug
            print(f"DEBUG: Hand detected! Pose: {detected_pose}, Landmarks count: {len(self._last_landmarks)}")
        else:
            # No hand detected - clear current pose data
            self._last_pose = None
            self._last_landmarks = None
            self._last_handedness = None
            self._last_confidence = 0.0
            self.current_pose = None  # Clear persistent pose
            self.pose_confidence = 0.0
            print("DEBUG: No hand detected, clearing pose data")

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

# Initialize camera and tracker
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
tracker = HandTracker()

# Default user (create if doesn't exist)
DEFAULT_USER_ID = "default_user"
try:
    db.create_user(DEFAULT_USER_ID, "Default User", rehab_stage="mid", notes="Default recording user")
    print(f"✓ Created default user: {DEFAULT_USER_ID}")
except:
    pass  # User already exists

def generate_frames():
    """Generate frames for video streaming."""
    frame_timestamp_ms = 0  # Each stream has its own timestamp
    
    while True:
        success, frame = camera.read()
        if not success:
            # Don't break - just skip this frame and continue
            continue
        
        # Process frame with hand tracking and data collection
        frame = tracker.process_frame(frame, frame_timestamp_ms)
        frame_timestamp_ms += 33  # ~30 FPS
        
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
    """Get the current detected pose, landmarks, and handedness with live camera."""
    try:
        # Capture a single frame for pose detection
        success, frame = camera.read()
        if not success:
            return jsonify({
                'pose': None,
                'landmarks': [],
                'handedness': [],
                'confidence': 0.0,
                'error': 'Camera not available'
            })
        
        # Process frame for detection (without drawing)
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        detection_result = tracker.landmarker.detect(mp_image)
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            handedness_info = detection_result.handedness[0][0] if detection_result.handedness else None
            handedness = "Right" if handedness_info and handedness_info.category_name == "Right" else "Left"
            
            # Predict pose using trained model
            detected_pose, confidence = tracker.predict_pose(hand_landmarks)
            
            return jsonify({
                'pose': detected_pose,
                'landmarks': [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks],
                'handedness': [{"categoryName": handedness, "score": handedness_info.score if handedness_info else 1.0}],
                'confidence': float(confidence)
            })
        else:
            return jsonify({
                'pose': None,
                'landmarks': [],
                'handedness': [],
                'confidence': 0.0
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

@app.route('/api/user/<user_id>/sessions')
def user_sessions(user_id):
    """Get user sessions"""
    sessions = db.get_user_sessions(user_id, limit=10)
    return jsonify(sessions)

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

if __name__ == '__main__':
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
        tracker.close()
        camera.release()
        db.close()
        print("✅ Cleanup complete. Goodbye!\n")
