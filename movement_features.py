"""
Movement Data Collection Module
Captures and computes hand movement features for rehabilitation tracking
"""

import numpy as np
from datetime import datetime
from collections import deque
import json

class MovementFeatureExtractor:
    """Extract features from hand landmark data for analysis"""
    
    def __init__(self, window_size=30, fps=30):
        """
        Args:
            window_size: Number of frames to keep in history for velocity/acceleration
            fps: Frames per second for time calculations
        """
        self.window_size = window_size
        self.fps = fps
        self.frame_time = 1.0 / fps
        
        # History buffers for temporal features
        self.landmark_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)
        
    def compute_features(self, hand_landmarks, timestamp_ms):
        """
        Compute comprehensive movement features from hand landmarks
        
        Args:
            hand_landmarks: MediaPipe hand landmarks (21 points)
            timestamp_ms: Current timestamp in milliseconds
            
        Returns:
            dict: Dictionary of computed features
        """
        if hand_landmarks is None:
            return None
        
        features = {}
        
        # Store current frame
        landmarks_array = self._landmarks_to_array(hand_landmarks)
        self.landmark_history.append(landmarks_array)
        self.timestamp_history.append(timestamp_ms)
        
        # Basic position features
        features['landmark_positions'] = landmarks_array.tolist()
        features['timestamp_ms'] = timestamp_ms
        
        # Compute joint angles
        features['joint_angles'] = self._compute_joint_angles(landmarks_array)
        
        # Compute hand openness (spread)
        features['hand_openness'] = self._compute_hand_openness(landmarks_array)
        
        # Temporal features (need history)
        if len(self.landmark_history) >= 2:
            features['velocity'] = self._compute_velocity()
            features['acceleration'] = self._compute_acceleration()
            features['smoothness'] = self._compute_smoothness()
            features['tremor_score'] = self._compute_tremor()
        
        # Range of motion
        if len(self.landmark_history) >= 10:
            features['range_of_motion'] = self._compute_range_of_motion()
        
        return features
    
    def _landmarks_to_array(self, hand_landmarks):
        """Convert MediaPipe landmarks to numpy array"""
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    def _compute_joint_angles(self, landmarks):
        """Compute key joint angles (in degrees)"""
        angles = {}
        
        # Finger tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
        # MCP joints: thumb=2, index=5, middle=9, ring=13, pinky=17
        
        # Thumb angle (relative to palm)
        angles['thumb_angle'] = self._angle_between_points(
            landmarks[0], landmarks[2], landmarks[4]
        )
        
        # Index finger angle
        angles['index_angle'] = self._angle_between_points(
            landmarks[5], landmarks[6], landmarks[8]
        )
        
        # Middle finger angle
        angles['middle_angle'] = self._angle_between_points(
            landmarks[9], landmarks[10], landmarks[12]
        )
        
        # Ring finger angle
        angles['ring_angle'] = self._angle_between_points(
            landmarks[13], landmarks[14], landmarks[16]
        )
        
        # Pinky angle
        angles['pinky_angle'] = self._angle_between_points(
            landmarks[17], landmarks[18], landmarks[20]
        )
        
        # Wrist flexion (using wrist and knuckles)
        angles['wrist_flexion'] = self._angle_between_points(
            landmarks[0], landmarks[9], landmarks[5]
        )
        
        return angles
    
    def _angle_between_points(self, p1, p2, p3):
        """Calculate angle at point p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _compute_hand_openness(self, landmarks):
        """Compute how open/closed the hand is (0=closed, 1=open)"""
        # Distance between fingertips
        distances = []
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[j]])
                distances.append(dist)
        
        # Normalize to 0-1 range (approximate)
        avg_distance = np.mean(distances)
        openness = np.clip(avg_distance * 2, 0, 1)
        
        return float(openness)
    
    def _compute_velocity(self):
        """Compute velocity for each landmark"""
        if len(self.landmark_history) < 2:
            return None
        
        current = self.landmark_history[-1]
        previous = self.landmark_history[-2]
        
        time_diff = (self.timestamp_history[-1] - self.timestamp_history[-2]) / 1000.0
        
        if time_diff < 1e-6:
            return None
        
        velocity = (current - previous) / time_diff
        
        # Compute speed (magnitude) for each landmark
        speeds = np.linalg.norm(velocity, axis=1)
        
        return {
            'mean_speed': float(np.mean(speeds)),
            'max_speed': float(np.max(speeds)),
            'speed_std': float(np.std(speeds))
        }
    
    def _compute_acceleration(self):
        """Compute acceleration from velocity changes"""
        if len(self.landmark_history) < 3:
            return None
        
        # Get velocities at two time points
        v1 = (self.landmark_history[-2] - self.landmark_history[-3]) / self.frame_time
        v2 = (self.landmark_history[-1] - self.landmark_history[-2]) / self.frame_time
        
        acceleration = (v2 - v1) / self.frame_time
        
        # Compute acceleration magnitude for each landmark
        accel_mag = np.linalg.norm(acceleration, axis=1)
        
        return {
            'mean_acceleration': float(np.mean(accel_mag)),
            'max_acceleration': float(np.max(accel_mag))
        }
    
    def _compute_smoothness(self):
        """
        Compute movement smoothness (lower values = smoother)
        Using jerk (derivative of acceleration) as a proxy
        """
        if len(self.landmark_history) < 4:
            return None
        
        # Compute velocities for last 4 frames
        velocities = []
        for i in range(-3, 0):
            v = (self.landmark_history[i] - self.landmark_history[i-1]) / self.frame_time
            velocities.append(v)
        
        # Compute accelerations
        accelerations = []
        for i in range(len(velocities) - 1):
            a = (velocities[i+1] - velocities[i]) / self.frame_time
            accelerations.append(a)
        
        # Compute jerk (change in acceleration)
        if len(accelerations) >= 2:
            jerk = (accelerations[1] - accelerations[0]) / self.frame_time
            jerk_mag = np.linalg.norm(jerk, axis=1)
            
            # Smoothness score (lower jerk = smoother)
            smoothness_score = float(np.mean(jerk_mag))
            return smoothness_score
        
        return None
    
    def _compute_tremor(self):
        """
        Compute tremor score based on high-frequency variations
        """
        if len(self.landmark_history) < 10:
            return None
        
        # Get recent positions
        recent = np.array(list(self.landmark_history)[-10:])
        
        # Compute variance in position for each landmark
        position_variance = np.var(recent, axis=0)
        tremor_score = float(np.mean(np.linalg.norm(position_variance, axis=1)))
        
        return tremor_score
    
    def _compute_range_of_motion(self):
        """Compute range of motion for each joint over recent history"""
        if len(self.landmark_history) < 10:
            return None
        
        recent = np.array(list(self.landmark_history)[-10:])
        
        # Compute min and max positions
        min_pos = np.min(recent, axis=0)
        max_pos = np.max(recent, axis=0)
        
        # Range of motion for each landmark
        rom = max_pos - min_pos
        rom_magnitude = np.linalg.norm(rom, axis=1)
        
        return {
            'mean_rom': float(np.mean(rom_magnitude)),
            'max_rom': float(np.max(rom_magnitude)),
            'per_joint_rom': rom_magnitude.tolist()
        }
    
    def reset(self):
        """Reset history buffers"""
        self.landmark_history.clear()
        self.timestamp_history.clear()


class SessionRecorder:
    """Records movement data for a complete session"""
    
    def __init__(self, session_id, user_id, level_name):
        self.session_id = session_id
        self.user_id = user_id
        self.level_name = level_name
        self.start_time = datetime.now()
        
        self.feature_extractor = MovementFeatureExtractor()
        self.frame_data = []
        self.session_events = []
        
    def record_frame(self, hand_landmarks, timestamp_ms, event_type=None, handedness="Right"):
        """
        Record a single frame of data
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            timestamp_ms: Current timestamp
            event_type: Optional event marker (e.g., 'pose_start', 'pose_complete')
            handedness: Which hand ("Right" or "Left")
        """
        if hand_landmarks:
            features = self.feature_extractor.compute_features(hand_landmarks, timestamp_ms)
            
            if features:
                frame_record = {
                    'timestamp_ms': timestamp_ms,
                    'features': features,
                    'event_type': event_type,
                    'handedness': handedness
                }
                self.frame_data.append(frame_record)
    
    def mark_event(self, event_type, event_data=None):
        """Mark a specific event during the session"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': event_data or {}
        }
        self.session_events.append(event)
    
    def get_session_summary(self):
        """Compute summary statistics for the session"""
        if not self.frame_data:
            return None
        
        # Aggregate features across all frames
        summary = {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'level_name': self.level_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'total_frames': len(self.frame_data),
            'events': self.session_events
        }
        
        # Compute average features - filter out frames with None features
        velocities = [f.get('features', {}).get('velocity', {}).get('mean_speed', 0) 
                     for f in self.frame_data 
                     if f.get('features') is not None and f.get('features', {}).get('velocity', {}).get('mean_speed') is not None]
        
        smoothness_scores = [f.get('features', {}).get('smoothness', 0) 
                            for f in self.frame_data 
                            if f.get('features') is not None and f.get('features', {}).get('smoothness') is not None]
        
        tremor_scores = [f.get('features', {}).get('tremor_score', 0) 
                        for f in self.frame_data 
                        if f.get('features') is not None and f.get('features', {}).get('tremor_score') is not None]
        
        if velocities:
            summary['avg_speed'] = float(np.mean(velocities))
            summary['max_speed'] = float(np.max(velocities))
        
        if smoothness_scores:
            summary['avg_smoothness'] = float(np.mean(smoothness_scores))
        
        if tremor_scores:
            summary['avg_tremor'] = float(np.mean(tremor_scores))
        
        return summary
    
    def save_to_json(self, filepath):
        """Save session data to JSON file"""
        data = {
            'summary': self.get_session_summary(),
            'frame_data': self.frame_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_time_series_data(self):
        """Get data formatted for time series database"""
        return self.frame_data
