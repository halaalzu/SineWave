"""
ML Model Training Pipeline for FlowState
Trains on ALL recorded sessions to learn movement patterns

This script will:
1. Load ALL movement data from database (all sessions, all frames)
2. Extract features from hand landmarks
3. Train ML model to classify movement quality
4. Save trained model for real-time predictions

Current Status: Data collection ready, ML training to be implemented
"""

import numpy as np
import json
from database import RehabDatabase
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime


class MovementMLTrainer:
    """Train ML models on ALL collected movement data"""
    
    def __init__(self, db_path='flowstate.db'):
        self.db = RehabDatabase(db_path)
        self.scaler = StandardScaler()
        self.model = None
    
    def load_all_training_data(self):
        """Load ALL movement data from database for training"""
        print("\n" + "="*60)
        print("📚 Loading ALL Movement Data for ML Training")
        print("="*60 + "\n")
        
        cursor = self.db.conn.cursor()
        
        # Get ALL completed sessions (not just recent ones)
        cursor.execute('''
            SELECT session_id, total_frames, avg_speed, avg_smoothness, avg_tremor
            FROM sessions 
            WHERE completed = 1 AND total_frames > 0
            ORDER BY start_time DESC
        ''')
        
        all_sessions = cursor.fetchall()
        print(f"✅ Found {len(all_sessions)} sessions for training")
        
        # Load ALL frame-level data
        all_features = []
        all_labels = []
        
        total_frames = 0
        for session_id, frames, speed, smoothness, tremor in all_sessions:
            # Get all frames for this session
            cursor.execute('''
                SELECT landmark_positions, joint_angles, velocity_mean, 
                       smoothness_score, tremor_score, range_of_motion
                FROM movement_timeseries 
                WHERE session_id = ?
                ORDER BY frame_number
            ''', (session_id,))
            
            frames_data = cursor.fetchall()
            total_frames += len(frames_data)
            
            for frame in frames_data:
                try:
                    landmarks = json.loads(frame[0]) if frame[0] else None
                    angles = json.loads(frame[1]) if frame[1] else {}
                    
                    if landmarks and len(landmarks) == 21:
                        # Extract features from this frame
                        features = self._extract_frame_features(
                            landmarks, angles, frame[2], frame[3], frame[4], frame[5]
                        )
                        all_features.append(features)
                        
                        # Label based on session-level metrics
                        # Excellent: speed > 0.6, smoothness < 800, tremor < 0.001
                        # Good: speed > 0.4, smoothness < 1200, tremor < 0.002
                        # Poor: below those thresholds
                        label = self._calculate_quality_label(speed, smoothness, tremor)
                        all_labels.append(label)
                        
                except (json.JSONDecodeError, IndexError, TypeError):
                    continue
        
        print(f"✅ Loaded {total_frames:,} frames from {len(all_sessions)} sessions")
        print(f"✅ Extracted {len(all_features):,} feature vectors")
        
        return np.array(all_features), np.array(all_labels), total_frames
    
    def _extract_frame_features(self, landmarks, angles, velocity, smoothness, tremor, rom):
        """Extract numerical features from a single frame"""
        features = []
        
        # Landmark positions (21 joints × 3 coords = 63 features)
        for landmark in landmarks:
            if isinstance(landmark, (list, tuple)) and len(landmark) >= 3:
                features.extend([landmark[0], landmark[1], landmark[2]])
            else:
                features.extend([0, 0, 0])
        
        # Joint angles (up to 5 fingers × 3 angles = 15 features)
        angle_keys = ['thumb_mcp', 'index_mcp', 'middle_mcp', 'ring_mcp', 'pinky_mcp',
                      'thumb_ip', 'index_pip', 'middle_pip', 'ring_pip', 'pinky_pip',
                      'index_dip', 'middle_dip', 'ring_dip', 'pinky_dip', 'thumb_tip']
        for key in angle_keys:
            features.append(angles.get(key, 0))
        
        # Movement metrics (6 features)
        features.extend([
            velocity or 0,
            smoothness or 0,
            tremor or 0,
            rom or 0,
            # Derived features
            velocity * (1 / (tremor + 0.0001)),  # Speed-to-tremor ratio
            rom * (1 / (smoothness + 1))  # ROM-to-smoothness ratio
        ])
        
        return features
    
    def _calculate_quality_label(self, speed, smoothness, tremor):
        """
        Calculate quality label for training
        0 = Excellent, 1 = Good, 2 = Moderate, 3 = Poor
        """
        score = 0
        
        # Speed scoring
        if speed and speed > 0.6:
            score += 0
        elif speed and speed > 0.4:
            score += 1
        elif speed and speed > 0.25:
            score += 2
        else:
            score += 3
        
        # Smoothness scoring (SPARC - lower is better)
        if smoothness and smoothness < 800:
            score += 0
        elif smoothness and smoothness < 1200:
            score += 1
        elif smoothness and smoothness < 1800:
            score += 2
        else:
            score += 3
        
        # Tremor scoring
        if tremor and tremor < 0.001:
            score += 0
        elif tremor and tremor < 0.002:
            score += 1
        elif tremor and tremor < 0.004:
            score += 2
        else:
            score += 3
        
        # Average the scores and round to category
        avg_score = score / 3
        if avg_score < 0.5:
            return 0  # Excellent
        elif avg_score < 1.5:
            return 1  # Good
        elif avg_score < 2.5:
            return 2  # Moderate
        else:
            return 3  # Poor
    
    def train_model(self):
        """Train ML model on ALL collected data"""
        print("\n🤖 Starting ML Model Training...")
        print("This will use ALL your recorded movement data\n")
        
        # Load all data
        X, y, total_frames = self.load_all_training_data()
        
        if len(X) < 100:
            print(f"❌ Not enough data for training. Need at least 100 frames, have {len(X)}")
            print("   Record more sessions and try again!")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train):,} samples")
        print(f"📊 Test set: {len(X_test):,} samples")
        print(f"📊 Label distribution: {np.bincount(y)}\n")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model (using Random Forest for now - can upgrade to LSTM later)
        from sklearn.ensemble import RandomForestClassifier
        
        print("🔧 Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        print(f"\n✅ Training Complete!")
        print(f"   Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
        
        return True
    
    def save_model(self, filename='movement_quality_model.pkl'):
        """Save trained model to disk"""
        if self.model is None:
            print("❌ No model to save. Train first!")
            return False
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'trained_at': datetime.now().isoformat(),
            'feature_count': 84  # 63 landmarks + 15 angles + 6 metrics
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Model saved to: {filename}")
        return True
    
    def load_model(self, filename='movement_quality_model.pkl'):
        """Load trained model from disk"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            
            print(f"✅ Model loaded from: {filename}")
            print(f"   Trained at: {model_data['trained_at']}")
            return True
        except FileNotFoundError:
            print(f"❌ Model file not found: {filename}")
            return False


def main():
    """Train ML model on all collected data"""
    print("\n" + "="*60)
    print("🎓 FlowState ML Model Training Pipeline")
    print("="*60)
    
    trainer = MovementMLTrainer()
    
    # Train on ALL data
    success = trainer.train_model()
    
    if success:
        # Save model
        trainer.save_model()
        
        print("\n" + "="*60)
        print("✅ ML Training Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. The model is now trained on ALL your movement data")
        print("2. Update app_with_data.py to use the trained model for predictions")
        print("3. Record more sessions to improve model accuracy")
        print("4. Retrain periodically as you collect more data\n")
    else:
        print("\n" + "="*60)
        print("❌ Training Failed")
        print("="*60)
        print("\nRecord more freestyle sessions and try again!\n")


if __name__ == '__main__':
    main()
