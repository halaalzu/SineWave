"""
ML-based Session Quality Analyzer
Scores freestyle sessions 0-100 based on ROM, smoothness, speed, consistency
"""
import numpy as np
import json
from database import RehabDatabase
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class SessionQualityAnalyzer:
    def __init__(self, db_path='flowstate.db'):
        self.db = RehabDatabase(db_path)
        self.model_path = 'session_quality_model.pkl'
        self.scaler_path = 'session_quality_scaler.pkl'
        
        # Initialize or load model
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("✅ Loaded existing quality scoring model")
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self._train_initial_model()
            print("✅ Created new quality scoring model")
    
    def _train_initial_model(self):
        """Train model with good/bad example templates"""
        # Define ideal "good" rehab session characteristics
        good_example = {
            'rom_score': 90,  # High range of motion
            'smoothness_score': 85,  # Smooth movements
            'speed_score': 75,  # Moderate speed (not too fast/slow)
            'consistency_score': 88,  # Consistent performance
            'pose_variety': 5,  # Uses all 5 poses
            'transition_quality': 82,  # Good transitions
            'target_score': 90  # Overall quality score
        }
        
        # Define "bad" rehab session characteristics
        bad_example = {
            'rom_score': 30,  # Limited range of motion
            'smoothness_score': 25,  # Jerky movements
            'speed_score': 20,  # Too slow or erratic
            'consistency_score': 28,  # Inconsistent
            'pose_variety': 2,  # Limited poses
            'transition_quality': 22,  # Poor transitions
            'target_score': 25  # Overall quality score
        }
        
        # Create synthetic training data based on examples
        X_train = []
        y_train = []
        
        # Generate variations of good examples (scores 70-100)
        for i in range(30):
            noise = np.random.normal(0, 5, 6)
            features = [
                max(60, min(100, good_example['rom_score'] + noise[0])),
                max(60, min(100, good_example['smoothness_score'] + noise[1])),
                max(60, min(100, good_example['speed_score'] + noise[2])),
                max(60, min(100, good_example['consistency_score'] + noise[3])),
                max(3, min(5, good_example['pose_variety'] + noise[4]/10)),
                max(60, min(100, good_example['transition_quality'] + noise[5]))
            ]
            target = max(70, min(100, good_example['target_score'] + noise[0]))
            X_train.append(features)
            y_train.append(target)
        
        # Generate variations of bad examples (scores 0-40)
        for i in range(30):
            noise = np.random.normal(0, 5, 6)
            features = [
                max(0, min(40, bad_example['rom_score'] + noise[0])),
                max(0, min(40, bad_example['smoothness_score'] + noise[1])),
                max(0, min(40, bad_example['speed_score'] + noise[2])),
                max(0, min(40, bad_example['consistency_score'] + noise[3])),
                max(1, min(3, bad_example['pose_variety'] + noise[4]/10)),
                max(0, min(40, bad_example['transition_quality'] + noise[5]))
            ]
            target = max(0, min(40, bad_example['target_score'] + noise[0]))
            X_train.append(features)
            y_train.append(target)
        
        # Generate medium examples (scores 40-70)
        for i in range(20):
            noise = np.random.normal(0, 8, 6)
            features = [
                40 + np.random.uniform(0, 30) + noise[0],
                40 + np.random.uniform(0, 30) + noise[1],
                40 + np.random.uniform(0, 30) + noise[2],
                40 + np.random.uniform(0, 30) + noise[3],
                2 + np.random.uniform(0, 2) + noise[4]/10,
                40 + np.random.uniform(0, 30) + noise[5]
            ]
            features = [max(0, min(100, f)) for f in features]
            target = 40 + np.random.uniform(0, 30)
            X_train.append(features)
            y_train.append(target)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_scaled, y_train)
        
        # Save model
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        print(f"✅ Trained initial model on {len(X_train)} synthetic examples")
    
    def extract_features(self, session_id):
        """Extract features from a session for scoring"""
        cursor = self.db.conn.cursor()
        
        # Get session data
        cursor.execute('''
            SELECT total_frames, avg_speed, avg_smoothness, avg_tremor, 
                   duration_seconds
            FROM sessions 
            WHERE session_id = ?
        ''', (session_id,))
        
        session_data = cursor.fetchone()
        if not session_data:
            return None
        
        total_frames, avg_speed, avg_smoothness, avg_tremor, duration = session_data
        
        # Get pose events
        cursor.execute('''
            SELECT event_data FROM session_events 
            WHERE session_id = ? AND event_type = 'pose_detected'
        ''', (session_id,))
        
        pose_events = cursor.fetchall()
        
        # Extract pose types
        poses = []
        for event in pose_events:
            try:
                data = json.loads(event[0])
                poses.append(data.get('pose'))
            except:
                pass
        
        # Calculate features
        
        # 1. ROM Score (based on pose variety and speed)
        unique_poses = len(set(poses))
        rom_score = min(100, (unique_poses / 5.0) * 100)  # 5 total poses
        if avg_speed:
            rom_score = min(100, rom_score * (1 + min(avg_speed / 2, 0.5)))
        
        # 2. Smoothness Score (inverse of tremor, high smoothness value)
        if avg_smoothness:
            # Normalize smoothness (lower values are better, so invert)
            smoothness_score = min(100, max(0, 100 - (avg_smoothness / 100)))
        else:
            smoothness_score = 50
        
        # 3. Speed Score (optimal speed range 0.5-2.0)
        if avg_speed:
            if 0.5 <= avg_speed <= 2.0:
                speed_score = 100
            elif avg_speed < 0.5:
                speed_score = (avg_speed / 0.5) * 100
            else:
                speed_score = max(0, 100 - ((avg_speed - 2.0) * 20))
        else:
            speed_score = 50
        
        # 4. Consistency Score (based on tremor - lower is better)
        if avg_tremor is not None:
            # Tremor typically 0.0001 to 0.05, lower is better
            consistency_score = max(0, min(100, 100 - (avg_tremor * 2000)))
        else:
            consistency_score = 50
        
        # 5. Pose Variety
        pose_variety = unique_poses
        
        # 6. Transition Quality (based on number of pose changes vs duration)
        if duration > 0:
            transitions_per_second = len(pose_events) / duration
            # Optimal: 2-5 transitions per second
            if 2 <= transitions_per_second <= 5:
                transition_quality = 100
            elif transitions_per_second < 2:
                transition_quality = (transitions_per_second / 2) * 100
            else:
                transition_quality = max(0, 100 - ((transitions_per_second - 5) * 10))
        else:
            transition_quality = 50
        
        return {
            'rom_score': rom_score,
            'smoothness_score': smoothness_score,
            'speed_score': speed_score,
            'consistency_score': consistency_score,
            'pose_variety': pose_variety,
            'transition_quality': transition_quality
        }
    
    def score_session(self, session_id):
        """Score a session from 0-100"""
        features = self.extract_features(session_id)
        
        if not features:
            return None
        
        # Prepare features for model
        X = np.array([[
            features['rom_score'],
            features['smoothness_score'],
            features['speed_score'],
            features['consistency_score'],
            features['pose_variety'],
            features['transition_quality']
        ]])
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        score = self.model.predict(X_scaled)[0]
        
        # Clip to 0-100
        score = max(0, min(100, score))
        
        # Save score to database
        self._save_score(session_id, score, features)
        
        return {
            'session_id': session_id,
            'quality_score': round(score, 1),
            'breakdown': {
                'rom': round(features['rom_score'], 1),
                'smoothness': round(features['smoothness_score'], 1),
                'speed': round(features['speed_score'], 1),
                'consistency': round(features['consistency_score'], 1)
            },
            'pose_variety': int(features['pose_variety']),
            'transition_quality': round(features['transition_quality'], 1)
        }
    
    def _save_score(self, session_id, score, features):
        """Save quality score to database"""
        cursor = self.db.conn.cursor()
        
        # Create table if doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_quality_scores (
                session_id TEXT PRIMARY KEY,
                quality_score REAL,
                rom_score REAL,
                smoothness_score REAL,
                speed_score REAL,
                consistency_score REAL,
                pose_variety INTEGER,
                transition_quality REAL,
                scored_at TEXT
            )
        ''')
        
        # Insert score
        cursor.execute('''
            INSERT OR REPLACE INTO session_quality_scores 
            (session_id, quality_score, rom_score, smoothness_score, 
             speed_score, consistency_score, pose_variety, transition_quality, scored_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, score, 
            features['rom_score'], features['smoothness_score'],
            features['speed_score'], features['consistency_score'],
            features['pose_variety'], features['transition_quality'],
            datetime.now().isoformat()
        ))
        
        self.db.conn.commit()
    
    def retrain_with_session(self, session_id, user_feedback_score=None):
        """Retrain model with new session data"""
        features = self.extract_features(session_id)
        
        if not features:
            return False
        
        # If user provided feedback, use it; otherwise use model's prediction
        if user_feedback_score is None:
            result = self.score_session(session_id)
            target_score = result['quality_score']
        else:
            target_score = user_feedback_score
        
        # Prepare data
        X_new = np.array([[
            features['rom_score'],
            features['smoothness_score'],
            features['speed_score'],
            features['consistency_score'],
            features['pose_variety'],
            features['transition_quality']
        ]])
        
        y_new = np.array([target_score])
        
        # Get existing training data (from model's training)
        # In practice, we'd store this, but for now we'll do incremental learning
        
        # Scale new data
        X_scaled = self.scaler.transform(X_new)
        
        # Incremental fit (warm start)
        self.model.fit(X_scaled, y_new)
        
        # Save updated model
        joblib.dump(self.model, self.model_path)
        
        print(f"✅ Retrained model with session {session_id[:8]}...")
        
        return True
    
    def get_most_recent_score(self, user_id='default_user'):
        """Get the most recent session score for a user"""
        cursor = self.db.conn.cursor()
        
        # Get most recent session
        cursor.execute('''
            SELECT s.session_id, s.start_time, s.total_frames,
                   q.quality_score, q.rom_score, q.smoothness_score,
                   q.speed_score, q.consistency_score, q.pose_variety,
                   q.transition_quality
            FROM sessions s
            LEFT JOIN session_quality_scores q ON s.session_id = q.session_id
            WHERE s.user_id = ? AND s.completed = 1
            ORDER BY s.start_time DESC
            LIMIT 1
        ''', (user_id,))
        
        result = cursor.fetchone()
        
        if not result:
            return None
        
        session_id, start_time, total_frames, quality_score, rom, smoothness, speed, consistency, variety, transition = result
        
        # If not scored yet, score it now
        if quality_score is None:
            score_result = self.score_session(session_id)
            if score_result:
                quality_score = score_result['quality_score']
                rom = score_result['breakdown']['rom']
                smoothness = score_result['breakdown']['smoothness']
                speed = score_result['breakdown']['speed']
                consistency = score_result['breakdown']['consistency']
                variety = score_result['pose_variety']
                transition = score_result['transition_quality']
        
        return {
            'session_id': session_id,
            'start_time': start_time,
            'total_frames': total_frames,
            'quality_score': round(quality_score, 1) if quality_score else 0,
            'breakdown': {
                'rom': round(rom, 1) if rom else 0,
                'smoothness': round(smoothness, 1) if smoothness else 0,
                'speed': round(speed, 1) if speed else 0,
                'consistency': round(consistency, 1) if consistency else 0
            },
            'pose_variety': int(variety) if variety else 0,
            'transition_quality': round(transition, 1) if transition else 0
        }


if __name__ == '__main__':
    # Test the analyzer
    analyzer = SessionQualityAnalyzer()
    
    # Get most recent session
    result = analyzer.get_most_recent_score()
    
    if result:
        print("\n" + "="*60)
        print("📊 MOST RECENT SESSION QUALITY ANALYSIS")
        print("="*60)
        print(f"Session: {result['session_id'][:16]}...")
        print(f"Time: {result['start_time']}")
        print(f"Frames: {result['total_frames']}")
        print(f"\n🎯 Overall Quality Score: {result['quality_score']}/100")
        print(f"\n📈 Breakdown:")
        print(f"  • ROM (Range of Motion): {result['breakdown']['rom']}/100")
        print(f"  • Smoothness: {result['breakdown']['smoothness']}/100")
        print(f"  • Speed: {result['breakdown']['speed']}/100")
        print(f"  • Consistency: {result['breakdown']['consistency']}/100")
        print(f"\n🎨 Additional Metrics:")
        print(f"  • Pose Variety: {result['pose_variety']}/5 poses")
        print(f"  • Transition Quality: {result['transition_quality']}/100")
        print("="*60)
    else:
        print("No sessions found")
