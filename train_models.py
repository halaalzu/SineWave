"""
ML Training Pipeline for FlowState
Train models to classify rehabilitation stages and predict progress
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
from datetime import datetime
from typing import List, Dict
import os

# Import your database
try:
    from database_postgres import PostgresRehabDatabase
    USE_POSTGRES = True
except ImportError:
    USE_POSTGRES = False

from database import RehabDatabase


class RehabModelTrainer:
    """Train ML models for rehabilitation progress estimation"""
    
    def __init__(self, db_path='flowstate.db', use_postgres=False):
        """Initialize trainer with database connection"""
        if use_postgres and USE_POSTGRES:
            self.db = PostgresRehabDatabase()
        else:
            self.db = RehabDatabase(db_path)
        
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
    
    def extract_features_from_session(self, session_data: List[Dict]) -> np.ndarray:
        """
        Extract feature vector from session time series data
        
        Features extracted:
        - Velocity statistics (mean, std, max, min)
        - Acceleration statistics
        - Smoothness metrics
        - Tremor scores
        - Range of motion
        - Hand openness
        - Temporal patterns
        """
        if not session_data:
            return None
        
        # Convert to arrays
        velocities = [d.get('velocity_mean', 0) for d in session_data if d.get('velocity_mean')]
        accelerations = [d.get('acceleration_mean', 0) for d in session_data if d.get('acceleration_mean')]
        smoothness = [d.get('smoothness_score', 0) for d in session_data if d.get('smoothness_score')]
        tremor = [d.get('tremor_score', 0) for d in session_data if d.get('tremor_score')]
        rom = [d.get('range_of_motion', 0) for d in session_data if d.get('range_of_motion')]
        hand_open = [d.get('hand_openness', 0) for d in session_data if d.get('hand_openness')]
        
        features = []
        
        # Velocity features (4)
        if velocities:
            features.extend([
                np.mean(velocities),
                np.std(velocities),
                np.max(velocities),
                np.min(velocities)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Acceleration features (4)
        if accelerations:
            features.extend([
                np.mean(accelerations),
                np.std(accelerations),
                np.max(accelerations),
                np.percentile(accelerations, 90) if len(accelerations) > 10 else np.max(accelerations)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Smoothness features (3)
        if smoothness:
            features.extend([
                np.mean(smoothness),
                np.std(smoothness),
                np.min(smoothness)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Tremor features (4)
        if tremor:
            features.extend([
                np.mean(tremor),
                np.std(tremor),
                np.max(tremor),
                np.percentile(tremor, 90) if len(tremor) > 10 else np.max(tremor)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Range of motion features (3)
        if rom:
            features.extend([
                np.mean(rom),
                np.std(rom),
                np.max(rom)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Hand openness features (2)
        if hand_open:
            features.extend([
                np.mean(hand_open),
                np.std(hand_open)
            ])
        else:
            features.extend([0, 0])
        
        # Temporal features (2)
        features.extend([
            len(session_data),  # Total frames
            len(session_data) / 30.0 if len(session_data) > 0 else 0  # Duration estimate (assuming 30fps)
        ])
        
        return np.array(features)
    
    def prepare_training_data(self, min_frames=100):
        """
        Prepare training dataset from database
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            session_ids: List of session IDs for reference
        """
        print("Loading sessions from database...")
        
        if USE_POSTGRES:
            sessions = self.db.get_all_sessions_for_training(min_frames=min_frames)
        else:
            # For SQLite, get all sessions
            cursor = self.db.conn.cursor()
            cursor.execute('''
                SELECT s.*, u.rehab_stage, u.diagnosis
                FROM sessions s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.total_frames >= ? AND s.completed = 1
            ''', (min_frames,))
            
            columns = [desc[0] for desc in cursor.description]
            sessions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        print(f"Found {len(sessions)} valid sessions")
        
        if len(sessions) < 10:
            print("WARNING: Not enough sessions for training. Need at least 10 sessions.")
            print("Please collect more data first.")
            return None, None, None
        
        X_list = []
        y_list = []
        session_ids = []
        
        print("Extracting features from sessions...")
        for i, session in enumerate(sessions):
            session_id = session['session_id']
            rehab_stage = session.get('rehab_stage')
            
            if not rehab_stage or rehab_stage == 'unknown':
                continue
            
            # Get time series data
            timeseries = self.db.get_session_timeseries(session_id)
            
            if len(timeseries) < min_frames:
                continue
            
            # Extract features
            features = self.extract_features_from_session(timeseries)
            
            if features is not None:
                X_list.append(features)
                y_list.append(rehab_stage)
                session_ids.append(session_id)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(sessions)} sessions")
        
        if len(X_list) == 0:
            print("ERROR: No valid feature vectors extracted")
            return None, None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"\nDataset prepared:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {np.unique(y)}")
        print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Store feature names for later reference
        self.feature_names = [
            'velocity_mean', 'velocity_std', 'velocity_max', 'velocity_min',
            'acceleration_mean', 'acceleration_std', 'acceleration_max', 'acceleration_p90',
            'smoothness_mean', 'smoothness_std', 'smoothness_min',
            'tremor_mean', 'tremor_std', 'tremor_max', 'tremor_p90',
            'rom_mean', 'rom_std', 'rom_max',
            'hand_openness_mean', 'hand_openness_std',
            'total_frames', 'duration_seconds'
        ]
        
        return X, y, session_ids
    
    def train_random_forest(self, X, y, test_size=0.2):
        """
        Train Random Forest classifier
        
        Good for:
        - Non-linear relationships
        - Feature importance analysis
        - Robust to outliers
        """
        print("\n" + "="*60)
        print("Training Random Forest Classifier")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        print("\nTop 10 Most Important Features:")
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        # Cross-validation
        print("\nCross-validation scores:")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return {
            'accuracy': accuracy,
            'cv_scores': cv_scores.tolist(),
            'feature_importance': dict(zip(self.feature_names, importances.tolist()))
        }
    
    def train_gradient_boosting(self, X, y, test_size=0.2):
        """
        Train Gradient Boosting classifier
        
        Often more accurate than Random Forest
        """
        print("\n" + "="*60)
        print("Training Gradient Boosting Classifier")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy
        }
    
    def save_model(self, model_name='rehab_classifier', model_type='random_forest'):
        """Save trained model and scaler to disk"""
        if self.model is None:
            print("ERROR: No model trained yet")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = f"{model_dir}/{model_name}_{timestamp}.pkl"
        scaler_path = f"{model_dir}/{model_name}_scaler_{timestamp}.pkl"
        metadata_path = f"{model_dir}/{model_name}_metadata_{timestamp}.json"
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'timestamp': timestamp,
            'feature_names': self.feature_names,
            'model_path': model_path,
            'scaler_path': scaler_path
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to {scaler_path}")
        print(f"✓ Metadata saved to {metadata_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler (same directory and timestamp)
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        print(f"✓ Model loaded from {model_path}")
    
    def predict(self, session_data: List[Dict]) -> Dict:
        """
        Predict rehabilitation stage for a session
        
        Returns:
            {
                'predicted_stage': str,
                'confidence': float,
                'probabilities': dict
            }
        """
        if self.model is None:
            print("ERROR: No model loaded")
            return None
        
        # Extract features
        features = self.extract_features_from_session(session_data)
        
        if features is None:
            return None
        
        # Scale and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(features_scaled)[0]
            class_probas = dict(zip(self.model.classes_, probas.tolist()))
            confidence = float(np.max(probas))
        else:
            class_probas = {}
            confidence = 1.0
        
        return {
            'predicted_stage': prediction,
            'confidence': confidence,
            'probabilities': class_probas
        }


def main():
    """
    Main training script
    
    Usage:
        python train_models.py
    """
    print("\n" + "="*60)
    print("FlowState ML Training Pipeline")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = RehabModelTrainer()
    
    # Prepare data
    X, y, session_ids = trainer.prepare_training_data(min_frames=50)
    
    if X is None or len(X) < 10:
        print("\n❌ Not enough data to train models")
        print("Please collect more rehabilitation sessions first.")
        return
    
    # Train Random Forest (recommended)
    metrics = trainer.train_random_forest(X, y)
    
    # Save model
    model_path = trainer.save_model('rehab_stage_classifier', 'random_forest')
    
    # Optional: Train Gradient Boosting for comparison
    # metrics_gb = trainer.train_gradient_boosting(X, y)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print("\nTo use the model:")
    print("  1. Load it with trainer.load_model(model_path)")
    print("  2. Predict with trainer.predict(session_data)")


if __name__ == '__main__':
    main()
