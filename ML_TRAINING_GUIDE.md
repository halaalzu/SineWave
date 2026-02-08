# ML Training & PostgreSQL Setup Guide

## Overview

This guide covers:
1. Setting up PostgreSQL database
2. Training ML models for rehabilitation stage classification
3. Using DTW analysis for movement comparison
4. Model deployment and inference

---

## 1. PostgreSQL Setup

### Install PostgreSQL (macOS)

```bash
# Install PostgreSQL
brew install postgresql

# Start PostgreSQL service
brew services start postgresql

# Create database
createdb flowstate

# Optional: Install TimescaleDB for better time series performance
brew tap timescale/tap
brew install timescaledb
```

### Migrate from SQLite to PostgreSQL

```bash
# Run migration script
python -c "from database_postgres import migrate_sqlite_to_postgres; migrate_sqlite_to_postgres('flowstate.db')"
```

This will:
- Create all tables in PostgreSQL
- Migrate users, sessions, and time series data
- Set up indexes for fast queries
- Enable TimescaleDB if available

### Test PostgreSQL Connection

```python
from database_postgres import PostgresRehabDatabase

# Connect to database
db = PostgresRehabDatabase(
    dbname='flowstate',
    user='postgres',
    password='',
    host='localhost',
    port=5432
)

# Test with a query
sessions = db.get_all_sessions_for_training()
print(f"Found {len(sessions)} training sessions")
```

---

## 2. Training ML Models

### Collect Training Data

**Important**: You need at least 10-20 sessions with labeled rehabilitation stages before training.

Make sure your users have `rehab_stage` set to one of:
- `'early'` - Early rehabilitation (high tremor, low ROM)
- `'mid'` - Mid rehabilitation (improving control)
- `'advanced'` - Advanced rehabilitation (good control, full ROM)

### Run Training

```bash
# Train models on your data
python train_models.py
```

This will:
1. Load all completed sessions from database
2. Extract 22 features per session:
   - Velocity statistics (mean, std, max, min)
   - Acceleration metrics
   - Smoothness scores
   - Tremor analysis
   - Range of motion
   - Hand openness
   - Temporal features
3. Train Random Forest classifier
4. Display accuracy metrics and feature importance
5. Save model to `models/` directory

### Training Output Example

```
====================================================================
FlowState ML Training Pipeline
====================================================================

Loading sessions from database...
Found 45 valid sessions

Extracting features from sessions...
  Processed 10/45 sessions
  Processed 20/45 sessions
  ...

Dataset prepared:
  Samples: 45
  Features: 22
  Classes: ['early' 'mid' 'advanced']
  Class distribution: {'early': 15, 'mid': 18, 'advanced': 12}

====================================================================
Training Random Forest Classifier
====================================================================

Training model...

Test Accuracy: 0.889

Classification Report:
              precision    recall  f1-score   support

       early       0.86      0.90      0.88         5
         mid       0.88      0.88      0.88         4
    advanced       0.95      0.89      0.92         4

Top 10 Most Important Features:
  1. tremor_mean: 0.1523
  2. smoothness_mean: 0.1342
  3. velocity_std: 0.0987
  4. rom_max: 0.0856
  ...

✓ Model saved to models/rehab_stage_classifier_20250115_143022.pkl
```

### Model Performance Tips

**If accuracy is low (<70%)**:
- Collect more data (aim for 50+ sessions)
- Ensure rehab stages are labeled correctly
- Check that sessions have enough frames (>100 frames minimum)
- Try increasing `min_frames` in training script

**If certain classes perform poorly**:
- Collect more samples for those classes
- Review if stage definitions are clear
- Consider merging similar stages

---

## 3. Using Trained Models

### Load Model for Inference

```python
from train_models import RehabModelTrainer

# Initialize trainer
trainer = RehabModelTrainer()

# Load saved model
trainer.load_model('models/rehab_stage_classifier_20250115_143022.pkl')

# Get session data
session_data = db.get_session_timeseries('session_xyz')

# Predict rehabilitation stage
prediction = trainer.predict(session_data)

print(f"Predicted Stage: {prediction['predicted_stage']}")
print(f"Confidence: {prediction['confidence']:.2f}")
print(f"Probabilities: {prediction['probabilities']}")
```

Output:
```
Predicted Stage: mid
Confidence: 0.87
Probabilities: {'early': 0.08, 'mid': 0.87, 'advanced': 0.05}
```

### Integrate with Flask App

```python
# In your Flask app
from train_models import RehabModelTrainer

# Load model at startup
model_trainer = RehabModelTrainer()
model_trainer.load_model('models/rehab_stage_classifier_20250115_143022.pkl')

@app.route('/api/predict/<session_id>')
def predict_stage(session_id):
    session_data = db.get_session_timeseries(session_id)
    prediction = model_trainer.predict(session_data)
    return jsonify(prediction)
```

---

## 4. DTW Movement Analysis

### Compare Patient to Baseline

```python
from dtw_analysis import MovementAnalyzer

analyzer = MovementAnalyzer()

# Compare patient session to a baseline/ideal movement
result = analyzer.compare_to_baseline(
    session_id='patient_session_123',
    baseline_session_id='ideal_movement_456'
)

print(f"Similarity: {result['overall_similarity']:.1f}%")
print(f"DTW Distance: {result['dtw_distance']:.2f}")

# Per-landmark similarity
for landmark in result['landmark_similarities']:
    print(f"Landmark {landmark['landmark_id']}: {landmark['similarity']:.1f}%")
```

### Analyze Movement Quality

```python
# Analyze without baseline comparison
analysis = analyzer.analyze_movement_quality('session_123')

print(f"Overall Quality: {analysis['overall_quality_score']:.1f}/100")
print(f"Smoothness: {analysis['smoothness']['quality_score']:.1f}/100")
print(f"Tremor Score: {analysis['tremor']['mean']:.3f}")
print(f"Average Speed: {analysis['velocity']['mean']:.3f}")
```

### Track Improvement Over Time

```python
# Get improvement metrics for a user
improvement = analyzer.get_improvement_metrics('user_123', recent_sessions=10)

print(f"Sessions Analyzed: {improvement['sessions_analyzed']}")
print(f"Time Period: {improvement['time_period']['start']} to {improvement['time_period']['end']}")

for metric, data in improvement['improvements'].items():
    print(f"\n{metric}:")
    print(f"  First: {data['first_value']:.3f}")
    print(f"  Last: {data['last_value']:.3f}")
    print(f"  Change: {data['percent_change']:.1f}%")
    print(f"  Trend: {data['trend']}")
```

---

## 5. Complete Workflow Example

### Full Pipeline: Data Collection → Training → Inference

```python
from database_postgres import PostgresRehabDatabase
from train_models import RehabModelTrainer
from dtw_analysis import MovementAnalyzer

# 1. Setup database
db = PostgresRehabDatabase()

# 2. Train model (after collecting data)
trainer = RehabModelTrainer()
X, y, session_ids = trainer.prepare_training_data(min_frames=50)

if X is not None and len(X) >= 10:
    metrics = trainer.train_random_forest(X, y)
    model_path = trainer.save_model()
    print(f"Model trained with {metrics['accuracy']:.2f} accuracy")
else:
    print("Need more training data")
    exit()

# 3. Use trained model for new sessions
trainer.load_model(model_path)

new_session_data = db.get_session_timeseries('new_session_789')
prediction = trainer.predict(new_session_data)

print(f"\nPrediction for new session:")
print(f"  Stage: {prediction['predicted_stage']}")
print(f"  Confidence: {prediction['confidence']:.2%}")

# 4. DTW analysis
analyzer = MovementAnalyzer()
quality = analyzer.analyze_movement_quality('new_session_789')

print(f"\nMovement Quality Analysis:")
print(f"  Overall Quality: {quality['overall_quality_score']:.1f}/100")
print(f"  Smoothness: {quality['smoothness']['mean']:.3f}")
print(f"  Tremor: {quality['tremor']['mean']:.3f}")

# 5. Compare to baseline
baseline_comparison = analyzer.compare_to_baseline(
    'new_session_789',
    'ideal_baseline_session'
)

print(f"\nBaseline Comparison:")
print(f"  Similarity: {baseline_comparison['overall_similarity']:.1f}%")
```

---

## 6. Production Deployment

### Environment Variables

Create `.env` file:
```bash
# PostgreSQL
DB_NAME=flowstate
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432

# ML Models
MODEL_PATH=models/rehab_stage_classifier_latest.pkl
```

### Load in Code

```python
import os
from dotenv import load_dotenv

load_dotenv()

db = PostgresRehabDatabase(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT', 5432))
)
```

### Model Versioning

```python
# Save model with version
trainer.save_model('rehab_classifier_v1.2', 'random_forest')

# Track in database
db.save_ml_model_metadata(
    model_name='rehab_classifier',
    model_type='random_forest',
    version='1.2',
    training_sessions=session_ids,
    metrics={'accuracy': 0.89, 'cv_score': 0.87},
    model_path='models/rehab_classifier_v1.2.pkl',
    description='Improved training with 50 sessions'
)
```

---

## 7. Troubleshooting

### PostgreSQL Connection Issues

```bash
# Check if PostgreSQL is running
brew services list | grep postgresql

# Start if not running
brew services start postgresql

# Check connection
psql -d flowstate -c "SELECT version();"
```

### Not Enough Training Data

```python
# Check how many sessions you have
sessions = db.get_all_sessions_for_training()
print(f"Training sessions available: {len(sessions)}")

# Check class distribution
from collections import Counter
stages = [s['rehab_stage'] for s in sessions]
print(Counter(stages))
```

### Low Model Accuracy

1. **Collect more data**: Aim for 20+ sessions per class
2. **Check labels**: Ensure rehab_stage is set correctly
3. **Feature engineering**: Add more features in `extract_features_from_session()`
4. **Try different models**: Use Gradient Boosting instead of Random Forest
5. **Hyperparameter tuning**: Adjust `n_estimators`, `max_depth`, etc.

### DTW Analysis Errors

```python
# Check if session has enough data
session_data = db.get_session_timeseries('session_id')
print(f"Frames in session: {len(session_data)}")

# Verify landmark data exists
if session_data:
    print(f"Has landmarks: {'landmark_positions' in session_data[0]}")
```

---

## 8. Next Steps

### Advanced Features to Implement

1. **LSTM Models**: For sequence prediction (time series)
2. **Progress Prediction**: Forecast rehabilitation timeline
3. **Anomaly Detection**: Identify unusual movement patterns
4. **Real-time Inference**: Predict stage during live session
5. **Dashboard**: Visualize progress and predictions

### Example: LSTM Implementation

```python
# TODO: Create lstm_model.py with PyTorch/TensorFlow
# Features:
# - Sequence-to-sequence learning
# - Predict next movement
# - Time series forecasting
```

### Dashboard Integration

```python
# Flask route for dashboard
@app.route('/dashboard/<user_id>')
def dashboard(user_id):
    # Get user sessions
    sessions = db.get_user_sessions(user_id)
    
    # Get predictions for each session
    predictions = []
    for session in sessions:
        data = db.get_session_timeseries(session['session_id'])
        pred = trainer.predict(data)
        predictions.append(pred)
    
    # Get improvement metrics
    analyzer = MovementAnalyzer()
    improvement = analyzer.get_improvement_metrics(user_id)
    
    return render_template('dashboard.html',
                         sessions=sessions,
                         predictions=predictions,
                         improvement=improvement)
```

---

## Quick Reference Commands

```bash
# Setup PostgreSQL
brew install postgresql
brew services start postgresql
createdb flowstate

# Migrate data
python -c "from database_postgres import migrate_sqlite_to_postgres; migrate_sqlite_to_postgres()"

# Train models
python train_models.py

# Run DTW analysis
python dtw_analysis.py

# Test connection
python -c "from database_postgres import PostgresRehabDatabase; db = PostgresRehabDatabase(); print('✓ Connected')"
```

---

## Resources

- **PostgreSQL Docs**: https://www.postgresql.org/docs/
- **TimescaleDB**: https://docs.timescale.com/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **DTW Tutorial**: https://rtavenar.github.io/blog/dtw.html
