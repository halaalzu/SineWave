# ML-Based Session Quality Scoring System

## Overview
FlowState now includes an ML-powered system that automatically scores each freestyle rehab session on a scale of 0-100 based on movement quality.

## Key Features

### 1. **Automatic Scoring**
- Every completed session is automatically scored
- Score appears in real-time after session ends
- No manual intervention required

### 2. **ML Training**
- Trained initially on synthetic good/bad examples
- **Self-improving**: Each new session contributes to model training
- Uses RandomForestRegressor for robust predictions
- Model persists between server restarts

### 3. **Scoring Metrics**

#### **ROM (Range of Motion)** - 0-100
- Measures pose variety (1-5 different poses)
- Considers speed of movements
- Higher score = more complete range of motion

#### **Smoothness** - 0-100
- Inverse of jerkiness/tremor
- Smoother movements = higher score
- Based on movement_timeseries data

#### **Speed** - 0-100
- Optimal range: 0.5-2.0 units/second
- Too slow or too fast reduces score
- Moderate, controlled movements score best

#### **Consistency** - 0-100
- Based on tremor measurements
- Lower tremor = higher consistency
- Measures stability throughout session

#### **Additional Metrics**
- **Pose Variety**: Number of unique poses (1-5)
- **Transition Quality**: Smoothness of pose changes

### 4. **Score Interpretation**

```
85-100: ⭐ Excellent - Outstanding progress
70-84:  👍 Good - Solid progress
50-69:  💪 Fair - On the right track
30-49:  📈 Needs Improvement - Keep practicing
0-29:   🌱 Beginning - Building foundation
```

## API Endpoint

### GET `/api/user/<user_id>/session-quality`

Returns ML quality analysis for most recent session:

```json
{
  "success": true,
  "session": {
    "id": "05ed4569-b9cd-4f46-b0b4-98ba0996acc6",
    "timestamp": "2026-02-07T21:01:09.976404",
    "frames": 83
  },
  "qualityScore": 72.3,
  "breakdown": {
    "rom": 68.5,
    "smoothness": 75.2,
    "speed": 82.1,
    "consistency": 71.9
  },
  "metrics": {
    "poseVariety": 4,
    "transitionQuality": 68.3
  },
  "interpretation": {
    "level": "Good",
    "message": "Great work! You're making solid progress in your rehabilitation.",
    "emoji": "👍"
  }
}
```

## How It Works

### Training Phase
1. **Initial Training**: Model trains on 80 synthetic examples
   - 30 "good" examples (scores 70-100)
   - 30 "bad" examples (scores 0-40)
   - 20 "medium" examples (scores 40-70)

2. **Continuous Learning**: After each session:
   - Session is scored by current model
   - Session data is added to training set
   - Model retrains with new data
   - Improves accuracy over time

### Feature Extraction
For each session, the system extracts:
- Average speed from movement data
- Smoothness metrics
- Tremor measurements
- Pose variety count
- Transition frequency and quality
- Duration and frame count

### Scoring Process
1. Session completes and is saved to database
2. `SessionQualityAnalyzer.score_session()` is called
3. Features are extracted from session data
4. Features are scaled using StandardScaler
5. RandomForest model predicts quality score
6. Score and breakdown saved to `session_quality_scores` table
7. Model retrains with this session's data

## Database Schema

```sql
CREATE TABLE session_quality_scores (
    session_id TEXT PRIMARY KEY,
    quality_score REAL,
    rom_score REAL,
    smoothness_score REAL,
    speed_score REAL,
    consistency_score REAL,
    pose_variety INTEGER,
    transition_quality REAL,
    scored_at TEXT
);
```

## Files

- **`session_quality_ml.py`**: Main ML analyzer class
- **`session_quality_model.pkl`**: Trained RandomForest model
- **`session_quality_scaler.pkl`**: Feature scaler
- **`app_with_data.py`**: Integrated scoring in session completion

## Usage Example

```python
from session_quality_ml import SessionQualityAnalyzer

analyzer = SessionQualityAnalyzer('flowstate.db')

# Score a specific session
result = analyzer.score_session('session_id_here')
print(f"Quality Score: {result['quality_score']}/100")
print(f"ROM: {result['breakdown']['rom']}")
print(f"Smoothness: {result['breakdown']['smoothness']}")

# Get most recent session score
recent = analyzer.get_most_recent_score('default_user')
print(f"Latest score: {recent['quality_score']}/100")
```

## Benefits

1. **Objective Measurement**: No human bias in scoring
2. **Consistent Standards**: Same criteria every time
3. **Adaptive**: Improves with more data
4. **Real-time Feedback**: Immediate results after session
5. **Progress Tracking**: Compare scores over time
6. **Motivation**: Clear numeric goal to improve

## Future Enhancements

- [ ] Compare scores across multiple sessions (trends)
- [ ] Personalized scoring based on user baseline
- [ ] Different scoring profiles for different rehab goals
- [ ] Confidence intervals for predictions
- [ ] Feature importance analysis
- [ ] Export training data for model improvement

---

**Note**: The ML model automatically improves with each session. The more you practice, the smarter the scoring becomes! 🎯
