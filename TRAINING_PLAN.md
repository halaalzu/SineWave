# Training Plan for FlowState Pose Detection

## Current Status
- **Model**: PyTorch PoseNet (42 features → 5 classes)
- **Current accuracy**: Unknown (likely low with minimal training data)
- **Data collected**: ~23 frames per session (WAY TOO SMALL)
- **Target poses**: palm, 1, 2, 3, fist

## Problem
Your current model was likely pre-trained on someone else's data (mentioned as "friend's rhythm game"). It needs to be retrained for YOUR specific hand poses to work accurately.

---

## 🎯 RECOMMENDED APPROACH: Collect Your Own Data

### Phase 1: Data Collection (1-2 hours)
**Goal**: Get 500+ examples per pose

#### Quick Collection Script
Use the existing Freestyle page to record data:

1. **Record 5 separate sessions per pose**:
   - Hold pose "palm" for 30 seconds → ~600 frames
   - Hold pose "1" for 30 seconds → ~600 frames
   - Hold pose "2" for 30 seconds → ~600 frames
   - Hold pose "3" for 30 seconds → ~600 frames
   - Hold pose "fist" for 30 seconds → ~600 frames

2. **Vary conditions**:
   - Different hand distances from camera
   - Different angles (slight rotation)
   - Different lighting (turn lights on/off)
   - Different backgrounds

3. **Total target**: 3,000+ frames (600 per pose × 5 poses)

### Phase 2: Data Preparation
**Extract features from database**:
```bash
cd FlowState
python prepare_training_data.py
```

This will:
- Load all recorded sessions
- Extract MediaPipe landmarks (21 × 3 = 63 features)
- Label each frame with its pose
- Save as training dataset

### Phase 3: Model Training
**Train new model on YOUR data**:
```bash
python train_pose_model.py
```

This will:
- Use your collected data
- Train PyTorch neural network
- Save new model as `pose_model.pt`
- Show accuracy metrics

### Phase 4: Testing & Refinement
1. Test model on Freestyle page
2. Record which poses are confused
3. Collect MORE data for confused poses
4. Retrain

**Expected accuracy after proper training**: 90-95%

---

## 🌐 ALTERNATIVE: Use Public Dataset (Faster but Less Accurate)

### Option A: Fine-tune MediaPipe Model
MediaPipe already has excellent hand tracking. You could:
1. Use MediaPipe's hand landmarks (already doing this)
2. Train just a small classifier on top
3. Use HaGRID dataset for generic hand poses
4. Fine-tune on your 5 specific poses

### Option B: Transfer Learning from HaGRID
```bash
# Download HaGRID subset
wget https://github.com/hukenovs/hagrid/releases/download/v1.0/hagrid_dataset_512.zip

# Use pre-trained features + train small classifier
python train_with_hagrid.py
```

**Pros**: 
- 552k high-quality images
- Pre-labeled gestures
- Faster training

**Cons**:
- Their gestures ≠ your gestures
- Need to map their classes to yours
- Still need YOUR data to fine-tune

---

## 📊 What You Need NOW

### Immediate Action (10 minutes):
Create a data collection session:

1. Open http://localhost:8080/freestyle
2. Make pose "1" and HOLD IT for 30 seconds
3. Check database for frame count:
   ```bash
   python -c "from database import RehabDatabase; db = RehabDatabase('flowstate.db'); cursor = db.conn.cursor(); cursor.execute('SELECT session_id, total_frames FROM sessions ORDER BY start_time DESC LIMIT 1'); print(cursor.fetchone())"
   ```
4. Should see ~600 frames

### Repeat for All Poses
- Do this 5 times per pose
- Total time: ~15 minutes × 5 poses = 1.5 hours
- Result: Enough data to train a decent model

---

## 🔧 Tools You Already Have

Your codebase has:
- ✅ `train_ml_model.py` - Movement quality training
- ✅ `train_models.py` - Rehab stage classifier
- ✅ Database with session storage
- ❌ **MISSING**: Pose model training script

You need a NEW script: `train_pose_model.py` specifically for the 5 poses.

---

## 🚀 Best Path Forward

**For DEMO in next few days:**
1. Collect 200+ frames per pose (10 min recording time)
2. Train quick model
3. Test and refine

**For PRODUCTION quality:**
1. Collect 1000+ frames per pose
2. Add data augmentation (rotation, brightness)
3. Use more complex model (CNN or LSTM)
4. Achieve 95%+ accuracy

---

## Public Datasets to Consider

If you want to supplement YOUR data:

1. **HaGRID** - Best for hand gestures
   - https://github.com/hukenovs/hagrid
   - 552k images, 18 classes
   - Download subset: ~2GB

2. **EgoHands** - Good for hand detection
   - http://vision.soic.indiana.edu/projects/egohands/
   - 4,800 images
   - Smaller, easier to use

3. **FreiHAND** - Research quality
   - https://lmb.informatik.uni-freiburg.de/projects/freihand/
   - 130k images with 3D poses
   - Overkill for your use case

**Reality check**: Public datasets won't have YOUR exact poses. You'll still need your own data.

---

## Summary

**BEST OPTION**: Collect your own data (1-2 hours) + Train custom model
- Most accurate for YOUR specific poses
- Full control over quality
- Quick to collect with existing UI

**BACKUP OPTION**: Use HaGRID for initial training + Fine-tune with your data
- Faster initial model
- Still need your data for accuracy
- More complex pipeline

**RECOMMENDED**: Start with self-collected data, it's simpler and more accurate for your use case.
