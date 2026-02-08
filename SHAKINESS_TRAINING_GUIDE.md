# 🎯 FlowState: Complete Shakiness & Smoothness Training Guide

## What You Have Now

✅ **Real-time shakiness detection** using tremor + jerk metrics
✅ **Handedness tracking** (left/right hand separation)
✅ **Analytics showing actual movement quality** (not fake multipliers)
✅ **Training mode** to collect labeled data
✅ **Data export** for model retraining
✅ **Left/Right hand comparison** UI

---

## How to Use Everything

### 1. 🎮 **Normal Use (Freestyle Page)**

Just use the app normally. It now tracks:
- **Tremor**: High-frequency shakiness (0.0001 to 0.01 range)
- **Smoothness**: Jerk/sudden movements (100 to 5000 range)
- **Which hand**: Left or Right

**Good scores:**
- Tremor < 0.001 = Very smooth ✅
- Smoothness < 800 = Very smooth ✅

**Bad scores:**
- Tremor > 0.005 = Shaky ⚠️
- Smoothness > 1500 = Jerky ⚠️

---

### 2. 🆚 **Compare Left vs Right Hands**

**Step 1:** Record session with LEFT hand only on Freestyle page
**Step 2:** Record session with RIGHT hand only
**Step 3:** Visit http://localhost:8080/hand-comparison

You'll see side-by-side comparison of shakiness!

---

### 3. 🎯 **Training Mode: Collect Labeled Data**

Want to improve the model's accuracy? Collect training data!

```bash
cd FlowState
python training_mode.py
```

**This will guide you through:**
1. Choose movement type: SMOOTH, SHAKY, or MIXED
2. Record 3 sessions (10 seconds each)
3. Data is automatically labeled and saved

**Pro tip:** 
- SMOOTH: Hold hand steady, move slowly and deliberately
- SHAKY: Intentionally shake/tremble your hand
- Record both types to train the model!

---

### 4. 📤 **Export Training Data**

After recording training sessions:

```bash
python export_training_data.py
```

**This creates:** `training_data_shakiness.csv`

Contains:
- 63 landmark features (21 joints × 3 coords)
- 4 movement metrics (velocity, smoothness, tremor, ROM)
- 15 joint angles
- **Label**: 0=smooth, 1=shaky, 2=mixed

---

### 5. 🤖 **Retrain the Model** (Future)

Once you have enough training data:

```bash
python train_pose_model.py  # Coming soon!
```

This will:
- Load `training_data_shakiness.csv`
- Train improved pose detection model
- Save as `pose_model_improved.pt`

**Minimum recommended data:**
- 500+ frames per label (smooth/shaky)
- Multiple sessions in different lighting
- Both hands recorded

---

## Testing Your Setup

### Test 1: Check Shakiness Detection

```bash
python test_shakiness_analytics.py
```

Shows:
- Latest session metrics
- Frame-by-frame tremor/smoothness
- Calculated scores
- Interpretation (smooth vs shaky)

### Test 2: Record Different Movement Types

1. **Smooth Test**: 
   - Go to Freestyle
   - Hold hand VERY steady
   - Move slowly
   - Check analytics → Should see high smoothness score (85-95)

2. **Shaky Test**:
   - Go to Freestyle  
   - Intentionally shake/tremble hand
   - Check analytics → Should see low smoothness score (30-50)

3. **Compare**:
   - Visit analytics page
   - See the difference in scores!

---

## File Reference

### New Files Created:

1. **`training_mode.py`** - Interactive training data collector
2. **`export_training_data.py`** - Exports data to CSV
3. **`test_shakiness_analytics.py`** - Test script
4. **`src/pages/HandComparison.tsx`** - Left/Right hand UI
5. **`TRAINING_PLAN.md`** - Full training documentation

### Modified Files:

1. **`app_with_data.py`** - Added handedness tracking + real tremor scoring
2. **`movement_features.py`** - Added handedness parameter
3. **`src/App.tsx`** - Added hand comparison route

---

## Quick Command Reference

```bash
# Test shakiness detection
python test_shakiness_analytics.py

# Collect training data
python training_mode.py

# Export to CSV
python export_training_data.py

# Start servers
cd FlowState
source ../.venv/bin/activate
python app_with_data.py  # Flask on :5001

# In another terminal:
npm run dev  # React on :8080
```

---

## Understanding the Metrics

### Tremor Score
- **What it measures**: High-frequency hand vibrations
- **Formula**: `100 - (tremor × 8000)`
- **Range**: 0-100 (higher = less shaky)
- **Good**: >90, **Bad**: <60

### Smoothness Score  
- **What it measures**: Sudden jerky movements (jerk = change in acceleration)
- **Formula**: `100 - (jerk / 50)`
- **Range**: 0-100 (higher = smoother)
- **Good**: >85, **Bad**: <50

### Combined Score
- **Formula**: `(tremor_score × 0.6) + (jerk_score × 0.4)`
- **Weight**: 60% tremor, 40% jerk
- **Why**: Tremor is more important for rehabilitation tracking

---

## Troubleshooting

### "All scores are the same"
- **Cause**: Not enough movement variation
- **Fix**: Record longer sessions (30+ seconds) with varied movements

### "No data for left hand"
- **Cause**: Haven't recorded with left hand yet
- **Fix**: Go to Freestyle, use ONLY left hand, record session

### "Training mode crashes"
- **Cause**: Camera access issues
- **Fix**: Check camera permissions, try `python camera_opencv.py` first

---

## Next Steps

1. ✅ **Test Now**: Run `python test_shakiness_analytics.py`
2. ✅ **Compare Hands**: Record with each hand, visit /hand-comparison
3. ✅ **Collect Training Data**: Run `python training_mode.py`
4. ⏳ **Train Model**: After 1000+ frames, retrain pose detection

---

## Summary

You now have:
- ✅ Real shakiness detection (not fake!)
- ✅ Left/Right hand tracking
- ✅ Training mode for data collection
- ✅ Data export for model retraining
- ✅ Visual comparison tools

**Try it:** Record a shaky session vs smooth session and see the scores change!
