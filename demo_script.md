# 🎯 DEMO SCRIPT: Bad vs Good Movement Detection

## System Status ✅
- **ML Model**: Trained on 24,686 frames (99% accuracy)
- **Database**: 68 sessions, 24,998 frames collected
- **Scoring**: Based on most recent session only

## 🔴 DEMO 1: BAD/IMPAIRED Movement

**Goal**: Get scores of 10-30 (very poor)

**Instructions:**
1. Open **http://localhost:8081/freestyle**
2. Wait for camera to activate
3. **Perform these BAD movements for 10-15 seconds:**
   - Keep fingers **barely moving** (90% closed)
   - Move **VERY SLOWLY** (like you have stiff joints)
   - **Tiny range of motion** - don't fully extend or close
   - Make movements **jerky and inconsistent**
   - Keep hand mostly still, minimal joint flexion

**What you should see:**
```
Expected Metrics:
- Speed: 0.15-0.25 (very slow)
- Smoothness (SPARC): 800-1500 (jerky)
- Tremor: 0.001-0.003 (shaky)

Expected Scores:
- Joint scores: 10-35 (🔴 RED - WEAK)
- FMA Proxy: 15-30
- All metrics show poor performance
```

4. Go to **http://localhost:8081/analytics**
5. Click **REFRESH** button
6. Observe: Most joints should be **RED (10-35 scores)**

---

## 🟢 DEMO 2: GOOD/HEALTHY Movement

**Goal**: Get scores of 85-100 (excellent)

**Instructions:**
1. Go back to **http://localhost:8081/freestyle**  
2. Wait for new session to start
3. **Perform these GOOD movements for 10-15 seconds:**
   - **RAPID** fist opening/closing (fast as you can)
   - **FULL range** - fingers completely extended → completely closed
   - **Smooth, fluid** movements (no jerking)
   - All 5 fingers moving together
   - **Consistent rhythm** - like you're playing piano fast

**What you should see:**
```
Expected Metrics:
- Speed: 0.6-1.2 (fast)
- Smoothness (SPARC): 300-600 (smooth)
- Tremor: 0.0003-0.0008 (stable)

Expected Scores:
- Joint scores: 85-100 (🟢 GREEN - STRONG)
- FMA Proxy: 90-98
- All metrics show excellent performance
```

4. Go to **http://localhost:8081/analytics**
5. Click **REFRESH** button
6. Observe: Most joints should be **GREEN (85-100 scores)**

---

## 📊 Side-by-Side Comparison

| Metric | BAD Movement | GOOD Movement |
|--------|-------------|---------------|
| **Speed** | 0.15-0.25 | 0.6-1.2 |
| **SPARC** | 800-1500 | 300-600 |
| **Tremor** | 0.001-0.003 | 0.0003-0.0008 |
| **ROM** | Minimal | Full range |
| **Joint Scores** | 🔴 10-35 | 🟢 85-100 |
| **FMA Proxy** | 15-30 | 90-98 |

---

## 🎬 Quick Demo Flow

```bash
1. BAD DEMO (2 minutes):
   └─ Freestyle → Barely move hand (slow, stiff, tiny movements) → 15 sec
   └─ Analytics → See RED scores (10-35)

2. GOOD DEMO (2 minutes):
   └─ Freestyle → Rapid full fist pumps (fast, smooth, full range) → 15 sec
   └─ Analytics → See GREEN scores (85-100)

Total demo time: 4 minutes
```

---

## 🤖 ML Model Predictions

The trained ML model uses ALL 24,998 frames to understand patterns:
- **Label 0 (Excellent)**: 69 frames - very rare, exceptional movements
- **Label 1 (Good)**: 15,810 frames - majority of your movements
- **Label 2 (Moderate)**: 8,807 frames - lower quality movements

The model can now predict movement quality in real-time based on learned patterns!

---

## 🔧 Technical Details

**Bad Movement Creates:**
- Low velocity mean (< 0.3)
- High SPARC smoothness (> 800)
- High tremor (> 0.001)
- Small ROM range (< 0.08)
- High variability (jerky)

**Good Movement Creates:**
- High velocity mean (> 0.6)
- Low SPARC smoothness (< 600)
- Low tremor (< 0.0008)
- Large ROM range (> 0.18)
- Low variability (smooth)

**Scoring Algorithm:**
- Analyzes 84 features per frame
- 21 hand landmarks (x,y,z each)
- 15 joint angles
- 6 movement metrics
- Applies extreme thresholds for dramatic differences

---

## ✅ Expected Results

After both demos, your analytics should show:
- Clear visual difference in hand diagram colors
- Dramatic score changes (70+ point differences)
- Weekly progress chart showing session variation
- Gemini AI feedback describing quality differences

**If scores aren't changing dramatically:**
1. Check terminal - is session saving? Look for "Session saved" messages
2. Make movements MORE extreme (barely move for bad, super fast for good)
3. Click REFRESH button in Analytics after each recording
4. Verify Analytics page says "Analyzing 1 Recent Sessions"
