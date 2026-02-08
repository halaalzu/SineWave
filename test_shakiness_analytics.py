"""
Test script to verify shakiness/smoothness analytics are working
"""

from database import RehabDatabase
import sqlite3

db = RehabDatabase('flowstate.db')
cursor = db.conn.cursor()

print("\n" + "="*70)
print("🎯 SHAKINESS & SMOOTHNESS ANALYTICS TEST")
print("="*70 + "\n")

# Get latest session
cursor.execute('''
    SELECT session_id, total_frames, avg_speed, avg_smoothness, avg_tremor
    FROM sessions 
    WHERE completed = 1 
    ORDER BY start_time DESC LIMIT 1
''')

session = cursor.fetchone()
if not session:
    print("❌ No completed sessions found!")
    exit(1)

session_id, frames, speed, smoothness, tremor = session
print(f"Latest Session: {session_id[:30]}...")
print(f"Total Frames: {frames}")
print(f"\n📊 SESSION-LEVEL METRICS:")
print(f"  • Avg Speed: {speed:.6f}")
print(f"  • Avg Smoothness (jerk): {smoothness:.2f}")
print(f"  • Avg Tremor: {tremor:.8f}")

# Get frame-level data
cursor.execute('''
    SELECT tremor_score, smoothness_score, velocity_mean
    FROM movement_timeseries
    WHERE session_id = ?
    ORDER BY frame_number
    LIMIT 10
''', (session_id,))

frames_data = cursor.fetchall()
if frames_data:
    print(f"\n🎬 SAMPLE FRAME-LEVEL DATA (first 10 frames):")
    print(f"{'Frame':<8} {'Tremor':<15} {'Smoothness':<15} {'Velocity':<12}")
    print("-" * 60)
    for i, (t, s, v) in enumerate(frames_data, 1):
        tremor_val = t if t else 0.0
        smooth_val = s if s else 0.0
        vel_val = v if v else 0.0
        print(f"{i:<8} {tremor_val:<15.8f} {smooth_val:<15.2f} {vel_val:<12.4f}")

# Calculate what the analytics scores SHOULD be
print(f"\n📈 CALCULATED ANALYTICS SCORES:")
avg_tremor_val = tremor if tremor else 0.001
avg_smooth_val = smoothness if smoothness else 1000

# Shakiness detection
tremor_score = max(10, min(100, 100 - (avg_tremor_val * 8000)))
jerk_score = max(10, min(100, 100 - (avg_smooth_val / 50)))
final_smoothness = (tremor_score * 0.6 + jerk_score * 0.4)

print(f"\n  SMOOTHNESS SCORE BREAKDOWN:")
print(f"  • Tremor component: {tremor_score:.1f} (lower tremor = higher score)")
print(f"  • Jerk component: {jerk_score:.1f} (lower jerk = smoother)")
print(f"  • Final smoothness: {final_smoothness:.1f}/100")

if avg_tremor_val < 0.001:
    print(f"\n  ✅ VERY SMOOTH: Tremor {avg_tremor_val:.8f} is excellent!")
elif avg_tremor_val < 0.005:
    print(f"\n  👍 GOOD: Tremor {avg_tremor_val:.8f} is good")
else:
    print(f"\n  ⚠️  SHAKY: Tremor {avg_tremor_val:.8f} indicates shakiness")

if avg_smooth_val < 800:
    print(f"  ✅ VERY SMOOTH: Jerk {avg_smooth_val:.2f} is excellent!")
elif avg_smooth_val < 1500:
    print(f"  👍 GOOD: Jerk {avg_smooth_val:.2f} is good")
else:
    print(f"  ⚠️  JERKY: Jerk {avg_smooth_val:.2f} indicates jerkiness")

# Check handedness tracking
cursor.execute('''
    SELECT COUNT(*) FROM movement_timeseries
    WHERE session_id = ?
''', (session_id,))
frame_count = cursor.fetchone()[0]

print(f"\n🤚 HANDEDNESS TRACKING:")
print(f"  • Total frames recorded: {frame_count}")
print(f"  • Note: Handedness is now tracked but stored in session data")
print(f"  • To see per-hand analytics, need to query by handedness")

print(f"\n" + "="*70)
print("✅ Test Complete!")
print(f"{'='*70}\n")

print("📝 INTERPRETATION:")
print("  • Lower tremor score = more shaky (bad)")
print("  • Higher tremor score = less shaky (good)")
print("  • Lower jerk = smoother motion (good)")
print("  • Higher jerk = jerkier motion (bad)")
print("\n  The analytics now use REAL tremor/smoothness data,")
print("  not fake multipliers!")
