"""
Show Session Data - Analyze and display the most recent recording
Use this to show me good vs bad examples
"""

import json
from database import RehabDatabase
import sys

def analyze_latest_session(label="Movement"):
    """Analyze and display the most recent session data"""
    db = RehabDatabase('flowstate.db')
    cursor = db.conn.cursor()
    
    # Get latest session
    cursor.execute('''
        SELECT session_id, start_time, total_frames, avg_speed, avg_smoothness, avg_tremor
        FROM sessions 
        WHERE completed = 1 AND total_frames > 0
        ORDER BY start_time DESC LIMIT 1
    ''')
    
    session = cursor.fetchone()
    if not session:
        print("❌ No sessions found!")
        return
    
    session_id, start_time, frames, speed, smoothness, tremor = session
    
    print("\n" + "="*70)
    print(f"📊 {label.upper()} - Session Analysis")
    print("="*70)
    print(f"\n🆔 Session: {session_id}")
    print(f"⏰ Time: {start_time}")
    print(f"📹 Frames: {frames:,}")
    
    print(f"\n📈 RAW METRICS:")
    print(f"   Speed (velocity):     {speed:.4f}")
    print(f"   Smoothness (SPARC):   {smoothness:.2f}")
    print(f"   Tremor:               {tremor:.6f}")
    
    # Get frame-level details for deeper analysis
    cursor.execute('''
        SELECT velocity_mean, smoothness_score, tremor_score, range_of_motion
        FROM movement_timeseries 
        WHERE session_id = ?
        ORDER BY frame_number
    ''', (session_id,))
    
    frame_data = cursor.fetchall()
    if frame_data:
        velocities = [f[0] for f in frame_data if f[0] is not None]
        smoothness_scores = [f[1] for f in frame_data if f[1] is not None]
        tremors = [f[2] for f in frame_data if f[2] is not None]
        roms = [f[3] for f in frame_data if f[3] is not None]
        
        print(f"\n📊 DETAILED FRAME ANALYSIS ({len(frame_data):,} frames):")
        if velocities:
            print(f"   Velocity:   min={min(velocities):.4f}, max={max(velocities):.4f}, avg={sum(velocities)/len(velocities):.4f}")
        if smoothness_scores:
            print(f"   Smoothness: min={min(smoothness_scores):.2f}, max={max(smoothness_scores):.2f}, avg={sum(smoothness_scores)/len(smoothness_scores):.2f}")
        if tremors:
            print(f"   Tremor:     min={min(tremors):.6f}, max={max(tremors):.6f}, avg={sum(tremors)/len(tremors):.6f}")
        if roms:
            print(f"   ROM:        min={min(roms):.4f}, max={max(roms):.4f}, avg={sum(roms)/len(roms):.4f}")
    
    # Categorize movement quality
    print(f"\n🎯 QUALITY ASSESSMENT:")
    
    # Speed assessment
    if speed > 0.6:
        speed_quality = "🟢 EXCELLENT (Fast)"
    elif speed > 0.4:
        speed_quality = "🟡 GOOD (Moderate)"
    elif speed > 0.25:
        speed_quality = "🟠 BELOW AVERAGE (Slow)"
    else:
        speed_quality = "🔴 POOR (Very Slow)"
    print(f"   Speed:      {speed_quality}")
    
    # Smoothness assessment (lower is better)
    if smoothness < 600:
        smooth_quality = "🟢 EXCELLENT (Very Smooth)"
    elif smoothness < 900:
        smooth_quality = "🟡 GOOD (Smooth)"
    elif smoothness < 1500:
        smooth_quality = "🟠 BELOW AVERAGE (Some Jerkiness)"
    else:
        smooth_quality = "🔴 POOR (Very Jerky)"
    print(f"   Smoothness: {smooth_quality}")
    
    # Tremor assessment (lower is better)
    if tremor < 0.0008:
        tremor_quality = "🟢 EXCELLENT (Very Stable)"
    elif tremor < 0.0015:
        tremor_quality = "🟡 GOOD (Stable)"
    elif tremor < 0.003:
        tremor_quality = "🟠 BELOW AVERAGE (Some Shake)"
    else:
        tremor_quality = "🔴 POOR (Very Shaky)"
    print(f"   Tremor:     {tremor_quality}")
    
    # Overall prediction
    score = 0
    if speed > 0.6: score += 3
    elif speed > 0.4: score += 2
    elif speed > 0.25: score += 1
    
    if smoothness < 600: score += 3
    elif smoothness < 900: score += 2
    elif smoothness < 1500: score += 1
    
    if tremor < 0.0008: score += 3
    elif tremor < 0.0015: score += 2
    elif tremor < 0.003: score += 1
    
    if score >= 8:
        overall = "🟢 EXCELLENT - Expected Analytics Scores: 85-100"
    elif score >= 6:
        overall = "🟡 GOOD - Expected Analytics Scores: 65-85"
    elif score >= 4:
        overall = "🟠 MODERATE - Expected Analytics Scores: 45-65"
    else:
        overall = "🔴 POOR - Expected Analytics Scores: 10-45"
    
    print(f"\n🏆 OVERALL: {overall}")
    print("="*70 + "\n")
    
    return {
        'session_id': session_id,
        'speed': speed,
        'smoothness': smoothness,
        'tremor': tremor,
        'frames': frames,
        'quality_score': score
    }

if __name__ == '__main__':
    label = sys.argv[1] if len(sys.argv) > 1 else "Latest Movement"
    analyze_latest_session(label)
