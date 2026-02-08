"""
Export training data for model retraining
Creates CSV files with shakiness/smoothness labels
"""

import json
import csv
from database import RehabDatabase
import numpy as np

def export_training_data():
    """Export all training sessions to CSV for model training"""
    
    print("\n" + "="*70)
    print("📤 EXPORTING TRAINING DATA")
    print("="*70 + "\n")
    
    db = RehabDatabase('flowstate.db')
    cursor = db.conn.cursor()
    
    # Get all training sessions
    cursor.execute('''
        SELECT session_id, level_name, avg_tremor, avg_smoothness, total_frames
        FROM sessions
        WHERE level_name LIKE "training_%"
        ORDER BY start_time
    ''')
    
    training_sessions = cursor.fetchall()
    
    if not training_sessions:
        print("❌ No training sessions found!")
        print("   Run: python training_mode.py first")
        return
    
    print(f"✅ Found {len(training_sessions)} training sessions\n")
    
    # Export frame-level data
    all_frames = []
    
    for session_id, level_name, avg_tremor, avg_smooth, frames in training_sessions:
        # Extract label
        if "smooth" in level_name:
            label = 0  # Smooth = 0
        elif "shaky" in level_name:
            label = 1  # Shaky = 1
        else:
            label = 2  # Mixed = 2
        
        print(f"Processing: {session_id[:30]}... ({level_name}, {frames} frames)")
        
        # Get frame data
        cursor.execute('''
            SELECT landmark_positions, joint_angles, velocity_mean, 
                   smoothness_score, tremor_score, range_of_motion
            FROM movement_timeseries
            WHERE session_id = ?
        ''', (session_id,))
        
        frames_data = cursor.fetchall()
        
        for frame_row in frames_data:
            try:
                landmarks = json.loads(frame_row[0]) if frame_row[0] else None
                angles = json.loads(frame_row[1]) if frame_row[1] else {}
                velocity = frame_row[2] or 0
                smoothness = frame_row[3] or 0
                tremor = frame_row[4] or 0
                rom = frame_row[5] or 0
                
                if landmarks and len(landmarks) == 21:
                    # Flatten landmarks (21 points × 3 coords = 63 features)
                    features = []
                    for landmark in landmarks:
                        features.extend([landmark[0], landmark[1], landmark[2]])
                    
                    # Add movement metrics
                    features.extend([velocity, smoothness, tremor, rom])
                    
                    # Add angles (15 features)
                    angle_keys = ['thumb_mcp', 'index_mcp', 'middle_mcp', 'ring_mcp', 'pinky_mcp',
                                  'thumb_ip', 'index_pip', 'middle_pip', 'ring_pip', 'pinky_pip',
                                  'index_dip', 'middle_dip', 'ring_dip', 'pinky_dip', 'thumb_tip']
                    for key in angle_keys:
                        features.append(angles.get(key, 0))
                    
                    # Add label
                    features.append(label)
                    
                    all_frames.append(features)
            
            except (json.JSONDecodeError, TypeError, IndexError):
                continue
    
    # Save to CSV
    if all_frames:
        filename = 'training_data_shakiness.csv'
        
        # Create header
        header = []
        # Landmark features
        for i in range(21):
            header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
        # Movement metrics
        header.extend(['velocity', 'smoothness', 'tremor', 'rom'])
        # Angles
        angle_keys = ['thumb_mcp', 'index_mcp', 'middle_mcp', 'ring_mcp', 'pinky_mcp',
                      'thumb_ip', 'index_pip', 'middle_pip', 'ring_pip', 'pinky_pip',
                      'index_dip', 'middle_dip', 'ring_dip', 'pinky_dip', 'thumb_tip']
        header.extend(angle_keys)
        # Label
        header.append('label')
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_frames)
        
        print(f"\n✅ Exported {len(all_frames)} frames to {filename}")
        print(f"\nFile structure:")
        print(f"  • Total features: {len(header) - 1}")
        print(f"  • Labels: 0=smooth, 1=shaky, 2=mixed")
        print(f"  • Total rows: {len(all_frames)}")
        
        # Show label distribution
        labels = [frame[-1] for frame in all_frames]
        print(f"\n📊 Label distribution:")
        print(f"  • Smooth (0): {labels.count(0)} frames")
        print(f"  • Shaky (1): {labels.count(1)} frames")
        print(f"  • Mixed (2): {labels.count(2)} frames")
        
        print(f"\n💡 Next step: python train_pose_model.py")
    else:
        print("❌ No valid frame data found!")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    export_training_data()
