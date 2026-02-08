"""
Quick Data Collection & Labeling Tool
Use this to populate the database with sample data for training
"""

from database import RehabDatabase
import uuid
from datetime import datetime, timedelta
import random
import json

def create_sample_users():
    """Create sample users with different rehab stages"""
    db = RehabDatabase('flowstate.db')
    
    users = [
        {
            'user_id': 'user_early_001',
            'name': 'Patient A (Early Stage)',
            'diagnosis': 'stroke',
            'rehab_stage': 'early',
            'notes': 'Week 2 of rehabilitation'
        },
        {
            'user_id': 'user_mid_001',
            'name': 'Patient B (Mid Stage)',
            'diagnosis': 'stroke',
            'rehab_stage': 'mid',
            'notes': 'Week 6 of rehabilitation'
        },
        {
            'user_id': 'user_advanced_001',
            'name': 'Patient C (Advanced)',
            'diagnosis': 'carpal_tunnel',
            'rehab_stage': 'advanced',
            'notes': 'Week 12 of rehabilitation'
        },
    ]
    
    print("Creating sample users...")
    for user in users:
        db.create_user(**user)
        print(f"  ✓ {user['name']}")
    
    db.close()
    return [u['user_id'] for u in users]


def generate_synthetic_session(db, user_id, rehab_stage, session_num):
    """
    Generate synthetic session data based on rehab stage
    This simulates what real recorded data would look like
    """
    
    # Session characteristics vary by rehab stage
    stage_params = {
        'early': {
            'velocity_mean': random.uniform(0.05, 0.15),
            'smoothness': random.uniform(0.3, 0.5),
            'tremor': random.uniform(0.4, 0.7),
            'rom': random.uniform(0.2, 0.4),
            'frames': random.randint(150, 300)
        },
        'mid': {
            'velocity_mean': random.uniform(0.15, 0.30),
            'smoothness': random.uniform(0.5, 0.7),
            'tremor': random.uniform(0.2, 0.4),
            'rom': random.uniform(0.4, 0.6),
            'frames': random.randint(200, 400)
        },
        'advanced': {
            'velocity_mean': random.uniform(0.25, 0.45),
            'smoothness': random.uniform(0.7, 0.9),
            'tremor': random.uniform(0.05, 0.2),
            'rom': random.uniform(0.6, 0.9),
            'frames': random.randint(300, 500)
        }
    }
    
    params = stage_params[rehab_stage]
    
    # Create session
    session_id = f'session_{rehab_stage}_{session_num}_{uuid.uuid4().hex[:8]}'
    start_time = datetime.now() - timedelta(days=30-session_num)
    
    db.create_session(
        session_id=session_id,
        user_id=user_id,
        level_name='hand_exercise',
        start_time=start_time.isoformat()
    )
    
    # Generate time series data
    num_frames = params['frames']
    frame_data = []
    
    for i in range(num_frames):
        # Simulate landmark positions (simplified)
        landmarks = [
            {'x': random.random(), 'y': random.random(), 'z': random.random()}
            for _ in range(21)
        ]
        
        # Add some noise based on tremor level
        tremor_noise = params['tremor'] * 0.1
        
        frame = {
            'timestamp_ms': i * 33,  # ~30fps
            'features': {
                'landmark_positions': landmarks,
                'joint_angles': {
                    'thumb': random.uniform(0, 90),
                    'index': random.uniform(0, 90),
                    'middle': random.uniform(0, 90),
                    'ring': random.uniform(0, 90),
                    'pinky': random.uniform(0, 90),
                    'wrist': random.uniform(-30, 30)
                },
                'hand_openness': random.uniform(0.3, 0.9),
                'velocity': {
                    'mean_speed': params['velocity_mean'] + random.gauss(0, 0.05)
                },
                'acceleration': {
                    'mean_acceleration': random.uniform(0.01, 0.3)
                },
                'smoothness': params['smoothness'] + random.gauss(0, 0.1),
                'tremor_score': params['tremor'] + random.gauss(0, tremor_noise),
                'range_of_motion': {
                    'mean_rom': params['rom'] + random.gauss(0, 0.05)
                }
            }
        }
        frame_data.append(frame)
    
    # Save time series data
    db.save_frame_data(session_id, frame_data)
    
    # Update session with completion data
    duration = num_frames / 30.0
    db.update_session(
        session_id=session_id,
        end_time=(start_time + timedelta(seconds=duration)).isoformat(),
        duration_seconds=duration,
        total_frames=num_frames,
        completed=True,
        avg_speed=params['velocity_mean'],
        max_speed=params['velocity_mean'] * 1.5,
        avg_smoothness=params['smoothness'],
        avg_tremor=params['tremor'],
        stars_earned=3 if rehab_stage == 'advanced' else 2 if rehab_stage == 'mid' else 1
    )
    
    return session_id


def populate_database(sessions_per_user=5):
    """
    Populate database with synthetic training data
    
    Args:
        sessions_per_user: Number of sessions to create per user
    """
    print("\n" + "="*60)
    print("  FlowState - Data Population Tool")
    print("="*60 + "\n")
    
    # Create users
    user_ids = create_sample_users()
    
    print(f"\nGenerating {sessions_per_user} sessions per user...")
    
    db = RehabDatabase('flowstate.db')
    
    # Get rehab stage for each user
    cursor = db.conn.cursor()
    
    total_sessions = 0
    
    for user_id in user_ids:
        cursor.execute('SELECT rehab_stage, name FROM users WHERE user_id = ?', (user_id,))
        rehab_stage, name = cursor.fetchone()
        
        print(f"\n{name}:")
        
        for i in range(sessions_per_user):
            session_id = generate_synthetic_session(db, user_id, rehab_stage, i+1)
            print(f"  ✓ Session {i+1}/{sessions_per_user}")
            total_sessions += 1
    
    db.close()
    
    print("\n" + "="*60)
    print(f"✓ Created {total_sessions} training sessions")
    print("="*60)
    print("\nYou can now run: python train_models.py")
    print()


if __name__ == '__main__':
    import sys
    
    # Get number of sessions from command line
    sessions = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    if sessions < 3:
        print("ERROR: Need at least 3 sessions per user")
        sys.exit(1)
    
    if sessions > 20:
        print(f"⚠️  Creating {sessions} sessions per user (this may take a minute)")
    
    populate_database(sessions_per_user=sessions)
