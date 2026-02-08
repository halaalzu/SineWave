"""
Real Data Collection Workflow
Start the web app and record real hand movements for training
"""

from database import RehabDatabase
import sys

def setup_real_user():
    """Create a user for real data collection"""
    db = RehabDatabase('flowstate.db')
    
    print("\n" + "="*60)
    print("  FlowState - Real Data Collection Setup")
    print("="*60 + "\n")
    
    print("Let's create your user profile for data collection:\n")
    
    # Get user info
    name = input("Your name: ").strip()
    if not name:
        name = "Test User"
    
    print("\nRehabilitation stage:")
    print("  1. Early (high tremor, low control)")
    print("  2. Mid (improving control)")
    print("  3. Advanced (good control)")
    choice = input("Choose stage [1-3]: ").strip()
    
    stage_map = {'1': 'early', '2': 'mid', '3': 'advanced'}
    rehab_stage = stage_map.get(choice, 'mid')
    
    diagnosis = input("\nDiagnosis (optional, e.g., 'stroke', 'carpal tunnel'): ").strip()
    
    user_id = name.lower().replace(' ', '_')
    
    # Create user
    db.create_user(
        user_id=user_id,
        name=name,
        diagnosis=diagnosis or None,
        rehab_stage=rehab_stage,
        notes='Real data collection user'
    )
    
    print(f"\n✓ User created: {name} ({rehab_stage} stage)")
    print(f"  User ID: {user_id}")
    
    db.close()
    
    print("\n" + "="*60)
    print("  Next Steps:")
    print("="*60)
    print("\n1. Start the web app:")
    print("     python app_with_data.py")
    print("\n2. Open browser: http://localhost:5001")
    print("\n3. Record sessions:")
    print("     - Press 'R' to start recording")
    print("     - Perform hand movements")
    print("     - Press 'S' to stop recording")
    print("     - Repeat 10-20 times")
    print("\n4. Train the model:")
    print("     python train_models.py")
    print("\n5. Optional - Create more users at different stages")
    print("     python setup_real_data.py")
    print()
    
    return user_id

def check_real_data():
    """Check how much real data has been collected"""
    db = RehabDatabase('flowstate.db')
    
    cursor = db.conn.cursor()
    
    # Get counts
    cursor.execute("SELECT COUNT(*) FROM users WHERE notes NOT LIKE '%synthetic%'")
    real_users = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM sessions WHERE session_id NOT LIKE 'session_early_%' AND session_id NOT LIKE 'session_mid_%' AND session_id NOT LIKE 'session_advanced_%'")
    real_sessions = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM sessions")
    total_sessions = cursor.fetchone()[0]
    
    print("\n" + "="*60)
    print("  Current Database Status")
    print("="*60)
    print(f"\nReal users: {real_users}")
    print(f"Real sessions: {real_sessions}")
    print(f"Synthetic sessions: {total_sessions - real_sessions}")
    print(f"Total sessions: {total_sessions}")
    
    if real_sessions < 10:
        print("\n⚠️  You need at least 10-20 real sessions for training")
        print("   Start the web app and record more movements!")
    elif real_sessions < 20:
        print("\n✓ Good start! Collect a few more sessions for better accuracy")
    else:
        print("\n✓ Great! You have enough data to train a model")
    
    # Show sessions by user
    cursor.execute('''
        SELECT u.name, u.rehab_stage, COUNT(s.session_id) as session_count
        FROM users u
        LEFT JOIN sessions s ON u.user_id = s.user_id
        WHERE u.notes NOT LIKE '%synthetic%' OR u.notes IS NULL
        GROUP BY u.user_id, u.name, u.rehab_stage
    ''')
    
    real_user_sessions = cursor.fetchall()
    
    if real_user_sessions:
        print("\nSessions by user:")
        for name, stage, count in real_user_sessions:
            print(f"  {name} ({stage}): {count} sessions")
    
    db.close()
    print()

def clear_synthetic_data():
    """Remove synthetic test data"""
    db = RehabDatabase('flowstate.db')
    
    print("\n⚠️  This will delete ALL synthetic test data")
    confirm = input("Are you sure? [y/N]: ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled")
        return
    
    cursor = db.conn.cursor()
    
    # Delete synthetic sessions
    cursor.execute('''
        DELETE FROM movement_timeseries 
        WHERE session_id IN (
            SELECT session_id FROM sessions 
            WHERE session_id LIKE 'session_early_%' 
            OR session_id LIKE 'session_mid_%' 
            OR session_id LIKE 'session_advanced_%'
        )
    ''')
    
    cursor.execute('''
        DELETE FROM sessions 
        WHERE session_id LIKE 'session_early_%' 
        OR session_id LIKE 'session_mid_%' 
        OR session_id LIKE 'session_advanced_%'
    ''')
    
    # Delete synthetic users
    cursor.execute("DELETE FROM users WHERE user_id LIKE 'user_%_001'")
    
    db.conn.commit()
    
    print("✓ Synthetic data deleted")
    print("  Your real recorded sessions are safe!")
    
    db.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'check':
            check_real_data()
        elif sys.argv[1] == 'clear':
            clear_synthetic_data()
        else:
            print("Usage:")
            print("  python setup_real_data.py        - Create new user")
            print("  python setup_real_data.py check  - Check data status")
            print("  python setup_real_data.py clear  - Delete synthetic data")
    else:
        setup_real_user()
