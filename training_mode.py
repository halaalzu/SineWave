"""
Training Mode: Collect labeled data for shakiness vs smoothness
Run this to deliberately record shaky and smooth movements
"""

import cv2
import time
import sys
from database import RehabDatabase
from movement_features import SessionRecorder

def training_mode():
    """Interactive training mode to collect labeled data"""
    
    print("\n" + "="*70)
    print("🎯 TRAINING MODE: Shakiness vs Smoothness")
    print("="*70)
    print("\nThis mode helps you collect labeled training data:")
    print("  • SMOOTH movements: Hold hand steady, move slowly")
    print("  • SHAKY movements: Intentionally shake your hand")
    print("\nThis data will train the model to better detect shakiness!")
    print("="*70 + "\n")
    
    # Choose training type
    print("What type of movement do you want to record?")
    print("  1. SMOOTH (steady, controlled)")
    print("  2. SHAKY (intentional tremor)")
    print("  3. MIXED (normal use)")
    
    choice = input("\nEnter 1, 2, or 3: ").strip()
    
    if choice == "1":
        label = "smooth"
        instructions = "Hold your hand STEADY and move SLOWLY"
    elif choice == "2":
        label = "shaky"
        instructions = "SHAKE your hand with intentional tremor"
    elif choice == "3":
        label = "mixed"
        instructions = "Move naturally as you normally would"
    else:
        print("Invalid choice!")
        return
    
    print(f"\n✅ Recording {label.upper()} movements")
    print(f"📝 Instructions: {instructions}")
    print("\nPress ENTER to start recording (or Ctrl+C to cancel)...")
    input()
    
    # Record 3 sessions of this type
    for i in range(3):
        print(f"\n🎬 Session {i+1}/3")
        print(f"Instructions: {instructions}")
        print("Press ENTER to start, then perform the movement for 10 seconds...")
        input()
        
        print("🔴 RECORDING... (10 seconds)")
        
        # Import the tracker
        import sys
        sys.path.insert(0, '.')
        from app_with_data import HandTracker
        
        tracker = HandTracker()
        
        # Start session with label
        session_id = tracker.start_session("training_user", f"training_{label}")
        
        # Record for 10 seconds
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_timestamp = int((time.time() - start_time) * 1000)
            frame = tracker.process_frame(frame, frame_timestamp)
            
            # Show countdown
            remaining = 10 - int(time.time() - start_time)
            cv2.putText(frame, f"Recording {label.upper()}: {remaining}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Training Mode', frame)
            
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Stop session
        tracker.stop_session()
        
        print(f"✅ Session {i+1} complete! Recorded {frame_count} frames")
        print(f"   Session ID: {session_id}")
        
        # Add label to database
        db = RehabDatabase('flowstate.db')
        cursor = db.conn.cursor()
        cursor.execute('''
            UPDATE sessions 
            SET level_name = ? 
            WHERE session_id = ?
        ''', (f"training_{label}", session_id))
        db.conn.commit()
        
        if i < 2:
            print("\nGet ready for next session...")
            time.sleep(2)
    
    print("\n" + "="*70)
    print(f"✅ Training complete! Recorded 3 {label.upper()} sessions")
    print("="*70)
    
    # Show summary
    db = RehabDatabase('flowstate.db')
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT level_name, COUNT(*), AVG(avg_tremor), AVG(avg_smoothness)
        FROM sessions
        WHERE level_name LIKE "training_%"
        GROUP BY level_name
    ''')
    
    print("\n📊 TRAINING DATA SUMMARY:")
    for row in cursor.fetchall():
        label_type, count, avg_tremor, avg_smooth = row
        print(f"\n  {label_type}:")
        print(f"    • Sessions: {count}")
        print(f"    • Avg tremor: {avg_tremor:.8f}")
        print(f"    • Avg smoothness: {avg_smooth:.2f}")
    
    print("\n💡 Next steps:")
    print("  1. Record more sessions with different labels")
    print("  2. Run: python export_training_data.py")
    print("  3. Run: python train_pose_model.py")
    print()

if __name__ == "__main__":
    try:
        training_mode()
    except KeyboardInterrupt:
        print("\n\n❌ Training cancelled")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
