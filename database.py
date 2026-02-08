"""
Database schemas and management for rehabilitation tracking
Supports both SQLite (local) and PostgreSQL (production)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

class RehabDatabase:
    """Database manager for rehabilitation tracking data"""
    
    def __init__(self, db_path='flowstate.db'):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self.setup_database()
    
    def setup_database(self):
        """Create database tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                diagnosis TEXT,
                rehab_stage TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')
        
        # Sessions table (metadata)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                level_name TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_seconds REAL,
                total_frames INTEGER,
                stars_earned INTEGER DEFAULT 0,
                completed BOOLEAN DEFAULT 0,
                avg_speed REAL,
                max_speed REAL,
                avg_smoothness REAL,
                avg_tremor REAL,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Time series data table (detailed frame-by-frame)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS movement_timeseries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                frame_number INTEGER,
                landmark_positions TEXT,
                joint_angles TEXT,
                hand_openness REAL,
                velocity_mean REAL,
                velocity_max REAL,
                acceleration_mean REAL,
                smoothness_score REAL,
                tremor_score REAL,
                range_of_motion REAL,
                event_type TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Create index for faster time series queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timeseries_session 
            ON movement_timeseries(session_id, timestamp_ms)
        ''')
        
        # Session events table (markers like pose_start, pose_complete)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Baseline/reference movements for comparison
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS baseline_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movement_name TEXT NOT NULL,
                category TEXT,
                reference_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        ''')
        
        # Progress milestones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS progress_milestones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                milestone_type TEXT NOT NULL,
                achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                metric_value REAL,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        self.conn.commit()
    
    def create_user(self, user_id, name, diagnosis=None, rehab_stage=None, notes=None):
        """Create a new user"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO users (user_id, name, diagnosis, rehab_stage, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, name, diagnosis, rehab_stage, notes))
        self.conn.commit()
        return user_id
    
    def create_session(self, session_id, user_id, level_name, start_time):
        """Create a new session"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (session_id, user_id, level_name, start_time)
            VALUES (?, ?, ?, ?)
        ''', (session_id, user_id, level_name, start_time))
        self.conn.commit()
        return session_id
    
    def update_session(self, session_id, **kwargs):
        """Update session with completion data"""
        cursor = self.conn.cursor()
        
        # Build dynamic UPDATE query
        fields = []
        values = []
        for key, value in kwargs.items():
            fields.append(f"{key} = ?")
            values.append(value)
        
        values.append(session_id)
        query = f"UPDATE sessions SET {', '.join(fields)} WHERE session_id = ?"
        
        cursor.execute(query, values)
        self.conn.commit()
    
    def save_frame_data(self, session_id, frame_data):
        """Save time series frame data"""
        cursor = self.conn.cursor()
        
        for i, frame in enumerate(frame_data):
            features = frame['features']
            
            cursor.execute('''
                INSERT INTO movement_timeseries (
                    session_id, timestamp_ms, frame_number,
                    landmark_positions, joint_angles, hand_openness,
                    velocity_mean, velocity_max, acceleration_mean,
                    smoothness_score, tremor_score, range_of_motion, event_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                frame['timestamp_ms'],
                i,
                json.dumps(features.get('landmark_positions', [])),
                json.dumps(features.get('joint_angles', {})),
                features.get('hand_openness'),
                features.get('velocity', {}).get('mean_speed') if features.get('velocity') else None,
                features.get('velocity', {}).get('max_speed') if features.get('velocity') else None,
                features.get('acceleration', {}).get('mean_acceleration') if features.get('acceleration') else None,
                features.get('smoothness'),
                features.get('tremor_score'),
                features.get('range_of_motion', {}).get('mean_rom') if features.get('range_of_motion') else None,
                frame.get('event_type')
            ))
        
        self.conn.commit()
    
    def save_session_events(self, session_id, events):
        """Save session events"""
        cursor = self.conn.cursor()
        
        for event in events:
            cursor.execute('''
                INSERT INTO session_events (session_id, timestamp, event_type, event_data)
                VALUES (?, ?, ?, ?)
            ''', (
                session_id,
                event['timestamp'],
                event['event_type'],
                json.dumps(event.get('data', {}))
            ))
        
        self.conn.commit()
    
    def get_user_sessions(self, user_id, limit=None):
        """Get all sessions for a user"""
        cursor = self.conn.cursor()
        
        query = '''
            SELECT * FROM sessions 
            WHERE user_id = ? 
            ORDER BY start_time DESC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query, (user_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_session_timeseries(self, session_id):
        """Get time series data for a session"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM movement_timeseries
            WHERE session_id = ?
            ORDER BY timestamp_ms
        ''', (session_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_user_progress_over_time(self, user_id, metric='avg_smoothness'):
        """Get progression of a metric over time"""
        cursor = self.conn.cursor()
        cursor.execute(f'''
            SELECT start_time, level_name, {metric}
            FROM sessions
            WHERE user_id = ? AND {metric} IS NOT NULL
            ORDER BY start_time
        ''', (user_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_sessions_for_training(self, min_frames=100, completed_only=True):
        """Get all sessions suitable for ML training"""
        cursor = self.conn.cursor()
        
        query = '''
            SELECT s.*, u.rehab_stage, u.diagnosis
            FROM sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.total_frames >= ?
        '''
        
        if completed_only:
            query += ' AND s.completed = 1'
        
        query += ' ORDER BY s.start_time'
        
        cursor.execute(query, (min_frames,))
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def save_baseline(self, movement_name, category, reference_data, description=None):
        """Save a baseline/reference movement pattern"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO baseline_movements (movement_name, category, reference_data, description)
            VALUES (?, ?, ?, ?)
        ''', (movement_name, category, json.dumps(reference_data), description))
        self.conn.commit()
        return cursor.lastrowid
    
    def record_milestone(self, user_id, milestone_type, metric_name=None, 
                        metric_value=None, notes=None):
        """Record a progress milestone"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO progress_milestones (user_id, milestone_type, metric_name, 
                                             metric_value, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, milestone_type, metric_name, metric_value, notes))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_user_statistics(self, user_id):
        """Get comprehensive statistics for a user"""
        cursor = self.conn.cursor()
        
        # Total sessions
        cursor.execute('SELECT COUNT(*) as total FROM sessions WHERE user_id = ?', (user_id,))
        total_sessions = cursor.fetchone()['total']
        
        # Average metrics
        cursor.execute('''
            SELECT 
                AVG(avg_speed) as avg_speed,
                AVG(avg_smoothness) as avg_smoothness,
                AVG(avg_tremor) as avg_tremor,
                SUM(duration_seconds) as total_time,
                SUM(stars_earned) as total_stars
            FROM sessions
            WHERE user_id = ? AND completed = 1
        ''', (user_id,))
        
        stats = dict(cursor.fetchone())
        stats['total_sessions'] = total_sessions
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
