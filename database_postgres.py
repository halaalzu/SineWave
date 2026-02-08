"""
PostgreSQL Database Manager for FlowState
Supports production-grade storage with better performance for time series data
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import json
from datetime import datetime
from typing import List, Dict, Optional

class PostgresRehabDatabase:
    """PostgreSQL database manager for rehabilitation tracking"""
    
    def __init__(self, dbname='flowstate', user='postgres', password='', host='localhost', port=5432):
        """
        Initialize PostgreSQL connection
        
        For local development:
        1. Install PostgreSQL: brew install postgresql
        2. Start service: brew services start postgresql
        3. Create database: createdb flowstate
        """
        self.conn_params = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port
        }
        self.conn = None
        self.setup_database()
    
    def connect(self):
        """Establish database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**self.conn_params)
        return self.conn
    
    def setup_database(self):
        """Create database tables if they don't exist"""
        self.connect()
        cursor = self.conn.cursor()
        
        # Enable TimescaleDB extension (optional but recommended for time series)
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            self.conn.commit()
            print("✓ TimescaleDB extension enabled")
        except:
            print("⚠ TimescaleDB not available, using standard PostgreSQL")
            self.conn.rollback()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                diagnosis TEXT,
                rehab_stage VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL REFERENCES users(user_id),
                level_name VARCHAR(255) NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_seconds REAL,
                total_frames INTEGER,
                stars_earned INTEGER DEFAULT 0,
                completed BOOLEAN DEFAULT FALSE,
                avg_speed REAL,
                max_speed REAL,
                avg_smoothness REAL,
                avg_tremor REAL,
                notes TEXT
            )
        ''')
        
        # Time series data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS movement_timeseries (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL REFERENCES sessions(session_id),
                timestamp_ms BIGINT NOT NULL,
                frame_number INTEGER,
                landmark_positions JSONB,
                joint_angles JSONB,
                hand_openness REAL,
                velocity_mean REAL,
                velocity_max REAL,
                acceleration_mean REAL,
                smoothness_score REAL,
                tremor_score REAL,
                range_of_motion REAL,
                event_type VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Try to create hypertable if TimescaleDB is available
        try:
            cursor.execute('''
                SELECT create_hypertable('movement_timeseries', 'created_at',
                                        if_not_exists => TRUE);
            ''')
            print("✓ Hypertable created for time series data")
        except:
            pass
        
        # Create indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timeseries_session 
            ON movement_timeseries(session_id, timestamp_ms)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timeseries_created 
            ON movement_timeseries(created_at DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sessions_user 
            ON sessions(user_id, start_time DESC)
        ''')
        
        # Session events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_events (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL REFERENCES sessions(session_id),
                timestamp TIMESTAMP NOT NULL,
                event_type VARCHAR(100) NOT NULL,
                event_data JSONB
            )
        ''')
        
        # Baseline movements
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS baseline_movements (
                id SERIAL PRIMARY KEY,
                movement_name VARCHAR(255) NOT NULL,
                category VARCHAR(100),
                reference_data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        ''')
        
        # Progress milestones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS progress_milestones (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL REFERENCES users(user_id),
                milestone_type VARCHAR(100) NOT NULL,
                achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name VARCHAR(100),
                metric_value REAL,
                notes TEXT
            )
        ''')
        
        # ML model metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_models (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                model_type VARCHAR(100) NOT NULL,
                version VARCHAR(50),
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                training_data_sessions TEXT[],
                metrics JSONB,
                model_path TEXT,
                description TEXT
            )
        ''')
        
        self.conn.commit()
        print("✓ PostgreSQL database setup complete")
    
    def create_user(self, user_id, name, diagnosis=None, rehab_stage=None, notes=None):
        """Create a new user"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO users (user_id, name, diagnosis, rehab_stage, notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO NOTHING
        ''', (user_id, name, diagnosis, rehab_stage, notes))
        self.conn.commit()
        return user_id
    
    def create_session(self, session_id, user_id, level_name, start_time):
        """Create a new session"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (session_id, user_id, level_name, start_time)
            VALUES (%s, %s, %s, %s)
        ''', (session_id, user_id, level_name, start_time))
        self.conn.commit()
        return session_id
    
    def update_session(self, session_id, **kwargs):
        """Update session with completion data"""
        if not kwargs:
            return
        
        cursor = self.conn.cursor()
        fields = ', '.join([f"{key} = %s" for key in kwargs.keys()])
        values = list(kwargs.values()) + [session_id]
        
        query = f"UPDATE sessions SET {fields} WHERE session_id = %s"
        cursor.execute(query, values)
        self.conn.commit()
    
    def save_frame_data(self, session_id, frame_data):
        """Batch save time series frame data (much faster)"""
        cursor = self.conn.cursor()
        
        # Prepare batch data
        batch_data = []
        for i, frame in enumerate(frame_data):
            features = frame['features']
            batch_data.append((
                session_id,
                frame['timestamp_ms'],
                i,
                json.dumps(features.get('landmark_positions', [])),
                json.dumps(features.get('joint_angles', {})),
                features.get('hand_openness'),
                features.get('velocity', {}).get('mean_speed'),
                features.get('velocity', {}).get('max_speed'),
                features.get('acceleration', {}).get('mean_acceleration'),
                features.get('smoothness'),
                features.get('tremor_score'),
                features.get('range_of_motion', {}).get('mean_rom'),
                frame.get('event_type')
            ))
        
        # Batch insert
        execute_batch(cursor, '''
            INSERT INTO movement_timeseries (
                session_id, timestamp_ms, frame_number,
                landmark_positions, joint_angles, hand_openness,
                velocity_mean, velocity_max, acceleration_mean,
                smoothness_score, tremor_score, range_of_motion, event_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', batch_data, page_size=1000)
        
        self.conn.commit()
    
    def get_user_sessions(self, user_id, limit=None):
        """Get all sessions for a user"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        query = '''
            SELECT * FROM sessions 
            WHERE user_id = %s 
            ORDER BY start_time DESC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query, (user_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_session_timeseries(self, session_id):
        """Get time series data for a session"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('''
            SELECT * FROM movement_timeseries
            WHERE session_id = %s
            ORDER BY timestamp_ms
        ''', (session_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_sessions_for_training(self, min_frames=100, completed_only=True):
        """Get all sessions suitable for ML training"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        query = '''
            SELECT s.*, u.rehab_stage, u.diagnosis
            FROM sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.total_frames >= %s
        '''
        
        if completed_only:
            query += ' AND s.completed = TRUE'
        
        query += ' ORDER BY s.start_time'
        
        cursor.execute(query, (min_frames,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_session_features_for_ml(self, session_ids: List[str]):
        """
        Get aggregated features for ML training
        Returns flattened feature vectors for each session
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # Aggregate statistics per session
        cursor.execute('''
            SELECT 
                session_id,
                AVG(velocity_mean) as avg_velocity,
                STDDEV(velocity_mean) as std_velocity,
                AVG(acceleration_mean) as avg_acceleration,
                STDDEV(acceleration_mean) as std_acceleration,
                AVG(smoothness_score) as avg_smoothness,
                STDDEV(smoothness_score) as std_smoothness,
                AVG(tremor_score) as avg_tremor,
                AVG(range_of_motion) as avg_rom,
                COUNT(*) as frame_count
            FROM movement_timeseries
            WHERE session_id = ANY(%s)
            GROUP BY session_id
        ''', (session_ids,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def save_ml_model_metadata(self, model_name, model_type, version, 
                               training_sessions, metrics, model_path, description=None):
        """Save ML model metadata"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO ml_models (
                model_name, model_type, version, 
                training_data_sessions, metrics, model_path, description
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (model_name, model_type, version, training_sessions, 
              json.dumps(metrics), model_path, description))
        
        model_id = cursor.fetchone()[0]
        self.conn.commit()
        return model_id
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def migrate_sqlite_to_postgres(sqlite_db_path='flowstate.db'):
    """
    Migrate data from SQLite to PostgreSQL
    
    Usage:
        migrate_sqlite_to_postgres('flowstate.db')
    """
    import sqlite3
    
    print("Starting migration from SQLite to PostgreSQL...")
    
    # Connect to both databases
    sqlite_conn = sqlite3.connect(sqlite_db_path)
    sqlite_conn.row_factory = sqlite3.Row
    
    pg_db = PostgresRehabDatabase()
    
    # Migrate users
    print("Migrating users...")
    cursor = sqlite_conn.cursor()
    cursor.execute('SELECT * FROM users')
    for row in cursor.fetchall():
        pg_db.create_user(
            row['user_id'], row['name'], 
            row['diagnosis'], row['rehab_stage'], row['notes']
        )
    
    # Migrate sessions
    print("Migrating sessions...")
    cursor.execute('SELECT * FROM sessions')
    sessions = cursor.fetchall()
    
    for row in sessions:
        pg_cursor = pg_db.conn.cursor()
        pg_cursor.execute('''
            INSERT INTO sessions (
                session_id, user_id, level_name, start_time, end_time,
                duration_seconds, total_frames, stars_earned, completed,
                avg_speed, max_speed, avg_smoothness, avg_tremor, notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            row['session_id'], row['user_id'], row['level_name'],
            row['start_time'], row['end_time'], row['duration_seconds'],
            row['total_frames'], row['stars_earned'], bool(row['completed']),
            row['avg_speed'], row['max_speed'], row['avg_smoothness'],
            row['avg_tremor'], row['notes']
        ))
    
    pg_db.conn.commit()
    
    # Migrate time series data (in batches)
    print("Migrating time series data...")
    cursor.execute('SELECT * FROM movement_timeseries ORDER BY id')
    
    batch = []
    for i, row in enumerate(cursor.fetchall()):
        batch.append((
            row['session_id'], row['timestamp_ms'], row['frame_number'],
            row['landmark_positions'], row['joint_angles'], row['hand_openness'],
            row['velocity_mean'], row['velocity_max'], row['acceleration_mean'],
            row['smoothness_score'], row['tremor_score'], row['range_of_motion'],
            row['event_type']
        ))
        
        if len(batch) >= 1000:
            pg_cursor = pg_db.conn.cursor()
            execute_batch(pg_cursor, '''
                INSERT INTO movement_timeseries (
                    session_id, timestamp_ms, frame_number,
                    landmark_positions, joint_angles, hand_openness,
                    velocity_mean, velocity_max, acceleration_mean,
                    smoothness_score, tremor_score, range_of_motion, event_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', batch)
            pg_db.conn.commit()
            batch = []
            print(f"  Migrated {i+1} rows...")
    
    # Final batch
    if batch:
        pg_cursor = pg_db.conn.cursor()
        execute_batch(pg_cursor, '''
            INSERT INTO movement_timeseries (
                session_id, timestamp_ms, frame_number,
                landmark_positions, joint_angles, hand_openness,
                velocity_mean, velocity_max, acceleration_mean,
                smoothness_score, tremor_score, range_of_motion, event_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', batch)
        pg_db.conn.commit()
    
    sqlite_conn.close()
    pg_db.close()
    
    print("✓ Migration complete!")


if __name__ == '__main__':
    # Test connection
    print("Testing PostgreSQL connection...")
    db = PostgresRehabDatabase()
    
    # Create test user
    db.create_user('test_user', 'Test User', rehab_stage='early')
    
    print("✓ PostgreSQL database ready!")
    db.close()
