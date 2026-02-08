"""
Session Comparison & Progression Analytics
Compares sessions without needing a complex ML model
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

class SessionComparison:
    """Compare sessions and track progression over time"""
    
    def __init__(self, db_path='flowstate.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
    
    def get_user_sessions(self, user_id: str, days_back: int = 90) -> List[Dict]:
        """Get all sessions for a user within timeframe"""
        cursor = self.conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        cursor.execute('''
            SELECT 
                session_id, start_time, end_time, duration_seconds,
                total_frames, avg_speed, max_speed, avg_smoothness,
                avg_tremor, completed
            FROM sessions
            WHERE user_id = ? 
              AND completed = 1
              AND start_time >= ?
            ORDER BY start_time ASC
        ''', (user_id, cutoff_date.isoformat()))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row['session_id'],
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'duration': row['duration_seconds'],
                'frames': row['total_frames'],
                'avg_speed': row['avg_speed'],
                'max_speed': row['max_speed'],
                'smoothness': row['avg_smoothness'],
                'tremor': row['avg_tremor']
            })
        
        return sessions
    
    def get_bad_baseline(self, user_id: str) -> Optional[Dict]:
        """Get the user's designated bad_baseline session"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                session_id, start_time, end_time, duration_seconds,
                total_frames, avg_speed, max_speed, avg_smoothness,
                avg_tremor, session_type, notes
            FROM sessions
            WHERE user_id = ? 
              AND session_type = 'bad_baseline'
              AND completed = 1
            LIMIT 1
        ''', (user_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            'session_id': row['session_id'],
            'start_time': row['start_time'],
            'duration': row['duration_seconds'],
            'frames': row['total_frames'],
            'avg_speed': row['avg_speed'],
            'max_speed': row['max_speed'],
            'smoothness': row['avg_smoothness'],
            'tremor': row['avg_tremor'],
            'notes': row['notes'],
            'reference_score': 45  # Bad baseline reference score
        }
    
    def compare_to_baseline(self, session_id: str, user_id: str) -> Dict:
        """Compare current session to user's bad baseline (if exists) or first session"""
        sessions = self.get_user_sessions(user_id, days_back=365)
        
        if not sessions:
            return {'error': 'No sessions found'}
        
        # Check for designated bad_baseline first
        baseline = self.get_bad_baseline(user_id)
        baseline_type = 'bad_baseline'
        
        # If no bad_baseline, use first session
        if not baseline and len(sessions) >= 2:
            baseline = sessions[0]
            baseline_type = 'first_session'
        elif not baseline:
            return {'error': 'Need at least 2 sessions or 1 bad_baseline session to compare'}
        
        current = next((s for s in sessions if s['session_id'] == session_id), None)
        
        if not current:
            return {'error': 'Session not found'}
        
        return {
            'baseline_type': baseline_type,
            'baseline_date': baseline['start_time'],
            'baseline_notes': baseline.get('notes', 'Initial baseline session'),
            'current_date': current['start_time'],
            'improvements': {
                'smoothness': self._calc_improvement(baseline['smoothness'], current['smoothness']),
                'tremor': self._calc_improvement(baseline['tremor'], current['tremor'], lower_is_better=True),
                'speed': self._calc_improvement(baseline['avg_speed'], current['avg_speed']),
                'duration': self._calc_improvement(baseline['duration'], current['duration'])
            },
            'raw_values': {
                'baseline': baseline,
                'current': current
            }
        }
    
    def compare_to_best(self, session_id: str, user_id: str) -> Dict:
        """Compare current session to user's best session"""
        sessions = self.get_user_sessions(user_id)
        
        if not sessions:
            return {'error': 'No sessions found'}
        
        # Find best session (highest smoothness score)
        best = max(sessions, key=lambda s: s['smoothness'] or 0)
        current = next((s for s in sessions if s['session_id'] == session_id), None)
        
        if not current:
            return {'error': 'Session not found'}
        
        return {
            'best_date': best['start_time'],
            'current_date': current['start_time'],
            'is_personal_best': current['session_id'] == best['session_id'],
            'comparison': {
                'smoothness': (current['smoothness'] / best['smoothness']) * 100 if best['smoothness'] else 0,
                'tremor': (best['tremor'] / current['tremor']) * 100 if current['tremor'] else 100,
                'speed': (current['avg_speed'] / best['avg_speed']) * 100 if best['avg_speed'] else 0
            },
            'raw_values': {
                'best': best,
                'current': current
            }
        }
    
    def compare_to_previous(self, session_id: str, user_id: str) -> Dict:
        """Compare to immediately previous session"""
        sessions = self.get_user_sessions(user_id)
        
        if len(sessions) < 2:
            return {'error': 'Need at least 2 sessions'}
        
        current_idx = next((i for i, s in enumerate(sessions) if s['session_id'] == session_id), None)
        
        if current_idx is None or current_idx == 0:
            return {'error': 'No previous session'}
        
        previous = sessions[current_idx - 1]
        current = sessions[current_idx]
        
        return {
            'previous_date': previous['start_time'],
            'current_date': current['start_time'],
            'improvements': {
                'smoothness': self._calc_improvement(previous['smoothness'], current['smoothness']),
                'tremor': self._calc_improvement(previous['tremor'], current['tremor'], lower_is_better=True),
                'speed': self._calc_improvement(previous['avg_speed'], current['avg_speed']),
                'frames': self._calc_improvement(previous['frames'], current['frames'])
            }
        }
    
    def get_progression_trends(self, user_id: str, window_size: int = 5) -> Dict:
        """Calculate moving averages and trends"""
        sessions = self.get_user_sessions(user_id)
        
        if len(sessions) < window_size:
            return {'error': f'Need at least {window_size} sessions'}
        
        # Calculate moving averages
        smoothness_trend = self._moving_average([s['smoothness'] for s in sessions], window_size)
        tremor_trend = self._moving_average([s['tremor'] for s in sessions], window_size)
        speed_trend = self._moving_average([s['avg_speed'] for s in sessions], window_size)
        
        # Calculate improvement rate (linear regression slope)
        session_numbers = list(range(len(sessions)))
        
        return {
            'total_sessions': len(sessions),
            'first_session': sessions[0]['start_time'],
            'latest_session': sessions[-1]['start_time'],
            'trends': {
                'smoothness': {
                    'moving_avg': smoothness_trend,
                    'slope': self._calculate_slope(session_numbers, [s['smoothness'] for s in sessions]),
                    'improving': smoothness_trend[-1] > smoothness_trend[0]
                },
                'tremor': {
                    'moving_avg': tremor_trend,
                    'slope': self._calculate_slope(session_numbers, [s['tremor'] for s in sessions]),
                    'improving': tremor_trend[-1] < tremor_trend[0]  # Lower is better
                },
                'speed': {
                    'moving_avg': speed_trend,
                    'slope': self._calculate_slope(session_numbers, [s['avg_speed'] for s in sessions]),
                    'improving': speed_trend[-1] > speed_trend[0]
                }
            },
            'raw_sessions': sessions
        }
    
    def get_weekly_summary(self, user_id: str) -> List[Dict]:
        """Get weekly aggregated metrics"""
        sessions = self.get_user_sessions(user_id)
        
        # Group by week
        weekly_data = {}
        for session in sessions:
            date = datetime.fromisoformat(session['start_time'])
            week_key = f"{date.year}-W{date.isocalendar()[1]}"
            
            if week_key not in weekly_data:
                weekly_data[week_key] = {
                    'week': week_key,
                    'sessions': [],
                    'total_duration': 0,
                    'avg_smoothness': [],
                    'avg_tremor': [],
                    'avg_speed': []
                }
            
            weekly_data[week_key]['sessions'].append(session)
            weekly_data[week_key]['total_duration'] += session['duration'] or 0
            weekly_data[week_key]['avg_smoothness'].append(session['smoothness'] or 0)
            weekly_data[week_key]['avg_tremor'].append(session['tremor'] or 0)
            weekly_data[week_key]['avg_speed'].append(session['avg_speed'] or 0)
        
        # Calculate weekly averages
        summary = []
        for week, data in sorted(weekly_data.items()):
            summary.append({
                'week': week,
                'session_count': len(data['sessions']),
                'total_duration_minutes': data['total_duration'] / 60,
                'avg_smoothness': np.mean(data['avg_smoothness']),
                'avg_tremor': np.mean(data['avg_tremor']),
                'avg_speed': np.mean(data['avg_speed']),
                'consistency_score': self._calculate_consistency(data['avg_smoothness'])
            })
        
        return summary
    
    def compare_to_healthy_baseline(self, session_id: str) -> Dict:
        """Compare to healthy reference patterns (from synthetic data)"""
        cursor = self.conn.cursor()
        
        # Get current session
        cursor.execute('''
            SELECT avg_speed, avg_smoothness, avg_tremor
            FROM sessions WHERE session_id = ?
        ''', (session_id,))
        
        current = cursor.fetchone()
        if not current:
            return {'error': 'Session not found'}
        
        # Define healthy baselines (from your synthetic "good" sessions)
        healthy_baseline = {
            'smoothness': 2000,  # Your synthetic "good" sessions have high smoothness
            'tremor': 0.001,      # Low tremor
            'speed': 0.5          # Moderate controlled speed
        }
        
        return {
            'comparison_to_healthy': {
                'smoothness': (current['avg_smoothness'] / healthy_baseline['smoothness']) * 100 if current['avg_smoothness'] else 0,
                'tremor': (healthy_baseline['tremor'] / current['avg_tremor']) * 100 if current['avg_tremor'] else 100,
                'speed': (current['avg_speed'] / healthy_baseline['speed']) * 100 if current['avg_speed'] else 0
            },
            'recovery_percentage': self._calculate_recovery_percentage(current, healthy_baseline)
        }
    
    def get_session_analytics(self, session_id: str, user_id: str) -> Dict:
        """Comprehensive session analytics"""
        return {
            'baseline_comparison': self.compare_to_baseline(session_id, user_id),
            'best_comparison': self.compare_to_best(session_id, user_id),
            'previous_comparison': self.compare_to_previous(session_id, user_id),
            'healthy_comparison': self.compare_to_healthy_baseline(session_id),
            'progression_trends': self.get_progression_trends(user_id)
        }
    
    # Helper methods
    def _calc_improvement(self, old_val, new_val, lower_is_better=False):
        """Calculate percentage improvement"""
        if old_val is None or new_val is None or old_val == 0:
            return 0
        
        change = ((new_val - old_val) / abs(old_val)) * 100
        return -change if lower_is_better else change
    
    def _moving_average(self, data, window):
        """Calculate moving average"""
        data = [x for x in data if x is not None]
        if len(data) < window:
            return data
        
        averages = []
        for i in range(len(data) - window + 1):
            avg = np.mean(data[i:i + window])
            averages.append(avg)
        return averages
    
    def _calculate_slope(self, x, y):
        """Simple linear regression slope"""
        y = [val for val in y if val is not None]
        x = x[:len(y)]
        
        if len(x) < 2:
            return 0
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        
        return numerator / denominator if denominator != 0 else 0
    
    def _calculate_consistency(self, values):
        """Calculate consistency score (lower variance = higher consistency)"""
        values = [v for v in values if v is not None]
        if len(values) < 2:
            return 100
        
        std = np.std(values)
        mean = np.mean(values)
        
        # Coefficient of variation (lower is more consistent)
        cv = (std / mean) if mean != 0 else 1
        
        # Convert to 0-100 score (lower CV = higher score)
        consistency = max(0, 100 - (cv * 100))
        return consistency
    
    def _calculate_recovery_percentage(self, current, healthy):
        """Overall recovery percentage compared to healthy baseline"""
        scores = []
        
        if current['avg_smoothness'] and healthy['smoothness']:
            scores.append((current['avg_smoothness'] / healthy['smoothness']) * 100)
        
        if current['avg_tremor'] and healthy['tremor']:
            scores.append((healthy['tremor'] / current['avg_tremor']) * 100)
        
        if current['avg_speed'] and healthy['speed']:
            scores.append((current['avg_speed'] / healthy['speed']) * 100)
        
        return np.mean(scores) if scores else 0
