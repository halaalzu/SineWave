"""
Dynamic Time Warping (DTW) Analysis for Movement Comparison
Compare patient movements against baseline/ideal patterns
"""

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from typing import List, Dict, Tuple
import json

try:
    from database_postgres import PostgresRehabDatabase
    USE_POSTGRES = True
except ImportError:
    from database import RehabDatabase
    USE_POSTGRES = False


def dtw_distance(series1: np.ndarray, series2: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calculate Dynamic Time Warping distance between two time series
    
    DTW allows sequences to be compared even if they are:
    - Different lengths
    - Different speeds
    - Slightly out of phase
    
    Args:
        series1: First time series (n_samples, n_features)
        series2: Second time series (m_samples, n_features)
    
    Returns:
        distance: DTW distance
        path: Optimal alignment path
    """
    n, m = len(series1), len(series2)
    
    # Initialize DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(series1[i-1], series2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    distance = dtw_matrix[n, m]
    
    # Backtrack to find optimal path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        
        # Choose minimum of three neighbors
        candidates = [
            (i-1, j-1, dtw_matrix[i-1, j-1]),
            (i-1, j, dtw_matrix[i-1, j]),
            (i, j-1, dtw_matrix[i, j-1])
        ]
        i, j, _ = min(candidates, key=lambda x: x[2])
    
    path.reverse()
    
    return distance, np.array(path)


def normalize_time_series(series: np.ndarray) -> np.ndarray:
    """
    Normalize time series to zero mean and unit variance
    Makes DTW comparisons more robust
    """
    mean = np.mean(series, axis=0)
    std = np.std(series, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    return (series - mean) / std


def smooth_time_series(series: np.ndarray, window_length=5, polyorder=2) -> np.ndarray:
    """
    Smooth time series using Savitzky-Golay filter
    Reduces noise while preserving shape
    """
    if len(series) < window_length:
        return series
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Apply filter to each feature dimension
    smoothed = np.zeros_like(series)
    for i in range(series.shape[1]):
        smoothed[:, i] = savgol_filter(series[:, i], window_length, polyorder)
    
    return smoothed


class MovementAnalyzer:
    """Analyze patient movements using DTW comparison"""
    
    def __init__(self, db_path='flowstate.db'):
        """Initialize with database connection"""
        if USE_POSTGRES:
            self.db = PostgresRehabDatabase()
        else:
            self.db = RehabDatabase(db_path)
    
    def extract_landmark_trajectory(self, session_data: List[Dict], 
                                   landmark_ids: List[int] = None) -> np.ndarray:
        """
        Extract 3D trajectory of specific landmarks over time
        
        Args:
            session_data: List of frames from database
            landmark_ids: Which landmarks to extract (default: all 21)
        
        Returns:
            trajectory: (n_frames, n_landmarks * 3) array
        """
        if landmark_ids is None:
            # Use key landmarks: wrist, index finger tip, thumb tip
            landmark_ids = [0, 4, 8]  # Wrist, thumb tip, index tip
        
        trajectories = []
        
        for frame in session_data:
            positions = json.loads(frame['landmark_positions']) if isinstance(
                frame['landmark_positions'], str) else frame['landmark_positions']
            
            if not positions:
                continue
            
            # Extract selected landmarks
            frame_coords = []
            for lid in landmark_ids:
                if lid < len(positions):
                    landmark = positions[lid]
                    frame_coords.extend([landmark['x'], landmark['y'], landmark['z']])
                else:
                    frame_coords.extend([0, 0, 0])
            
            trajectories.append(frame_coords)
        
        return np.array(trajectories)
    
    def compare_to_baseline(self, session_id: str, baseline_session_id: str,
                           landmark_ids: List[int] = None) -> Dict:
        """
        Compare a session to a baseline using DTW
        
        Returns similarity score and detailed analysis
        """
        # Load both sessions
        patient_data = self.db.get_session_timeseries(session_id)
        baseline_data = self.db.get_session_timeseries(baseline_session_id)
        
        if not patient_data or not baseline_data:
            return {'error': 'Session data not found'}
        
        # Extract trajectories
        patient_traj = self.extract_landmark_trajectory(patient_data, landmark_ids)
        baseline_traj = self.extract_landmark_trajectory(baseline_data, landmark_ids)
        
        if len(patient_traj) == 0 or len(baseline_traj) == 0:
            return {'error': 'No trajectory data found'}
        
        # Smooth trajectories
        patient_traj = smooth_time_series(patient_traj)
        baseline_traj = smooth_time_series(baseline_traj)
        
        # Normalize
        patient_traj = normalize_time_series(patient_traj)
        baseline_traj = normalize_time_series(baseline_traj)
        
        # Calculate DTW distance
        dtw_dist, alignment_path = dtw_distance(patient_traj, baseline_traj)
        
        # Calculate similarity score (0-100, higher is better)
        # Normalize by average trajectory length
        avg_length = (len(patient_traj) + len(baseline_traj)) / 2
        normalized_dist = dtw_dist / (avg_length * patient_traj.shape[1])
        
        # Convert to similarity percentage (exponential decay)
        similarity_score = 100 * np.exp(-normalized_dist)
        
        # Calculate per-landmark similarity
        n_landmarks = len(landmark_ids) if landmark_ids else 3
        landmark_similarities = []
        
        for i in range(n_landmarks):
            # Extract single landmark trajectory (3 coords: x, y, z)
            start_idx = i * 3
            end_idx = start_idx + 3
            
            patient_landmark = patient_traj[:, start_idx:end_idx]
            baseline_landmark = baseline_traj[:, start_idx:end_idx]
            
            landmark_dist, _ = dtw_distance(patient_landmark, baseline_landmark)
            landmark_sim = 100 * np.exp(-landmark_dist / len(patient_landmark))
            
            landmark_similarities.append({
                'landmark_id': landmark_ids[i] if landmark_ids else i,
                'similarity': float(landmark_sim),
                'dtw_distance': float(landmark_dist)
            })
        
        return {
            'overall_similarity': float(similarity_score),
            'dtw_distance': float(dtw_dist),
            'normalized_distance': float(normalized_dist),
            'patient_frames': len(patient_data),
            'baseline_frames': len(baseline_data),
            'alignment_quality': float(len(alignment_path) / max(len(patient_traj), len(baseline_traj))),
            'landmark_similarities': landmark_similarities
        }
    
    def compare_to_multiple_baselines(self, session_id: str, 
                                      baseline_ids: List[str]) -> Dict:
        """
        Compare session to multiple baselines and find best match
        """
        results = []
        
        for baseline_id in baseline_ids:
            comparison = self.compare_to_baseline(session_id, baseline_id)
            if 'error' not in comparison:
                comparison['baseline_id'] = baseline_id
                results.append(comparison)
        
        if not results:
            return {'error': 'No valid comparisons'}
        
        # Find best match
        best_match = max(results, key=lambda x: x['overall_similarity'])
        
        return {
            'best_match': best_match,
            'all_comparisons': results,
            'average_similarity': float(np.mean([r['overall_similarity'] for r in results]))
        }
    
    def analyze_movement_quality(self, session_id: str) -> Dict:
        """
        Analyze movement quality without baseline comparison
        Uses statistical measures
        """
        session_data = self.db.get_session_timeseries(session_id)
        
        if not session_data:
            return {'error': 'Session not found'}
        
        # Extract various metrics
        velocities = [d['velocity_mean'] for d in session_data if d.get('velocity_mean')]
        smoothness = [d['smoothness_score'] for d in session_data if d.get('smoothness_score')]
        tremor = [d['tremor_score'] for d in session_data if d.get('tremor_score')]
        rom = [d['range_of_motion'] for d in session_data if d.get('range_of_motion')]
        
        analysis = {
            'session_id': session_id,
            'total_frames': len(session_data),
            'duration_seconds': len(session_data) / 30.0,
        }
        
        if velocities:
            analysis['velocity'] = {
                'mean': float(np.mean(velocities)),
                'std': float(np.std(velocities)),
                'max': float(np.max(velocities)),
                'consistency': float(1 - np.std(velocities) / (np.mean(velocities) + 1e-6))
            }
        
        if smoothness:
            analysis['smoothness'] = {
                'mean': float(np.mean(smoothness)),
                'std': float(np.std(smoothness)),
                'quality_score': float(np.mean(smoothness) * 100)
            }
        
        if tremor:
            analysis['tremor'] = {
                'mean': float(np.mean(tremor)),
                'max': float(np.max(tremor)),
                'stability': float(1 - np.mean(tremor))
            }
        
        if rom:
            analysis['range_of_motion'] = {
                'mean': float(np.mean(rom)),
                'max': float(np.max(rom)),
                'utilization': float(np.mean(rom) / (np.max(rom) + 1e-6))
            }
        
        # Overall quality score (0-100)
        quality_components = []
        
        if smoothness:
            quality_components.append(np.mean(smoothness) * 100)
        if tremor:
            quality_components.append((1 - np.mean(tremor)) * 100)
        if velocities:
            quality_components.append(min(np.mean(velocities) / 0.5, 1) * 100)
        
        if quality_components:
            analysis['overall_quality_score'] = float(np.mean(quality_components))
        
        return analysis
    
    def get_improvement_metrics(self, user_id: str, 
                               recent_sessions: int = 5) -> Dict:
        """
        Calculate improvement over time for a user
        """
        if USE_POSTGRES:
            sessions = self.db.get_user_sessions(user_id, limit=recent_sessions)
        else:
            cursor = self.db.conn.cursor()
            cursor.execute('''
                SELECT * FROM sessions
                WHERE user_id = ?
                ORDER BY start_time DESC
                LIMIT ?
            ''', (user_id, recent_sessions))
            
            columns = [desc[0] for desc in cursor.description]
            sessions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        if len(sessions) < 2:
            return {'error': 'Not enough sessions for comparison'}
        
        # Reverse to chronological order
        sessions = list(reversed(sessions))
        
        # Track metrics over time
        metrics_over_time = {
            'session_ids': [s['session_id'] for s in sessions],
            'timestamps': [s['start_time'] for s in sessions],
            'avg_smoothness': [s['avg_smoothness'] for s in sessions],
            'avg_tremor': [s['avg_tremor'] for s in sessions],
            'avg_speed': [s['avg_speed'] for s in sessions],
        }
        
        # Calculate improvement rates
        improvements = {}
        
        for metric in ['avg_smoothness', 'avg_speed']:
            values = metrics_over_time[metric]
            if all(v is not None for v in values):
                first_val = values[0]
                last_val = values[-1]
                change = ((last_val - first_val) / (first_val + 1e-6)) * 100
                improvements[metric] = {
                    'first_value': float(first_val),
                    'last_value': float(last_val),
                    'percent_change': float(change),
                    'trend': 'improving' if change > 5 else 'stable' if change > -5 else 'declining'
                }
        
        # Tremor improvement (lower is better)
        tremor_values = [v for v in metrics_over_time['avg_tremor'] if v is not None]
        if tremor_values:
            improvements['avg_tremor'] = {
                'first_value': float(tremor_values[0]),
                'last_value': float(tremor_values[-1]),
                'percent_change': float(((tremor_values[0] - tremor_values[-1]) / (tremor_values[0] + 1e-6)) * 100),
                'trend': 'improving' if tremor_values[-1] < tremor_values[0] else 'stable'
            }
        
        return {
            'user_id': user_id,
            'sessions_analyzed': len(sessions),
            'time_period': {
                'start': sessions[0]['start_time'],
                'end': sessions[-1]['start_time']
            },
            'improvements': improvements,
            'metrics_over_time': metrics_over_time
        }


def main():
    """
    Example usage of DTW analysis
    """
    print("\n" + "="*60)
    print("FlowState DTW Movement Analysis")
    print("="*60 + "\n")
    
    analyzer = MovementAnalyzer()
    
    # Example: Analyze a session
    print("Example 1: Analyze movement quality")
    print("-" * 60)
    
    # Get a recent session
    if USE_POSTGRES:
        cursor = analyzer.db.conn.cursor()
        cursor.execute("SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1")
        result = cursor.fetchone()
    else:
        cursor = analyzer.db.conn.cursor()
        cursor.execute("SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1")
        result = cursor.fetchone()
    
    if result:
        session_id = result[0]
        analysis = analyzer.analyze_movement_quality(session_id)
        
        print(f"Session: {session_id}")
        print(f"Duration: {analysis.get('duration_seconds', 0):.1f}s")
        print(f"Quality Score: {analysis.get('overall_quality_score', 0):.1f}/100")
        
        if 'velocity' in analysis:
            print(f"Average Speed: {analysis['velocity']['mean']:.3f}")
        if 'smoothness' in analysis:
            print(f"Smoothness: {analysis['smoothness']['quality_score']:.1f}/100")
        if 'tremor' in analysis:
            print(f"Tremor: {analysis['tremor']['mean']:.3f} (lower is better)")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
