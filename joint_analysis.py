"""
Joint-Specific Analysis for Rehabilitation
Identifies which joints need more work based on recorded movement data
"""

import numpy as np
import json
from database import RehabDatabase
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

# MediaPipe hand landmark indices
JOINT_GROUPS = {
    'Thumb': {
        'CMC': (0, 1),      # Carpometacarpal
        'MCP': (1, 2),      # Metacarpophalangeal
        'IP': (2, 3),       # Interphalangeal
        'TIP': (3, 4)       # Thumb tip
    },
    'Index': {
        'MCP': (0, 5),
        'PIP': (5, 6),      # Proximal interphalangeal
        'DIP': (6, 7),      # Distal interphalangeal
        'TIP': (7, 8)
    },
    'Middle': {
        'MCP': (0, 9),
        'PIP': (9, 10),
        'DIP': (10, 11),
        'TIP': (11, 12)
    },
    'Ring': {
        'MCP': (0, 13),
        'PIP': (13, 14),
        'DIP': (14, 15),
        'TIP': (15, 16)
    },
    'Pinky': {
        'MCP': (0, 17),
        'PIP': (17, 18),
        'DIP': (18, 19),
        'TIP': (19, 20)
    }
}


class JointAnalyzer:
    """Analyzes joint-specific performance and identifies weaknesses"""
    
    def __init__(self, db_path='flowstate.db'):
        self.db = RehabDatabase(db_path)
    
    def analyze_recent_sessions(self, session_ids):
        """Analyze specific recent sessions by ID for dynamic stats"""
        print(f"\n{'='*60}")
        print(f"🔍 Analyzing {len(session_ids)} Recent Sessions")
        print(f"{'='*60}\n")
        
        cursor = self.db.conn.cursor()
        
        # Get session details
        placeholders = ','.join('?' * len(session_ids))
        cursor.execute(f'''
            SELECT session_id, total_frames, start_time 
            FROM sessions 
            WHERE session_id IN ({placeholders}) AND completed = 1 AND total_frames > 0
            ORDER BY start_time DESC
        ''', session_ids)
        
        sessions = cursor.fetchall()
        
        if not sessions:
            print(f"❌ No sessions found")
            return None
        
        print(f"📊 Analyzing sessions with {sum(s[1] for s in sessions):,} total frames\n")
        
        # Collect joint metrics
        joint_metrics = self._compute_joint_metrics(sessions)
        
        # Analyze weaknesses
        analysis = self._identify_weaknesses(joint_metrics)
        
        return analysis
        
    def analyze_user_joints(self, user_id='default_user'):
        """Analyze all joints for a specific user"""
        print(f"\n{'='*60}")
        print(f"🔍 Joint-Specific Analysis for {user_id}")
        print(f"{'='*60}\n")
        
        # Get all completed sessions for user
        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT session_id, total_frames, start_time 
            FROM sessions 
            WHERE user_id = ? AND completed = 1 AND total_frames > 0
            ORDER BY start_time DESC
        ''', (user_id,))
        sessions = cursor.fetchall()
        
        if not sessions:
            print(f"❌ No completed sessions found for {user_id}")
            return None
        
        print(f"📊 Analyzing {len(sessions)} sessions with {sum(s[1] for s in sessions):,} total frames\n")
        
        # Collect joint metrics across all sessions
        joint_metrics = self._compute_joint_metrics(sessions)
        
        # Analyze weaknesses
        analysis = self._identify_weaknesses(joint_metrics)
        
        # Display results
        self._display_analysis(analysis)
        
        return analysis
    
    def _compute_joint_metrics(self, sessions):
        """Compute ROM, velocity, and smoothness for each joint"""
        joint_data = defaultdict(lambda: {
            'rom_values': [],
            'velocity_values': [],
            'angle_changes': []
        })
        
        cursor = self.db.conn.cursor()
        
        for session_id, total_frames, _ in sessions:
            # Get all frames for this session
            cursor.execute('''
                SELECT landmark_positions, joint_angles 
                FROM movement_timeseries 
                WHERE session_id = ?
                ORDER BY frame_number
            ''', (session_id,))
            
            frames = cursor.fetchall()
            
            for frame in frames:
                try:
                    landmarks = json.loads(frame[0])
                    angles = json.loads(frame[1])
                    
                    if not landmarks or len(landmarks) != 21:
                        continue
                    
                    # Analyze each joint
                    for finger, joints in JOINT_GROUPS.items():
                        for joint_name, (idx1, idx2) in joints.items():
                            joint_key = f"{finger}_{joint_name}"
                            
                            # Calculate distance between landmarks (ROM indicator)
                            p1 = np.array(landmarks[idx1])
                            p2 = np.array(landmarks[idx2])
                            distance = np.linalg.norm(p2 - p1)
                            joint_data[joint_key]['rom_values'].append(distance)
                            
                            # Get joint angle if available
                            angle_key = f"{finger.lower()}_{joint_name.lower()}"
                            if angle_key in angles:
                                joint_data[joint_key]['angle_changes'].append(angles[angle_key])
                
                except (json.JSONDecodeError, IndexError, KeyError) as e:
                    continue
        
        # Compute statistics for each joint
        joint_stats = {}
        for joint_key, data in joint_data.items():
            if data['rom_values']:
                rom_array = np.array(data['rom_values'])
                joint_stats[joint_key] = {
                    'rom_mean': float(np.mean(rom_array)),
                    'rom_std': float(np.std(rom_array)),
                    'rom_range': float(np.ptp(rom_array)),
                    'rom_max': float(np.max(rom_array)),
                    'rom_min': float(np.min(rom_array)),
                    'sample_count': len(rom_array)
                }
        
        return joint_stats
    
    def _identify_weaknesses(self, joint_metrics):
        """Identify joints that need more work based on ROM and variability"""
        if not joint_metrics:
            return None
        
        # Calculate average ROM across all joints for baseline
        all_rom_means = [m['rom_mean'] for m in joint_metrics.values()]
        all_rom_ranges = [m['rom_range'] for m in joint_metrics.values()]
        
        avg_rom = np.mean(all_rom_means)
        avg_range = np.mean(all_rom_ranges)
        
        # Score each joint (lower score = needs more work)
        joint_scores = {}
        for joint_key, metrics in joint_metrics.items():
            # Factors: ROM (50%), Range of motion (30%), Sample consistency (20%)
            rom_score = (metrics['rom_mean'] / avg_rom) * 50
            range_score = (metrics['rom_range'] / avg_range) * 30
            consistency_score = (1 - min(metrics['rom_std'] / metrics['rom_mean'], 1)) * 20
            
            total_score = rom_score + range_score + consistency_score
            
            joint_scores[joint_key] = {
                'score': total_score,
                'rom_mean': metrics['rom_mean'],
                'rom_range': metrics['rom_range'],
                'rom_std': metrics['rom_std'],
                'sample_count': metrics['sample_count']
            }
        
        # Sort by score (lowest first = needs most work)
        sorted_joints = sorted(joint_scores.items(), key=lambda x: x[1]['score'])
        
        return {
            'joint_scores': joint_scores,
            'weakest_joints': sorted_joints[:5],
            'strongest_joints': sorted_joints[-5:],
            'baseline_rom': avg_rom,
            'baseline_range': avg_range
        }
    
    def _display_analysis(self, analysis):
        """Display joint analysis results"""
        if not analysis:
            return
        
        print("🎯 JOINTS NEEDING MORE WORK (Top 5):")
        print("-" * 60)
        for i, (joint_key, data) in enumerate(analysis['weakest_joints'], 1):
            finger, joint = joint_key.split('_')
            score = data['score']
            rom = data['rom_mean']
            
            # Determine severity
            if score < 50:
                status = "🔴 High Priority"
            elif score < 70:
                status = "🟡 Moderate"
            else:
                status = "🟢 Minor"
            
            print(f"\n{i}. {status} | {finger} - {joint}")
            print(f"   Score: {score:.1f}/100")
            print(f"   ROM: {rom:.4f} (Range: {data['rom_range']:.4f})")
            print(f"   Consistency: ±{data['rom_std']:.4f}")
            print(f"   Recommendation: Focus on {joint.lower()} flexibility and control")
        
        print("\n" + "="*60)
        print("💪 STRONGEST JOINTS (Top 5):")
        print("-" * 60)
        for i, (joint_key, data) in enumerate(reversed(analysis['strongest_joints']), 1):
            finger, joint = joint_key.split('_')
            print(f"{i}. {finger} - {joint} | Score: {data['score']:.1f}/100")
        
        print("\n" + "="*60)
        print("\n📋 EXERCISE RECOMMENDATIONS:")
        print("-" * 60)
        
        weak_fingers = set()
        for joint_key, _ in analysis['weakest_joints']:
            finger, _ = joint_key.split('_')
            weak_fingers.add(finger)
        
        recommendations = {
            'Thumb': "• Thumb opposition exercises\n• Pinch grip strengthening\n• Thumb circles and stretches",
            'Index': "• Index finger taps\n• Pointing and flexion exercises\n• Resistance band pulls",
            'Middle': "• Middle finger isolation exercises\n• Grip strengthening\n• Flexion-extension drills",
            'Ring': "• Ring finger independence training\n• Grip variations\n• Finger spreads with resistance",
            'Pinky': "• Pinky isolation exercises\n• Grip strengthening (especially weak grip)\n• Finger abduction exercises"
        }
        
        for finger in weak_fingers:
            print(f"\n{finger} Finger:")
            print(recommendations.get(finger, "• General flexibility and strength exercises"))
        
        print("\n" + "="*60)
    
    def generate_report(self, user_id='default_user', output_file=None):
        """Generate a detailed JSON report"""
        analysis = self.analyze_user_joints(user_id)
        
        if not analysis:
            return None
        
        report = {
            'user_id': user_id,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'baseline_rom': analysis['baseline_rom'],
                'baseline_range': analysis['baseline_range']
            },
            'weakest_joints': [
                {'joint': k, **v} for k, v in analysis['weakest_joints']
            ],
            'strongest_joints': [
                {'joint': k, **v} for k, v in analysis['strongest_joints']
            ],
            'all_joints': analysis['joint_scores']
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Report saved to: {output_file}")
        
        return report


def main():
    """Run joint analysis"""
    analyzer = JointAnalyzer()
    
    # Analyze joints
    analysis = analyzer.analyze_user_joints('default_user')
    
    # Generate report
    if analysis:
        report_file = f"joint_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analyzer.generate_report('default_user', report_file)


if __name__ == '__main__':
    main()
