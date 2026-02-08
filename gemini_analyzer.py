"""
Gemini AI Analysis for Hand Movement Data
Provides intelligent feedback on rehabilitation progress
"""

import os
import json
import google.generativeai as genai
from database import RehabDatabase
from datetime import datetime


class GeminiHandAnalyzer:
    """Uses Gemini AI to analyze hand movement patterns"""
    
    def __init__(self, api_key=None):
        """Initialize Gemini with API key"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it as environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.db = RehabDatabase()
    
    def analyze_session(self, session_id, user_id='default_user'):
        """Analyze a specific session and provide feedback"""
        # Get session data
        session_data = self._get_session_data(session_id)
        
        if not session_data:
            return {"error": "Session not found or incomplete"}
        
        # Get user baseline
        baseline = self._get_user_baseline(user_id)
        
        # Create prompt for Gemini
        prompt = self._create_analysis_prompt(session_data, baseline)
        
        try:
            # Get Gemini analysis
            response = self.model.generate_content(prompt)
            
            return {
                "session_id": session_id,
                "analysis": response.text,
                "timestamp": datetime.now().isoformat(),
                "session_stats": session_data['summary']
            }
        except Exception as e:
            return {"error": f"Gemini API error: {str(e)}"}
    
    def analyze_live_session(self, current_session_data, user_baseline):
        """Analyze ongoing session data in real-time"""
        prompt = f"""You are a hand rehabilitation specialist analyzing a patient's live hand movement data.

**Current Session Performance:**
- Duration: {current_session_data.get('duration_seconds', 0):.1f} seconds
- Frames captured: {current_session_data.get('total_frames', 0)}
- Average speed: {current_session_data.get('avg_speed', 0):.4f}
- Max speed: {current_session_data.get('max_speed', 0):.4f}
- Smoothness: {current_session_data.get('avg_smoothness', 0):.2f}
- Tremor: {current_session_data.get('avg_tremor', 0):.4f}

**Patient's Baseline (Average from past sessions):**
- Typical speed: {user_baseline.get('avg_speed', 0):.4f}
- Typical smoothness: {user_baseline.get('avg_smoothness', 0):.2f}
- Typical tremor: {user_baseline.get('avg_tremor', 0):.4f}

Provide a brief, encouraging analysis (2-3 sentences) focusing on:
1. How this session compares to their baseline
2. Any improvements or areas to focus on
3. One specific actionable tip

Keep it concise, positive, and actionable."""

        try:
            response = self.model.generate_content(prompt)
            return {
                "feedback": response.text,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Gemini API error: {str(e)}"}
    
    def analyze_weak_joints(self, joint_analysis_data):
        """Provide personalized recommendations based on weak joints"""
        weak_joints = joint_analysis_data.get('weakest_joints', [])
        
        if not weak_joints:
            return {"feedback": "No joint weakness data available."}
        
        # Format weak joints for Gemini
        joints_text = "\n".join([
            f"- {joint['name']}: Score {joint['score']}/100 (Priority: {joint['priority']})"
            for joint in weak_joints[:5]
        ])
        
        prompt = f"""You are a hand rehabilitation therapist. A patient's hand tracking shows these weak joints:

{joints_text}

Provide:
1. A brief 2-3 sentence explanation of what these weaknesses mean
2. 3 specific exercises to improve these joints (be specific and actionable)
3. One daily habit they can adopt

Keep it encouraging, practical, and under 200 words."""

        try:
            response = self.model.generate_content(prompt)
            return {
                "recommendations": response.text,
                "analyzed_joints": len(weak_joints),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Gemini API error: {str(e)}"}
    
    def _get_session_data(self, session_id):
        """Fetch session data from database"""
        cursor = self.db.conn.cursor()
        
        # Get session summary
        cursor.execute('''
            SELECT start_time, end_time, duration_seconds, total_frames,
                   avg_speed, max_speed, avg_smoothness, avg_tremor
            FROM sessions 
            WHERE session_id = ?
        ''', (session_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        # Get frame count
        cursor.execute('''
            SELECT COUNT(*) FROM movement_timeseries WHERE session_id = ?
        ''', (session_id,))
        frame_count = cursor.fetchone()[0]
        
        return {
            "session_id": session_id,
            "summary": {
                "start_time": row[0],
                "end_time": row[1],
                "duration_seconds": row[2],
                "total_frames": row[3],
                "actual_frames_stored": frame_count,
                "avg_speed": row[4] or 0,
                "max_speed": row[5] or 0,
                "avg_smoothness": row[6] or 0,
                "avg_tremor": row[7] or 0
            }
        }
    
    def _get_user_baseline(self, user_id):
        """Calculate user's baseline from past sessions"""
        cursor = self.db.conn.cursor()
        
        cursor.execute('''
            SELECT 
                AVG(avg_speed) as avg_speed,
                AVG(max_speed) as max_speed,
                AVG(avg_smoothness) as avg_smoothness,
                AVG(avg_tremor) as avg_tremor,
                COUNT(*) as session_count
            FROM sessions 
            WHERE user_id = ? AND completed = 1 AND total_frames > 0
        ''', (user_id,))
        
        row = cursor.fetchone()
        
        return {
            "avg_speed": row[0] or 0,
            "max_speed": row[1] or 0,
            "avg_smoothness": row[2] or 0,
            "avg_tremor": row[3] or 0,
            "session_count": row[4] or 0
        }
    
    def _create_analysis_prompt(self, session_data, baseline):
        """Create a detailed prompt for Gemini"""
        summary = session_data['summary']
        
        prompt = f"""You are an expert hand rehabilitation therapist analyzing a patient's hand movement data.

**Session Data:**
- Duration: {summary['duration_seconds']:.1f} seconds
- Frames captured: {summary['total_frames']}
- Movement speed: {summary['avg_speed']:.4f} (max: {summary['max_speed']:.4f})
- Smoothness score: {summary['avg_smoothness']:.2f}
- Tremor level: {summary['avg_tremor']:.4f}

**Patient's Baseline (from {baseline['session_count']} previous sessions):**
- Average speed: {baseline['avg_speed']:.4f}
- Average smoothness: {baseline['avg_smoothness']:.2f}
- Average tremor: {baseline['avg_tremor']:.4f}

Analyze this session and provide:
1. **Progress Summary**: How does this session compare to their baseline? (2-3 sentences)
2. **Strengths**: What's improving? (1-2 specific observations)
3. **Areas for Focus**: What needs work? (1-2 specific observations)
4. **Recommendations**: 2-3 actionable exercises or tips

Be encouraging, specific, and professional. Focus on measurable improvements."""

        return prompt
    
    def chat_about_data(self, context):
        """Have a conversation about user's rehabilitation data"""
        prompt = f"""You are a hand rehabilitation specialist assistant. You have access to the patient's recent movement data and joint analysis. Answer their question in a helpful, encouraging, and professional manner.

{context}

Provide a clear, concise answer (2-4 sentences) that:
1. Directly addresses their question
2. References specific data when relevant
3. Offers actionable advice if appropriate
4. Maintains an encouraging, supportive tone

Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I'm having trouble accessing the AI right now. Error: {str(e)}"


def test_gemini_connection():
    """Test if Gemini API is properly configured"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return {
                "status": "error",
                "message": "GEMINI_API_KEY not set. Add it to your environment variables."
            }
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Say 'Hello' if you're working!")
        
        return {
            "status": "success",
            "message": "Gemini API connected successfully!",
            "test_response": response.text
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Gemini API error: {str(e)}"
        }


if __name__ == '__main__':
    # Test connection
    result = test_gemini_connection()
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    if result['status'] == 'success':
        print(f"Response: {result['test_response']}")
