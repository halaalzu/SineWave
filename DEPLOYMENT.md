# FlowState Deployment Guide

## Current Status
✅ Session analytics now completely session-based (no overall stats)
✅ Server stability improved (debug logging removed)
✅ Hand joint analysis with quality scoring implemented
✅ Interactive session selection interface

## Hosting Options

### Option 1: Cloud Hosting (Recommended)
Deploy the full Flask backend to a cloud service:

#### Railway.app (Free Tier)
1. Create account at [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Add environment variables:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   ```
4. Railway will auto-detect Flask and deploy

#### Render.com (Free Tier)
1. Create account at [render.com](https://render.com)
2. Create new Web Service from GitHub
3. Build command: `pip install -r requirements.txt`
4. Start command: `python app_with_data.py`
5. Add environment variables

#### PythonAnywhere (Free Tier)
1. Upload files to PythonAnywhere
2. Install requirements: `pip install --user -r requirements.txt`
3. Configure web app to point to `app_with_data.py`

#### AWS (Enterprise Option)
**EC2 + RDS Setup:**
1. Launch EC2 instance (t3.medium recommended for video processing)
2. Set up RDS for PostgreSQL (optional upgrade from SQLite)
3. Configure security groups (port 5001, SSH)
4. Install dependencies and deploy

**Elastic Beanstalk (Easier AWS):**
1. Create Elastic Beanstalk application
2. Upload code as ZIP
3. Configure environment variables
4. Auto-scaling and load balancing included

**Cost**: ~$20-50/month depending on usage

### Option 2: Local Hosting Improvements
Fix server crashes for local development:

#### Server Stability Issues
The crashes were caused by:
1. ❌ Debug logging in production endpoints (FIXED)
2. ❌ Missing error handling in analytics endpoints
3. ❌ Memory leaks in video processing
4. ❌ Database connection issues

#### Improvements Made:
- ✅ Removed all debug print statements
- ✅ Added proper error handling
- ✅ Session-based analytics (no aggregation load)

#### Additional Stability Fixes:
```python
# Add to app_with_data.py for better error handling
import logging
logging.basicConfig(level=logging.ERROR)  # Only log errors

# Add memory management for video
import gc
gc.collect()  # Add after video processing
```

### Option 3: Static Hosting (Limited)
For GitHub Pages, you'd need to:

1. **Remove Python Backend**: Convert to JavaScript-only
2. **Use Local Storage**: Instead of SQLite database
3. **Remove Real-time Features**: No video processing
4. **Client-side Analytics**: Basic scoring only

This would be a very limited version. **Cloud hosting is recommended.**

## Quick Deployment Steps

### For Railway (Easiest):
1. Push code to GitHub repository
2. Connect Railway to your GitHub
3. Add `GOOGLE_API_KEY` environment variable
4. Deploy automatically

### For Local Development:
```bash
# Install dependencies
pip install -r requirements.txt

# Run with stability improvements
python app_with_data.py
```

## Environment Variables Needed
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Files for Deployment
- `app_with_data.py` - Main application
- `requirements.txt` - Dependencies  
- `templates/` - HTML templates
- `static/` - CSS/JS assets
- `flowstate_data.db` - SQLite database (will be created)

## Architecture Notes
- Flask backend handles video processing, analytics, and database
- SQLite database stores session data and hand tracking
- Frontend is HTML/CSS/JS with real-time updates
- Gemini AI provides exercise feedback

## Troubleshooting
- **Server crashes**: Check that debug logging is removed
- **No data**: Ensure sessions are being recorded properly
- **Slow analytics**: Use session-based instead of overall analysis
- **Database errors**: Check SQLite file permissions

## Next Steps
1. Choose hosting option (Railway recommended)
2. Set up environment variables
3. Deploy and test session analytics
4. Monitor for stability issues

The session analytics are now completely session-focused as requested!