# Deploying ShantiView to Render

## Prerequisites
- GitHub account with your ShantiView repository pushed
- Render account (create at https://render.com)
- Google Gemini API key

## Step-by-Step Deployment

### Step 1: Prepare Your Repository
1. Ensure all changes are committed and pushed to GitHub
2. Verify these files exist in your repository root:
   - `Procfile` (specifies how to run the app)
   - `requirements.txt` (Python dependencies)
   - `.gitignore` (to avoid uploading unnecessary files)

### Step 2: Create a Render Web Service
1. Log in to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** button → Select **"Web Service"**
3. Connect your GitHub repository:
   - Click "Connect account" if prompted
   - Select "ShantiView" repository
   - Choose branch: `main`

### Step 3: Configure Your Web Service

**General Settings:**
- **Name**: `shantiview` (or your preferred name)
- **Environment**: `Python 3`
- **Region**: Choose closest to your users
- **Branch**: `main`

**Build & Deploy:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`

### Step 4: Set Environment Variables

On Render dashboard, go to **"Environment"** tab and add these variables:

```
GOOGLE_API_KEY=your_gemini_api_key_here
FLASK_SECRET_KEY=your_random_secret_key_here
FLASK_ENV=production
```

**To generate `FLASK_SECRET_KEY`**, run in terminal:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Step 5: Configure Instance & Pricing
- **Instance Type**: Choose based on your needs:
  - Free tier (limited, will sleep after 15 mins)
  - Paid tier: $7/month for basic
- Since your app uses heavy ML models (TensorFlow, DeepFace), recommend **at least** the paid plan

### Step 6: Deploy
1. Click **"Create Web Service"**
2. Render will start building your app
3. Check build logs in the **"Logs"** tab
4. Once deployed, you'll get a URL like: `https://shantiview.onrender.com`

## Important Considerations for ShantiView

### 1. **Heavy Dependencies**
Your app has large ML libraries (TensorFlow, DeepFace, torch). Initial build may take 10-15 minutes.

### 2. **Memory Constraints**
- Render free tier has limited memory
- Recommended: Use at least the $7/month plan
- Consider disabling features during development if builds fail

### 3. **Model Loading**
Your app loads ML models on startup:
```python
facial_emotion.py
voice_emotion_cnn.ipynb
voice_emotion_mlp.ipynb
```
Ensure these are properly initialized in your `app.py`.

### 4. **File Uploads**
The `uploads/` folder won't persist between deployments on free tier. Use Render Disks for persistent storage:

1. Go to Web Service settings
2. Click **"Disks"** tab
3. Add disk:
   - **Mount Path**: `/var/uploads` (or your path)
   - **Size**: 1GB minimum
4. Update your code to use this path

### 5. **Audio Processing**
If you're using `pydub` with FFmpeg, Render should have ffmpeg available in the build environment. If you get errors:
- Add a `build.sh` script or install via buildpacks

### 6. **Webcam/Microphone Limitations**
- Cloud deployment cannot access user's local camera or microphone directly
- These features need browser-based capture sent to server
- Ensure your frontend properly sends video/audio streams

## Troubleshooting

### Build Fails
- Check Render logs for specific error
- Common issues:
  - TensorFlow/torch installation failing → check memory limits
  - Missing environment variables → verify all are set in Render dashboard

### App Times Out
- ML model loading takes time
- Add timeouts in your Flask routes:
  ```python
  @app.route('/predict', timeout=120)
  ```

### Port Issues
- Render uses dynamic ports, ensure your Flask app listens to:
  ```python
  app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
  ```

## Verify Deployment

Once deployed:
1. Visit your Render URL
2. Test the welcome page
3. Check logs for any errors: Dashboard → "Logs" tab
4. Test each feature (facial, voice, questionnaire)

## Optimize for Production

### Before Final Deployment:
1. **Update `app.py` for production**:
   ```python
   if __name__ == '__main__':
       # Disable debug mode in production
       app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
   ```

2. **Check memory usage** of ML models
3. **Consider lazy loading** heavy models only when needed
4. **Enable compression** for responses

## Next Steps
- Monitor your Render dashboard
- Set up error alerting if needed
- Understand Render's free tier limitations
- Consider upgrading plan as traffic grows
