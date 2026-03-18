# Render Deployment Build Troubleshooting

## TensorFlow Build Error: "No matching distribution found"

### Root Cause
TensorFlow wheels may not be available for your Python version on Render's Linux environment (x86_64).

## ✅ Solution Applied

### 1. **Specified Python Version** (`runtime.txt`)
Created `runtime.txt` with:
```
python-3.10.13
```
This ensures Render uses Python 3.10, which has stable TensorFlow wheel support.

### 2. **Pinned Compatible Versions** (requirements.txt)
Updated `requirements.txt` with specific versions that work on Render:
- TensorFlow 2.14.0 (latest stable for Linux)
- Keras 2.14.0 
- NumPy 1.24.3 (compatible with TensorFlow 2.14)
- Python 3.10 runtime

### 3. **Removed Problematic Dependencies**
- ❌ Removed: `torch` (not used in code, adds 2GB+ to build)
- ❌ Removed: `kagglehub` (not imported)
- ❌ Removed: `ipykernel` (notebook tool, not needed for Flask server)
- ✅ Kept: `deepface` (used for facial emotion detection)

## 🚀 Next Steps

1. **Commit changes**:
   ```bash
   git add requirements.txt runtime.txt
   git commit -m "Fix TensorFlow build issues for Render deployment"
   git push origin main
   ```

2. **Rebuild on Render**:
   - Go to Render Dashboard
   - Find your web service
   - Click **"Manual Deploy"** → **"Deploy Latest Commit"**
   - Check logs for successful build

## If Build Still Fails

### Option A: Add Build Logs File
Create `build.sh` in project root:
```bash
#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt
```
Then in Render settings, use:
- **Build Command**: `chmod +x build.sh && ./build.sh`
- **Start Command**: `gunicorn app:app`

### Option B: Use Lighter ML Model Alternative
If TensorFlow still fails, consider:

**Replace Deepface with alternatives**:
```python
# Option 1: Use OpenCV-based emotion (simpler, faster)
# Option 2: Use a smaller pre-trained model

# Modify models/facial_emotion.py to use a lighter model
```

### Option C: Check Instance Memory
Render free tier has limited memory. TensorFlow + DeepFace require ~2GB during installation.
- **Upgrade to paid plan** ($7/month) for more memory
- Or **remove torch** (already done) and other unused packages

## Performance Optimization Tips

### 1. **Reduce Build Time**
- Current approach: Pre-load models at startup (can take 1-2 min)
- Better approach: Lazy-load models on first use
  ```python
  # In app.py
  mlp_model = None
  
  @app.route('/predict_audio', methods=['POST'])
  def predict_audio():
      global mlp_model
      if mlp_model is None:
          mlp_model = joblib.load("./models/mlp_emotion_model.joblib")
      # ... rest of code
  ```

### 2. **Use Render Disks**
For persistent file storage (`uploads/` folder):
1. In Render dashboard → Web Service settings
2. Click "**Disks**"
3. Add disk: Mount path `/var/data`, Size: 1GB
4. Update code:
   ```python
   UPLOADS_DIR = "/var/data/uploads"
   ```

### 3. **Monitor Build Progress**
Watch Render logs:
```
Building Docker image... 
Running build command: pip install -r requirements.txt
Collecting Flask==3.0.0
Collecting tensorflow==2.14.0  ← This step takes longest
...
Build successful ✓
```

## Expected Build Times
- **First build**: 15-20 minutes (downloads & installs TensorFlow)
- **Subsequent builds**: 2-5 minutes (cached dependencies)

## Verify Deployment

Once built successfully:
1. Visit your Render URL
2. You should see ShantiView welcome page
3. Test all features (facial, voice, questionnaire)
4. Check Render logs for any errors: **Logs** tab

## Contact Render Support

If issues persist:
- Go to Render Docs: https://render.com/docs/troubleshooting-deploys
- Check limits: https://render.com/docs/resource-limits
- High memory builds may need custom options

---

**Last Updated**: March 19, 2026
