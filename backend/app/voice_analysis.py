"""
Voice Emotion Analysis API for ShantiView
Uses the pre-trained MLP model with MFCC features from the RAVDESS dataset
"""

import os
import logging
import numpy as np
import librosa
import joblib
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Constants for MLP
N_MFCC = 40

# Paths to models
# __file__ is /app/app/voice_analysis.py in Docker, or C:\...\backend\app\voice_analysis.py locally
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Check if we're in Docker (/app) or local
if BACKEND_DIR == "/app":
    MODEL_DIR = os.path.join(BACKEND_DIR, "models")
else:
    # Local: models are in parent directory of backend
    MODEL_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "models")

# MLP model paths
MLP_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_emotion_model.joblib")
MLP_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Model and scaler cache
_model = None
_scaler = None


def _extract_mfcc_mean(file_path, n_mfcc=N_MFCC):
    """Extract mean MFCC features for MLP model."""
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        logger.error(f"Error extracting MFCC features from {file_path}: {e}")
        return None


def load_model():
    """Load the MLP emotion model and scaler."""
    global _model, _scaler
    
    if _model is not None:
        return _model, _scaler
    
    if not os.path.exists(MLP_MODEL_PATH):
        logger.error(f"MLP model file not found at {MLP_MODEL_PATH}")
        return None, None
    if not os.path.exists(MLP_SCALER_PATH):
        logger.error(f"Scaler file not found at {MLP_SCALER_PATH}")
        return None, None
    
    try:
        _model = joblib.load(MLP_MODEL_PATH)
        _scaler = joblib.load(MLP_SCALER_PATH)
        logger.info("MLP voice emotion model loaded successfully")
        return _model, _scaler
    except Exception as e:
        logger.error(f"Error loading MLP model: {e}")
        return None, None


async def predict_voice_emotion(audio_file_path: str) -> dict:
    """
    Predict the emotion of an audio file using the trained MLP model.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Load model
        model, scaler = load_model()
        
        if model is None:
            return {
                "error": True,
                "emotion": "Model not available",
                "message": "Voice emotion model is not loaded. Please ensure model files exist."
            }
        
        # Extract features
        features = _extract_mfcc_mean(audio_file_path)
        
        if features is None:
            return {
                "error": True,
                "emotion": "Feature extraction failed",
                "message": "Could not extract features from audio file"
            }
        
        # Scale features and predict
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0]
        
        # Get probabilities if available
        try:
            probabilities = model.predict_proba(features_scaled)[0]
            emotion_probs = {str(label): float(prob) for label, prob in zip(model.classes_, probabilities)}
            confidence = float(max(probabilities))
        except Exception:
            emotion_probs = {}
            confidence = 1.0
        
        emotion_display = str(prediction).capitalize()
        logger.info(f"Voice emotion prediction: {emotion_display} (confidence: {confidence:.3f})")
        
        return {
            "error": False,
            "emotion": emotion_display,
            "confidence": confidence,
            "all_emotions": emotion_probs,
            "model_type": "mlp"
        }
        
    except Exception as e:
        logger.error(f"Error in voice emotion prediction: {e}")
        return {
            "error": True,
            "emotion": "Error",
            "message": str(e)
        }