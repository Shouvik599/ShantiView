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

# Paths to the model and scaler
# Calculate the project root directory (works for both local dev and Docker)
# __file__ is backend/app/voice_analysis.py, so we go up 2 levels to get backend/
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_emotion_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Emotion labels as per the RAVDESS dataset
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Model and scaler cache
_model = None
_scaler = None


def load_model():
    """Load the MLP emotion model and scaler from disk."""
    global _model, _scaler
    
    if _model is None or _scaler is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return None, None
        if not os.path.exists(SCALER_PATH):
            logger.error(f"Scaler file not found at {SCALER_PATH}")
            return None, None
            
        try:
            _model = joblib.load(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
            logger.info("Voice emotion model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model/scaler: {e}")
            return None, None
            
    return _model, _scaler


def extract_mfcc_features(file_path, n_mfcc=40):
    """
    Extract Mel-frequency cepstral coefficients (MFCCs) from an audio file.
    
    Args:
        file_path: Path to the audio file
        n_mfcc: Number of MFCCs to extract
        
    Returns:
        numpy array of MFCC features, or None if extraction fails
    """
    try:
        # Load the audio file with parameters matching the training
        y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Calculate mean across time axis
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean
    except Exception as e:
        logger.error(f"Error extracting MFCC features from {file_path}: {e}")
        return None


async def predict_voice_emotion(audio_file_path: str) -> dict:
    """
    Predict the emotion of an audio file using the pre-trained MLP model.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Load model and scaler
        model, scaler = load_model()
        
        if model is None or scaler is None:
            return {
                "error": True,
                "emotion": "Model not available",
                "message": "Voice emotion model is not loaded. Please ensure model files exist."
            }
        
        # Extract features
        features = extract_mfcc_features(audio_file_path)
        
        if features is None:
            return {
                "error": True,
                "emotion": "Feature extraction failed",
                "message": "Could not extract features from audio file"
            }
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(features_scaled)[0]
            emotion_probs = {label: float(prob) for label, prob in zip(model.classes_, probabilities)}
            confidence = float(max(probabilities))
        except Exception:
            emotion_probs = {}
            confidence = 1.0
        
        # Format emotion label
        emotion_display = prediction.capitalize()
        
        logger.info(f"Voice emotion prediction: {emotion_display} (confidence: {confidence:.3f})")
        
        return {
            "error": False,
            "emotion": emotion_display,
            "confidence": confidence,
            "all_emotions": emotion_probs
        }
        
    except Exception as e:
        logger.error(f"Error in voice emotion prediction: {e}")
        return {
            "error": True,
            "emotion": "Error",
            "message": str(e)
        }