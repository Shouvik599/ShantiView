"""
Voice Emotion Analysis API for ShantiView
Uses the pre-trained CNN model with MFCC sequence features from the RAVDESS dataset
"""

import os
import logging
import numpy as np
import librosa
import joblib
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Model type configuration
# Set to "cnn" or "mlp" - CNN is preferred, MLP is fallback
MODEL_TYPE = "cnn"  # Can be "cnn" or "mlp"

# Constants for CNN
MAX_SEQ_LENGTH = 130
N_MFCC = 40
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Paths to models
# __file__ is /app/app/voice_analysis.py in Docker, or C:\...\backend\app\voice_analysis.py locally
# - os.path.dirname(__file__) = .../backend/app
# - os.path.dirname(os.path.dirname(__file__)) = .../backend (project root)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Check if we're in Docker (/app) or local (C:\...\backend)
if BACKEND_DIR == "/app":
    MODEL_DIR = os.path.join(BACKEND_DIR, "models")
else:
    # Local: models are in parent directory of backend
    MODEL_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "models")

# CNN model paths
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "ravdess_cnn_model.h5")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
CNN_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# MLP model paths (fallback)
MLP_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_emotion_model.joblib")
MLP_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Model and scaler cache
_cnn_model = None
_mlp_model = None
_scaler = None
_label_encoder = None


def _extract_mfcc_features(file_path, max_len=MAX_SEQ_LENGTH, n_mfcc=N_MFCC):
    """Extract MFCC sequence features for CNN model."""
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T
        
        if mfcc.shape[0] < max_len:
            pad_width = max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_len]
        
        return mfcc
    except Exception as e:
        logger.error(f"Error extracting MFCC features from {file_path}: {e}")
        return None


def _extract_mfcc_mean(file_path, n_mfcc=40):
    """Extract mean MFCC features for MLP model."""
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        logger.error(f"Error extracting MFCC features from {file_path}: {e}")
        return None


def load_cnn_model():
    """Load the CNN emotion model, scaler, and label encoder."""
    global _cnn_model, _scaler, _label_encoder
    
    if _cnn_model is not None:
        return _cnn_model, _scaler, _label_encoder
    
    if not os.path.exists(CNN_MODEL_PATH):
        logger.error(f"CNN model file not found at {CNN_MODEL_PATH}")
        return None, None, None
    if not os.path.exists(CNN_SCALER_PATH):
        logger.error(f"Scaler file not found at {CNN_SCALER_PATH}")
        return None, None, None
    if not os.path.exists(LABEL_ENCODER_PATH):
        logger.error(f"Label encoder file not found at {LABEL_ENCODER_PATH}")
        return None, None, None
    
    try:
        import tensorflow as tf
        # Try loading with custom_objects to handle compatibility issues
        _cnn_model = tf.keras.models.load_model(
            CNN_MODEL_PATH,
            custom_objects=None,
            compile=False  # Skip compilation to avoid deserialization issues
        )
        # Recompile the model since we skipped compilation
        _cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        _scaler = joblib.load(CNN_SCALER_PATH)
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info("CNN voice emotion model loaded successfully")
        return _cnn_model, _scaler, _label_encoder
    except Exception as e:
        logger.error(f"Error loading CNN model: {e}")
        return None, None, None


def load_mlp_model():
    """Load the MLP emotion model and scaler (fallback)."""
    global _mlp_model, _scaler
    
    if _mlp_model is not None:
        return _mlp_model, _scaler
    
    if not os.path.exists(MLP_MODEL_PATH):
        logger.error(f"MLP model file not found at {MLP_MODEL_PATH}")
        return None, None
    if not os.path.exists(MLP_SCALER_PATH):
        logger.error(f"Scaler file not found at {MLP_SCALER_PATH}")
        return None, None
    
    try:
        _mlp_model = joblib.load(MLP_MODEL_PATH)
        _scaler = joblib.load(MLP_SCALER_PATH)
        logger.info("MLP voice emotion model loaded successfully")
        return _mlp_model, _scaler
    except Exception as e:
        logger.error(f"Error loading MLP model: {e}")
        return None, None


def load_model():
    """Load the appropriate model based on configuration."""
    if MODEL_TYPE == "cnn":
        model, scaler, le = load_cnn_model()
        if model is not None:
            return model, scaler, le, "cnn"
        logger.warning("CNN model failed to load, falling back to MLP")
    
    # Fallback to MLP
    model, scaler = load_mlp_model()
    if model is not None:
        return model, scaler, None, "mlp"
    
    logger.error("Both CNN and MLP models failed to load")
    return None, None, None, None


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
    Predict the emotion of an audio file using the trained model.
    Supports both CNN and MLP models.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Load model
        result = load_model()
        model, scaler, label_encoder, model_type = result
        
        if model is None:
            return {
                "error": True,
                "emotion": "Model not available",
                "message": "Voice emotion model is not loaded. Please ensure model files exist."
            }
        
        if model_type == "cnn":
            # CNN prediction
            features = _extract_mfcc_features(audio_file_path)
            
            if features is None:
                return {
                    "error": True,
                    "emotion": "Feature extraction failed",
                    "message": "Could not extract features from audio file"
                }
            
            # Scale features (reshape for 2D scaler, then reshape back)
            original_shape = features.shape
            features_scaled = scaler.transform(features.reshape(-1, original_shape[1])).reshape(original_shape)
            
            # Add batch dimension and predict
            features_batch = np.expand_dims(features_scaled, axis=0)
            predictions = model.predict(features_batch, verbose=0)[0]
            
            # Get emotion from predictions
            predicted_class = np.argmax(predictions)
            confidence = float(predictions[predicted_class])
            
            # Use label encoder if available, otherwise use emotion labels
            if label_encoder is not None:
                emotion_display = label_encoder.inverse_transform([predicted_class])[0]
                emotion_labels = label_encoder.classes_
            else:
                emotion_display = EMOTION_LABELS[predicted_class]
                emotion_labels = EMOTION_LABELS
            
            # Build all_emotions dict using the correct label order (convert to native Python types)
            emotion_probs = {str(label): float(predictions[i]) for i, label in enumerate(emotion_labels)}
            
        else:
            # MLP prediction (fallback)
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
                emotion_probs = {label: float(prob) for label, prob in zip(model.classes_, probabilities)}
                confidence = float(max(probabilities))
            except Exception:
                emotion_probs = {}
                confidence = 1.0
            
            emotion_display = prediction.capitalize()
        
        emotion_display = emotion_display.capitalize()
        logger.info(f"Voice emotion prediction ({model_type}): {emotion_display} (confidence: {confidence:.3f})")
        
        return {
            "error": False,
            "emotion": emotion_display,
            "confidence": confidence,
            "all_emotions": emotion_probs,
            "model_type": model_type
        }
        
    except Exception as e:
        logger.error(f"Error in voice emotion prediction: {e}")
        return {
            "error": True,
            "emotion": "Error",
            "message": str(e)
        }
