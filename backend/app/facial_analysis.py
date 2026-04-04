"""
Facial Emotion Analysis API for ShantiView
Wraps the DeepFace-based facial emotion detection model
"""

import cv2
import logging
import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)

# Confidence threshold for emotion detection
CONFIDENCE_THRESHOLD = 30.0


def _convert_to_native_types(data):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(data, dict):
        return {k: _convert_to_native_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_to_native_types(item) for item in data]
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def analyze_frame_api(image_path: str) -> dict:
    """
    Analyze an image file for facial emotion detection.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with detection results (JSON serializable)
    """
    try:
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            return {
                "detected": False,
                "emotion": "No image",
                "score": 0.0,
                "message": "Could not read image file"
            }
        
        # Analyze using DeepFace
        results = DeepFace.analyze(
            frame, 
            actions=['emotion'], 
            enforce_detection=False, 
            silent=True
        )
        
        # Process results
        if results and isinstance(results, list) and len(results) > 0:
            result = results[0]
            dominant_emotion = str(result.get('dominant_emotion', 'unknown'))
            emotion_scores = result.get('emotion', {})
            emotion_score = float(emotion_scores.get(dominant_emotion, 0.0))
            
            # Check confidence threshold
            detected = emotion_score >= CONFIDENCE_THRESHOLD
            emotion = dominant_emotion if detected else "Uncertain"
            
            # Convert all_emotions to native Python types
            all_emotions = {}
            for k, v in emotion_scores.items():
                all_emotions[str(k)] = float(v) / 100.0
            
            return {
                "detected": True,
                "emotion": emotion.capitalize(),
                "score": float(emotion_score) / 100.0,  # Convert to 0-1 scale
                "all_emotions": _convert_to_native_types(all_emotions)
            }
        else:
            return {
                "detected": False,
                "emotion": "No face",
                "score": 0.0,
                "message": "No face detected in image"
            }
            
    except Exception as e:
        logger.error(f"Error in facial analysis: {e}")
        return {
            "detected": False,
            "emotion": "Error",
            "score": 0.0,
            "message": str(e)
        }
