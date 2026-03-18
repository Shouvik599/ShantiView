import cv2
from deepface import DeepFace
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set a threshold for emotion confidence. Emotions detected with a score below this
# will be considered "Uncertain". 50.0 is a good starting point.
CONFIDENCE_THRESHOLD = 50.0

def analyze_frame(frame):
    """
    Analyzes a single video frame to detect faces and their emotions using DeepFace.
    Only returns an emotion if its confidence score is above the threshold.

    Args:
        frame: A numpy array representing the video frame.

    Returns:
        A tuple containing:
        - dominant_emotion (str): The name of the emotion, "Uncertain", or None.
        - emotion_score (float): The confidence score of the dominant emotion.
        - The original frame with the bounding box and emotion text drawn on it.
    """
    try:
        # Using enforce_detection=False to avoid exceptions when no face is found.
        # DeepFace.analyze returns a list of dictionaries, one for each detected face.
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)

        # DeepFace returns a list, one dict per face. We'll process the first one.
        if results and isinstance(results, list) and len(results) > 0:
            result = results[0]
            dominant_emotion = result['dominant_emotion']
            emotion_score = result['emotion'][dominant_emotion]
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Determine the final emotion based on the confidence threshold
            display_emotion = dominant_emotion
            if emotion_score < CONFIDENCE_THRESHOLD:
                display_emotion = "Uncertain"
                color = (255, 255, 0)  # Yellow for uncertain
            else:
                color = (0, 255, 0)  # Green for confident

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Prepare text with emotion and confidence
            emotion_text = f"{display_emotion.capitalize()} ({emotion_score:.1f}%)"

            # Put the emotion text above the rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, emotion_text, (x, y - 10), font, 0.7, color, 2, cv2.LINE_AA)

            # Return the emotion if confident, otherwise "Uncertain"
            if emotion_score >= CONFIDENCE_THRESHOLD:
                return dominant_emotion, emotion_score, frame
            else:
                return "Uncertain", emotion_score, frame
    except Exception as e:
        # Log other potential errors but continue the loop
        logging.error("An error occurred during analysis: %s", e)
    
    # If no face is detected or an error occurs, return None and the original frame
    return None, None, frame
