# Import packages

import os
import json
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, session
from flask_cors import CORS
from dotenv import load_dotenv
import time
import models.facial_emotion as facial_emotion  # Import the new module
import cv2
import requests
import base64
import numpy as np
import librosa
import sys
import tensorflow as tf
import atexit
import joblib
from pydub import AudioSegment
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
    
# Load the NVIDIA API key and configure the OpenAI client
try:
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY not found in environment variables")
    
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY
    )
    logging.info("NVIDIA API (Llama 3.3 70B) configured successfully.")
except Exception as e:
    logging.error("Error configuring NVIDIA API: %s", e)
    client = None

def call_nvidia_llm(prompt, temperature=0.2, top_p=0.7, max_tokens=1024, stream=False):
    """
    Calls the NVIDIA Llama 3.3 70B model with the given prompt.
    Returns the complete response as a string.
    """
    if not client:
        raise Exception("NVIDIA API client not configured")
    
    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            # Collect streamed chunks into a single string
            response_text = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    response_text += chunk.choices[0].delta.content
            return response_text
        else:
            # Non-streaming response
            return completion.choices[0].message.content
    
    except Exception as e:
        logging.error(f"Error calling NVIDIA LLM: {e}")
        raise
    
# Initialize Flask app
app = Flask(__name__)
# A secret key is crucial for securing sessions. It must be a long, random string.
app.secret_key = os.getenv('FLASK_SECRET_KEY')

# Enable CORS (Cross-Origin Resource Sharing)
# This is crucial for the frontend (your HTML file) to be able to make
# requests to this backend, as they are on different "origins" (the local file system
# versus the local server).
CORS(app, supports_credentials=True)


# Global variable for webcam access - this is okay to be global as it's a resource, not user data
camera = None

# --- Home page endpoints ---
@app.route('/api/health')
def health_check():
    """Health check endpoint to verify app status and model loading."""
    return jsonify({
        "status": "ok",
        "models_loaded": {
            "mlp_model": mlp_model is not None,
            "scaler": scaler is not None,
            "facial_emotion": True  # Always available after import
        },
        "app_dir": APP_DIR,
        "uploads_dir": UPLOADS_DIR,
        "uploads_dir_exists": os.path.exists(UPLOADS_DIR),
        "model_file_exists": os.path.exists(os.path.join(APP_DIR, "models", "mlp_emotion_model.joblib")),
        "scaler_file_exists": os.path.exists(os.path.join(APP_DIR, "models", "scaler.joblib"))
    })

@app.route('/')
def welcome():
    """Renders the main welcome page."""
    return render_template('welcome.html')

@app.route('/details')
def user_details():
    """Renders the user details form and handles submission."""
    return render_template('user_details.html')

@app.route('/submit_data', methods=['GET', 'POST'])
def submit_data():
    """Handles POST to store data and GET to provide data."""
    if request.method == 'POST':
        name = request.form.get('name')
        city = request.form.get('city')
        
        session['user_name'] = name
        session['user_city'] = city
        
        # Redirect to the facial analysis page
        return redirect(url_for('facial'))

    elif request.method == 'GET':
        name = session.get('user_name')
        city = session.get('user_city')
        
        # Return the data as JSON
        return jsonify({
            'name': name,
            'city': city
        })

# --- Facial emotion detection endpoints ---

def generate_frames():
    """
    A generator function that captures frames from the webcam,
    analyzes them for emotions, and encodes them for streaming.
    """
    global camera
    if not camera:
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        logging.error("Error: Could not open video device.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Analyze the frame and get the annotated frame back
            _, _, annotated_frame = facial_emotion.analyze_frame(frame)
            
            # Encode the frame as a JPEG image
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/facial')
def facial():
    """Renders the facial analysis page."""
    return render_template('facial.html')

# The endpoint that streams the video from the webcam
@app.route('/video_feed')
def video_feed():
    """
    This route streams the video frames from the webcam to the web page.
    It uses a multipart response to send a continuous stream of JPEG images.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# The endpoint to stop the webcam stream
@app.route('/stop_video', methods=['POST'])
def stop_video():
    """Stops the webcam stream."""
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "success"})

@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame_api():
    """
    Receives a base64 image from the frontend, analyzes it for emotion.
    This endpoint is STATELESS - it only analyzes and returns the result.
    The client is responsible for accumulating results.
    """
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'detected': False, 'emotion': 'No image provided', 'score': 0}), 400

        # Remove header "data:image/jpeg;base64,"
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Analyze the frame
        dominant_emotion, emotion_score, _ = facial_emotion.analyze_frame(frame)

        # Build response based on what was detected
        if dominant_emotion is None:
            # No face detected at all
            return jsonify({
                'detected': False,
                'emotion': 'No face detected',
                'score': 0
            })
        
        if dominant_emotion == "Uncertain":
            # Face detected but below confidence threshold
            return jsonify({
                'detected': False,
                'emotion': 'Uncertain',
                'score': round(float(emotion_score), 2) if emotion_score else 0
            })
        
        # Confident emotion detected
        return jsonify({
            'detected': True,
            'emotion': str(dominant_emotion),
            'score': round(float(emotion_score), 2)
        })

    except Exception as e:
        logging.error("Error in /api/analyze-frame: %s", e)
        return jsonify({'detected': False, 'emotion': 'Error', 'score': 0}), 500
    
@app.route('/api/get-emotions', methods=['GET'])
def get_emotions():
    """This endpoint is no longer the source of truth. 
    Kept for backward compatibility but client manages state now."""
    return jsonify({'emotions': session.get('emotion_results', [])})


@app.route('/api/reset-emotions', methods=['POST'])
def reset_emotions():
    """Clears any stored facial emotion results from the session."""
    session.pop('emotion_results', None)
    return jsonify({"status": "reset"})

# --- Voice emotion detection endpoints ---

# Create an 'uploads' directory if it doesn't exist
# Get the directory where this app.py file is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOADS_DIR = os.path.join(APP_DIR, "uploads")
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# --- 2. LOAD THE PRE-TRAINED MODEL AND SCALER ---
try:
    model_path = os.path.join(APP_DIR, "models", "mlp_emotion_model.joblib")
    scaler_path = os.path.join(APP_DIR, "models", "scaler.joblib")
    
    logging.info(f"Loading model from: {model_path}")
    logging.info(f"Loading scaler from: {scaler_path}")
    
    mlp_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logging.info("Model and scaler loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    mlp_model = None
    scaler = None

# --- 3. HELPER FUNCTION FOR FEATURE EXTRACTION ---
def extract_features(file_path, n_mfcc=40):
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio file.
    """
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}. Error: {e}")
        return None

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    """
    Endpoint to receive an audio file, make a prediction, and return the result.
    """
    logging.info("Received request for /predict_audio")
    # Check if models are loaded
    if mlp_model is None:
        logging.error("Model not loaded during prediction request")
        return jsonify({"error": "Voice emotion model not loaded. Check server logs."}), 500
    
    if scaler is None:
        logging.error("Scaler not loaded during prediction request")
        return jsonify({"error": "Scaler not loaded. Check server logs."}), 500

    if 'audio_file' not in request.files:
        logging.warning("No audio file in request.files")
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        logging.warning("Audio file has no filename")
        return jsonify({"error": "No selected file."}), 400

    temp_path = os.path.join(UPLOADS_DIR, "temp_audio_upload")
    wav_path = os.path.join(UPLOADS_DIR, "temp_audio.wav")

    try:
        logging.info(f"Saving uploaded file to: {temp_path}")
        audio_file.save(temp_path)
        
        logging.info(f"Loading audio file with pydub from: {temp_path}")
        audio = AudioSegment.from_file(temp_path)
        
        logging.info(f"Exporting audio to WAV format at: {wav_path}")
        audio.export(wav_path, format="wav")

        logging.info("Extracting features with librosa")
        mfccs = extract_features(wav_path)

        if mfccs is None:
            logging.error("Feature extraction failed, mfccs is None.")
            return jsonify({"error": "Failed to extract features from the audio file."}), 500

        logging.info("Scaling features and making prediction")
        scaled_mfccs = scaler.transform(mfccs.reshape(1, -1))
        predicted_emotion = mlp_model.predict(scaled_mfccs)[0]

        # Store the result in the session
        session['voice_emotion_result'] = {
            "emotion": str(predicted_emotion)
        }
        session.modified = True
        
        logging.info(f"Prediction successful: {predicted_emotion}")
        return jsonify({"emotion": predicted_emotion})

    except Exception as e:
        logging.error(f"Error during audio prediction: {e}")
        logging.error(f"Exception type: {type(e)}")
        # Check if the exception message contains hints about ffmpeg
        if "ffmpeg" in str(e).lower() or isinstance(e, FileNotFoundError):
             return jsonify({"error": "Audio processing failed. Please ensure FFmpeg is installed and accessible in your system's PATH."}), 500
        return jsonify({"error": "An error occurred during audio processing."}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

@app.route('/api/get-voice-emotion', methods=['GET'])
def get_voice_emotion():
    """
    Returns the last analyzed voice emotion from the session.
    """
    voice_emotion_result = session.get('voice_emotion_result', None)
    if not voice_emotion_result:
        return jsonify({"error": "No voice emotion result available."}), 404
    return jsonify(voice_emotion_result)

@app.route('/voice')
def voice():
    """Renders the voice analysis page."""
    return render_template('voice.html')

# --- Questionnaire endpoints ---
@app.route('/api/get_question_results', methods=['POST', 'GET'])
def get_question_results():
    """
    POST: Receives and stores questionnaire results.
    GET: Returns raw questionnaire results (legacy, session-based).
    """
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON data received'}), 400
            # Store in session (works locally, may not work on HF Spaces)
            session['questionnaire_results'] = data
            session.modified = True
            return jsonify({'message': 'Results received successfully!', 'data_received': data}), 200
        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify({'error': 'An internal server error occurred'}), 500
    
    elif request.method == 'GET':
        questionnaire_results = session.get('questionnaire_results', None)
        if not questionnaire_results:
            return jsonify({'message': 'No questionnaire results available yet.'}), 404
        return jsonify({'status': 'success', 'data': questionnaire_results}), 200


@app.route('/api/analyze_questionnaire', methods=['POST'])
def analyze_questionnaire():
    try:
        questionnaire_results = request.get_json()
        if not questionnaire_results:
            return jsonify({'status': 'error', 'message': 'No questionnaire data provided.'}), 400

        prompt = f"""
        Based on the following emotional and mental well-being data from a user questionnaire, provide a direct, empathetic, and encouraging feedback response in JSON format. The JSON object should have two keys: "summary" and "suggestions". The "summary" should be a string that summarizes the user's current state based on their ratings and feelings. The "suggestions" should be an array of strings, with each string containing an actionable and personalized suggestion for improvement.
        User Data:
        - Current Stress Level: {questionnaire_results.get('stressLevel', 'N/A')} out of 10
        - Overall Mood: {questionnaire_results.get('moodLevel', 'N/A')} out of 10
        - Energy Levels: {questionnaire_results.get('energyLevel', 'N/A')} out of 10
        - Feeling in one word: {questionnaire_results.get('feelingWord', 'N/A')}
        - Hours of Sleep: {questionnaire_results.get('sleepHours', 'N/A')}
        - Sleep Quality: {questionnaire_results.get('sleepQuality', 'N/A')}
        - Social Connection: {questionnaire_results.get('socialConnection', 'N/A')} out of 5
        - Physical Activity Today: {questionnaire_results.get('physicalActivity', 'N/A')}
        - Post-exercise Energy: {questionnaire_results.get('postExerciseEnergy', 'N/A')} out of 5 (if applicable)
        - Workload Stress: {questionnaire_results.get('workloadStress', 'N/A')} out of 5
        - Work-Life Balance: {questionnaire_results.get('workLifeBalance', 'N/A')} out of 5
        - Manager Support: {questionnaire_results.get('managerSupport', 'N/A')}
        - Biggest Emotional Challenge at Work: {questionnaire_results.get('corporateFeedback', 'N/A')}
        
        Return only a valid JSON object with "summary" and "suggestions" keys.
        """
        
        response_text = call_nvidia_llm(prompt, temperature=0.3, max_tokens=1500)
        logging.info("Questionnaire analysis response: %s", response_text)
        
        if response_text.strip().startswith('```json'):
            response_text = response_text.strip()[7:-3].strip()
        
        analysis_dict = json.loads(response_text)
        return jsonify({"status": "success", "analysis_and_suggestions": analysis_dict})
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from LLM response: {e}")
        return jsonify({"status": "error", "message": "Invalid analysis format received from LLM."}), 500
    except Exception as e:
        logging.error(f"Error during questionnaire analysis: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred. Please try again."}), 500
            
@app.route('/questionnaire')
def questionnaire():
    """Renders the questionnaire page."""
    return render_template('questionnaire.html')

# --- Wellness stats and news endpoints (Global variables are fine here for caching) ---
# Cache for wellness stats - This is okay to be global because it's application-wide data, not user-specific.
wellness_stats_cache = {
    'data': None,
    'timestamp': 0
}
CACHE_DURATION = 60 * 60 # 1 hour in seconds

@app.route('/api/wellness-snapshot')
def wellness_snapshot():
    global wellness_stats_cache
    now = time.time()

    if wellness_stats_cache['data'] is not None and (now - wellness_stats_cache['timestamp'] < CACHE_DURATION):
        return jsonify(wellness_stats_cache['data'])

    if not client:
        logging.error("NVIDIA API not configured.")
        return jsonify({"error": "AI API not configured"}), 500

    fallback_data = [
        {"title": "Engagement", "value": "85%", "description": "Employee engagement remains high.", "color": "text-teal-400"},
        {"title": "Stress Levels", "value": "15% ↓", "description": "Average stress levels have decreased this week.", "color": "text-blue-400"},
        {"title": "Burnout Risk", "value": "7%", "description": "Percentage of employees at high risk of burnout.", "color": "text-purple-400"},
        {"title": "Positive Sentiment", "value": "92%", "description": "Sentiment in team communications is positive.", "color": "text-amber-400"},
        {"title": "Wellness Sessions", "value": "4.8/5", "description": "Average rating for wellness workshops.", "color": "text-emerald-400"}
    ]

    try:
        prompt = """
        Generate a list of 5 realistic but fictional daily corporate wellness statistics for a company dashboard.
        The statistics should be diverse, covering topics like engagement, stress, burnout, and positive sentiment.
        Provide the output as a valid JSON array of objects. Each object should have the following keys:
        - "title": The name of the metric (e.g., "Engagement", "Stress Levels").
        - "value": The main statistic value as a string (e.g., "85%", "15% ↓", "7%"). Should be a number or percentage or a rating.
        - "description": A short, one-sentence description of the statistic.
        - "color": A Tailwind CSS text color class (e.g., "text-teal-400", "text-blue-400", "text-purple-400", "text-amber-400", "text-emerald-400").

        Return only the JSON array, without any markdown formatting.
        """
        
        response_text = call_nvidia_llm(prompt, temperature=0.5, max_tokens=1500)
        logging.info("NVIDIA response for wellness stats: %s", response_text)
        
        cleaned_response = response_text.strip().replace('```json', '').replace('```', '').strip()
        
        if not cleaned_response:
            logging.error("NVIDIA API returned empty response.")
            return jsonify(fallback_data), 500

        stats_data = json.loads(cleaned_response)
        wellness_stats_cache['data'] = stats_data
        wellness_stats_cache['timestamp'] = now
        return jsonify(stats_data)
        
    except Exception as e:
        logging.error("Error in /api/wellness-snapshot: %s", e)
        wellness_stats_cache['data'] = fallback_data
        wellness_stats_cache['timestamp'] = now
        return jsonify(fallback_data), 500
    
@app.route('/api/news-snapshot')
def news_snapshot():
    if not client:
        logging.error("NVIDIA API not configured.")
        return jsonify({"error": "AI API not configured"}), 500

    fallback_data = [
        {"title": "Mindfulness at Work: A Guide", "description": "Learn simple techniques to integrate mindfulness into your daily routine for a more focused and productive day."},
        {"title": "The Importance of Digital Detox", "description": "Why taking breaks from screens can improve your mental health, reduce stress, and improve real-life social connections."},
        {"title": "Building a Resilient Team", "description": "Expert tips on fostering a supportive and resilient work environment through open communication and adaptability."},
        {"title": "The Power of a Five-Minute Walk", "description": "Discover how short, brisk walks can boost creativity, reduce stress, and improve brain power throughout the day."},
        {"title": "Nutrition for Mental Clarity", "description": "How a balanced diet rich in healthy fats, whole grains, and leafy greens can have a profound impact on cognitive function."}
    ]

    try:
        prompt = """
        Generate a list of 5 short, positive wellness news headlines and descriptions for a company dashboard.
        For each news item, provide:
        - "title": The news headline.
        - "description": A very short, one-sentence summary of the article.
        Return only the JSON array, without any markdown formatting.
        """
        
        response_text = call_nvidia_llm(prompt, temperature=0.6, max_tokens=1000)
        logging.info("NVIDIA response for news: %s", response_text)
        
        cleaned_response = response_text.strip().replace('```json', '').replace('```', '').strip()
        
        if not cleaned_response:
            logging.error("NVIDIA API returned empty response for news.")
            return jsonify(fallback_data), 500

        news_data = json.loads(cleaned_response)
        return jsonify(news_data)

    except Exception as e:
        logging.error("Error in /api/news-snapshot: %s", e)
        return jsonify(fallback_data), 500
    
@app.route('/api/combined-analysis', methods=['POST'])
def combined_analysis():
    if not client:
        logging.error("NVIDIA API not configured for combined analysis.")
        return jsonify({"status": "error", "message": "AI model is not configured."}), 500

    name = session.get('user_name', 'there')
    
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Invalid request body."}), 400
        
    facial_results = data.get('facial_emotions_list', [])
    voice_result = data.get('vocal_emotion', {})
    
    logging.info("Received facial results: %s", facial_results)
    logging.info("Received voice result: %s", voice_result)

    if not facial_results and not voice_result:
        return jsonify({
            "status": "error",
            "message": "No facial or voice analysis data found. Please complete the previous steps."
        }), 404

    facial_summary = ", ".join([f"{res['emotion']} ({res['score'] * 100:.0f}%)" for res in facial_results]) if facial_results else "No facial data provided."
    voice_emotion = voice_result if isinstance(voice_result, str) else 'No voice data provided.'

    prompt = f"""
    As a compassionate wellness assistant named ShantiView, analyze the following emotional data for a user named {name} and provide a holistic, empathetic summary and actionable suggestions.

    **User's Emotional Data:**
    - **Facial Expressions Detected (chronological):** {facial_summary}
    - **Vocal Tone Emotion:** {voice_emotion}

    **Your Task:**
    Based on this combined data, generate a response in a valid JSON format with two keys: "summary" and "suggestions".

    1.  **"summary" (string):** Write a concise, empathetic paragraph (3-4 sentences) that synthesizes the data. Acknowledge the user's emotional state as indicated by both their facial expressions and voice. If there's a conflict (e.g., smiling face but stressed voice), gently point it out as a sign of potential emotional masking.
    2.  **"suggestions" (array of strings):** Provide 3 concise, empathetic actionable, positive, and personalized suggestions. These should be practical tips for improving well-being, tailored to the detected emotions.

    Please provide only the raw JSON object in your response.
    """

    try:
        response_text = call_nvidia_llm(prompt, temperature=0.3, max_tokens=1500)
        logging.info("Combined analysis response from NVIDIA: %s", response_text)

        if response_text.strip().startswith('```json'):
            response_text = response_text.strip()[7:-3].strip()
        
        analysis_dict = json.loads(response_text)
        return jsonify({"status": "success", "analysis": analysis_dict})

    except Exception as e:
        logging.error(f"Error during combined analysis: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred. Please try again."}), 500
    
@app.route('/api/suggestions', methods=['POST'])
def api_suggestions():
    if not client:
        logging.error("NVIDIA API not configured.")
        return jsonify({"error": "AI API not configured"}), 500

    try:
        data = request.get_json()
        location = data.get("location")

        if not location:
            logging.warning("No location provided in request.")
            return jsonify({"error": "No location provided"}), 400

        prompt = f"""
        You are ShantiView, a wellness assistant. Provide specific suggestions for Mindfulness, Food, Music, and Community related to the location '{location}'.
        Provide 3 suggestions for each category. For each suggestion, provide a title and a brief description.
        
        Return your response as a valid JSON array following this exact structure:
        [
            {{
                "category": "Mindfulness",
                "suggestions": [
                    {{"title": "...", "description": "..."}},
                    {{"title": "...", "description": "..."}},
                    {{"title": "...", "description": "..."}}
                ]
            }},
            {{
                "category": "Food",
                "suggestions": [...]
            }},
            {{
                "category": "Music",
                "suggestions": [...]
            }},
            {{
                "category": "Community",
                "suggestions": [...]
            }}
        ]
        
        Return only the JSON array without any markdown formatting or explanation.
        """

        response_text = call_nvidia_llm(prompt, temperature=0.4, max_tokens=2000)
        logging.info("NVIDIA response for suggestions: %s", response_text)
        
        cleaned_response = response_text.strip().replace('```json', '').replace('```', '').strip()
        suggestions_data = json.loads(cleaned_response)
        return jsonify(suggestions_data)

    except Exception as e:
        logging.error("Error in /api/suggestions: %s", e)
        return jsonify({"error": "An error occurred while processing your request."}), 500
    
@app.route('/results', methods=['GET', 'POST'])
def results():
    """Renders the results page."""
    # If you expect POST data, handle it here
    return render_template('results.html')

# Chatbot endpoints
@app.route('/chatbot')
def chatbot():
    """Renders the chatbot page."""
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if not client:
        logging.error("NVIDIA API not configured.")
        return jsonify({"error": "AI API not configured"}), 500

    try:
        data = request.get_json()
        user_message = data.get("message")

        if not user_message:
            logging.warning("No message provided in request.")
            return jsonify({"error": "No message provided"}), 400

        prompt = f"You are ShantiView, a friendly and supportive wellness assistant. A user is talking to you. User's message: '{user_message}'. Your response should be helpful and focused on mental wellness, mindfulness, or providing a supportive ear. Be concise and empathetic."
        
        response_text = call_nvidia_llm(prompt, temperature=0.6, max_tokens=500, stream=True)
        
        logging.info("User message: '%s'", user_message)
        logging.info("ShantiView response: '%s'", response_text)

        return jsonify({"response": response_text})

    except Exception as e:
        logging.error("Error in /api/chat: %s", e)
        return jsonify({"error": "An error occurred while processing your request."}), 500
                
# --- Main app execution ---
if __name__ == '__main__':
    # Ensure the camera is released when the app exits
    def cleanup():
        global camera
        if camera:
            camera.release()
            
    atexit.register(cleanup)
    
    # Start the Flask server
    # For production (Render), disable debug and use dynamic port
    port = int(os.environ.get('PORT', 7860))
    debug_mode = os.environ.get('FLASK_ENV') == 'development' 
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
