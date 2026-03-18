# Import packages

import os
import google.generativeai as genai
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


# Load environment variables from .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
    
# Load the Gemini API key from the environment
try:
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    logging.info("Gemini API configured successfully.")
except Exception as e:
    logging.error("Error configuring Gemini API: %s", e)
    model = None
    
# Initialize Flask app
app = Flask(__name__)
# A secret key is crucial for securing sessions. It must be a long, random string.
app.secret_key = os.getenv('FLASK_SECRET_KEY')

# Enable CORS (Cross-Origin Resource Sharing)
# This is crucial for the frontend (your HTML file) to be able to make
# requests to this backend, as they are on different "origins" (the local file system
# versus the local server).
CORS(app)


# Global variable for webcam access - this is okay to be global as it's a resource, not user data
camera = None

# --- Home page endpoints ---
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
    Receives a base64 image from the frontend, analyzes it for emotion,
    and stores the result in the session.
    """
    # Use session to store emotion results
    if 'emotion_results' not in session:
        session['emotion_results'] = []
    
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        # Remove header "data:image/jpeg;base64,"
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Analyze the frame
        dominant_emotion, emotion_score, _ = facial_emotion.analyze_frame(frame)

        # The emotion to be sent back to the frontend for status updates
        response_emotion = "No face detected"

        if dominant_emotion:
            response_emotion = dominant_emotion
            # Only add confident emotions to our results list
            if dominant_emotion != "Uncertain":
                emotion_results = session.get('emotion_results', [])
                emotion_results.append({
                    "emotion": str(dominant_emotion),
                    "score": float(round(emotion_score, 2))
                })
                # Keep only the last 15
                if len(emotion_results) > 15:
                    emotion_results = emotion_results[-15:]
                session['emotion_results'] = emotion_results # Update the session
                
        return jsonify({
            'emotion': response_emotion,
            'count': len(session.get('emotion_results', [])),
            'done': len(session.get('emotion_results', [])) >= 15
        })
    except Exception as e:
        logging.error("Error in /api/analyze-frame: %s", e)
        return jsonify({'error': 'Emotion analysis failed'}), 500

@app.route('/api/get-emotions', methods=['GET'])
def get_emotions():
    """Returns the list of captured emotions from the session."""
    return jsonify({'emotions': session.get('emotion_results', [])})

@app.route('/api/reset-emotions', methods=['POST'])
def reset_emotions():
    """Clears the stored facial emotion results from the session."""
    session.pop('emotion_results', None)
    return jsonify({"status": "reset"})


# --- Voice emotion detection endpoints ---

# Create an 'uploads' directory if it doesn't exist
UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# --- 2. LOAD THE PRE-TRAINED MODEL AND SCALER ---
try:
    mlp_model = joblib.load("./models/mlp_emotion_model.joblib")
    scaler = joblib.load("./models/scaler.joblib")
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
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
    if mlp_model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded."}), 500

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    temp_path = os.path.join(UPLOADS_DIR, "temp_audio_upload")
    wav_path = os.path.join(UPLOADS_DIR, "temp_audio.wav")

    try:
        audio_file.save(temp_path)
        audio = AudioSegment.from_file(temp_path)
        audio.export(wav_path, format="wav")

        mfccs = extract_features(wav_path)

        if mfccs is None:
            return jsonify({"error": "Failed to extract features from the audio file."}), 500

        scaled_mfccs = scaler.transform(mfccs.reshape(1, -1))
        predicted_emotion = mlp_model.predict(scaled_mfccs)[0]

        # Store the result in the session
        session['voice_emotion_result'] = {
            "emotion": str(predicted_emotion)
        }

        return jsonify({"emotion": predicted_emotion})

    except Exception as e:
        logging.error(f"Error during audio prediction: {e}")
        if isinstance(e, FileNotFoundError):
             return jsonify({"error": "Failed to process audio file. Ensure FFmpeg is installed and in your system's PATH."}), 500
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
@app.route('/api/get_question_results', methods=['POST','GET'])
def get_question_results():
    """
    This endpoint receives and processes the questionnaire results.
    It stores the results in the session.
    """
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON data received'}), 400
            # Store the results in the session
            session['questionnaire_results'] = data
            return jsonify({'message': 'Results received successfully!', 'data_received': data}), 200
        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify({'error': 'An internal server error occurred'}), 500
    elif request.method == 'GET':
        questionnaire_results = session.get('questionnaire_results', None)
        if not questionnaire_results:
            return jsonify({'message': 'No questionnaire results available yet.'}), 404

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
        """
        try:
            response = model.generate_content(prompt)
            analysis_text = response.text
            print("Results feedback: ")
            print(analysis_text)
            if analysis_text.strip().startswith('```json'):
                analysis_text = analysis_text.strip()[7:-3].strip()
            analysis_dict = json.loads(analysis_text)
            return jsonify({
                "status": "success",
                "analysis_and_suggestions": analysis_dict
            })
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from LLM response: {e}")
            return jsonify({
                "status": "error",
                "message": "Invalid analysis format received from LLM."
            }), 500
        except Exception as e:
            print(f"An unexpected error occurred during LLM call: {e}")
            return jsonify({
                "status": "error",
                "message": "An unexpected error occurred. Please try again."
            }), 500

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
    """API endpoint to get corporate wellness stats from Gemini, with caching."""
    global wellness_stats_cache
    now = time.time()

    # Serve from cache if not expired
    if wellness_stats_cache['data'] is not None and (now - wellness_stats_cache['timestamp'] < CACHE_DURATION):
        return jsonify(wellness_stats_cache['data'])

    if not model:
        logging.error("Gemini API not configured.")
        return jsonify({"error": "Gemini API not configured"}), 500

    fallback_data = [
        {"title": "Engagement", "value": "85%", "description": "Employee engagement remains high.", "color": "text-teal-400"},
        {"title": "Stress Levels", "value": "15% ↓", "description": "Average stress levels have decreased this week.", "color": "text-blue-400"},
        {"title": "Burnout Risk", "value": "7%", "description": "Percentage of employees at high risk of burnout.", "color": "text-purple-400"},
        {"title": "Positive Sentiment", "value": "92%", "description": "Sentiment in team communications is positive.", "color": "text-amber-400"},
        {"title": "Wellness Sessions", "value": "4.8/5", "description": "Average rating for wellness workshops.", "color": "text-emerald-400"}
    ]

    try:
        # A prompt designed to get structured JSON output.
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
        
        response = model.generate_content(prompt)

        # Log the prompt and response for debugging
        # logging.info("Prompt sent to Gemini: %s", prompt)
        logging.info("Response from Gemini: %s", response.text)
        
        # Explicitly check for valid content before processing
        if not (response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
            logging.error("Gemini API returned an invalid response structure.")
            return jsonify(fallback_data), 500

        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        # Check for empty response before parsing
        if not cleaned_response:
            logging.error("Gemini API returned an empty or invalid JSON string.")
            return jsonify(fallback_data), 500

        # Attempt to parse the cleaned JSON string
        stats_data = json.loads(cleaned_response)
                
        # Update cache
        wellness_stats_cache['data'] = stats_data
        wellness_stats_cache['timestamp'] = now
        return jsonify(stats_data)
    except Exception as e:
        logging.error("Error in /api/wellness-snapshot: %s", e)
        # Update cache with fallback
        wellness_stats_cache['data'] = fallback_data
        wellness_stats_cache['timestamp'] = now
        logging.warning("Returning fallback data due to API error.")
        return jsonify(fallback_data), 500

@app.route('/api/news-snapshot')
def news_snapshot():
    """API endpoint to get a snapshot of wellness news from Gemini."""
    if not model:
        logging.error("Gemini API not configured.")
        return jsonify({"error": "Gemini API not configured"}), 500

    fallback_data = [
        {
    "title": "Mindfulness at Work: A Guide",
    "description": "Learn simple techniques to integrate mindfulness into your daily routine for a more focused and productive day."
        },
    {
    "title": "The Importance of Digital Detox",
    "description": "Why taking breaks from screens can improve your mental health, reduce stress, and improve real-life social connections."
    },
    {
    "title": "Building a Resilient Team",
    "description": "Expert tips on fostering a supportive and resilient work environment through open communication and adaptability."
    },
    {
    "title": "The Power of a Five-Minute Walk",
    "description": "Discover how short, brisk walks can boost creativity, reduce stress, and improve brain power throughout the day."
    },
  {
    "title": "Nutrition for Mental Clarity",
    "description": "How a balanced diet rich in healthy fats, whole grains, and leafy greens can have a profound impact on cognitive function."
  }
]

    try:
        # Updated prompt for real URLs
        prompt = """
        Generate a list of 5 short, positive wellness news headlines and descriptions for a company dashboard.
        For each news item, provide:
        - "title": The news headline.
        - "description": A very short, one-sentence summary of the article.
        Return only the JSON array, without any markdown formatting.

        """
        
        response = model.generate_content(prompt)

        # log response
        logging.info("Gemini response for news: %s", response.text)
        if not (response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
            logging.error("Gemini API returned an invalid response structure for news.")
            return jsonify(fallback_data), 500
        
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        if not cleaned_response:
            logging.error("Gemini API returned an empty or invalid JSON string for news.")
            return jsonify(fallback_data), 500

        news_data = json.loads(cleaned_response)
                
        return jsonify(news_data)

    except json.JSONDecodeError as e:
        logging.error("JSONDecodeError in /api/news-snapshot: %s. Response text: '%s'", e, cleaned_response)
        return jsonify(fallback_data), 500
    except Exception as e:
        logging.error("Error in /api/news-snapshot: %s", e)
        logging.warning("Returning fallback data for news due to API error.")
        return jsonify(fallback_data), 500

@app.route('/api/combined-analysis', methods=['POST'])
def combined_analysis():
    """
    Fetches facial and voice emotion data from the request body, and uses the Gemini
    model to generate a combined analysis and suggestions.
    """
    if not model:
        logging.error("Gemini API not configured for combined analysis.")
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

    # Prepare the data for the prompt
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
    2.  **"suggestions" (array of strings):** Provide 3 concise, empathetic actionable, positive, and personalized suggestions. These should be practical tips for improving well-being, tailored to the detected emotions. For example, if stress is detected, suggest a breathing exercise. If sadness is detected, suggest a self-compassion activity.


    Please provide only the raw JSON object in your response.
    """

    try:
        response = model.generate_content(prompt)
        analysis_text = response.text
        logging.info("Combined analysis response from Gemini: %s", analysis_text)

        # Clean up the response to ensure it's valid JSON
        if analysis_text.strip().startswith('```json'):
            analysis_text = analysis_text.strip()[7:-3].strip()
        
        analysis_dict = json.loads(analysis_text)
        
        logging.info("Parsed combined analysis: %s", {"status": "success", "analysis": analysis_dict} )

        return jsonify({"status": "success", "analysis": analysis_dict})

    except Exception as e:
        logging.error(f"An unexpected error occurred during combined analysis LLM call: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred while generating your analysis. Please try again."}), 500

@app.route('/api/suggestions', methods=['POST'])
def api_suggestions():
    """
    API endpoint to fetch personalized wellness suggestions from the Gemini model
    based on a provided location. The model is instructed to return a structured
    JSON response without images.
    """
    if not model:
        logging.error("Gemini API not configured.")
        return jsonify({"error": "Gemini API not configured"}), 500

    try:
        data = request.get_json()
        location = data.get("location")

        if not location:
            logging.warning("No location provided in request.")
            return jsonify({"error": "No location provided"}), 400

        # System prompt to guide the model's behavior and desired output format
        prompt = (
            f"You are ShantiView- a wellness assistant. Provide specific suggestions for "
            f"Mindfulness, Food, Music, and Community related to the location '{location}'. "
            f"Provide 3 suggestions for each category. For each suggestion, provide a "
            f"title and a brief description. The response must be "
            f"in JSON format, following the provided schema."
        )
        
        # Generation configuration with a JSON schema to ensure a structured response
        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "category": { "type": "STRING" },
                        "suggestions": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "title": { "type": "STRING" },
                                    "description": { "type": "STRING" }
                                }
                            }
                        }
                    }
                }
            }
        }

        # Generate content from the model
        response = model.generate_content(prompt, generation_config=generation_config)

        # The response text is a JSON string, so we can return it directly
        return jsonify(json.loads(response.text))

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
    """API endpoint to interact with the Gemini model."""
    if not model:
        logging.error("Gemini API not configured.")
        return jsonify({"error": "Gemini API not configured"}), 500

    try:
        data = request.get_json()
        user_message = data.get("message")

        if not user_message:
            logging.warning("No message provided in request.")
            return jsonify({"error": "No message provided"}), 400

        # System prompt to guide the model's behavior
        prompt = f"You are ShantiView, a friendly and supportive wellness assistant. A user is talking to you. User's message: '{user_message}'. Your response should be helpful and focused on mental wellness, mindfulness, or providing a supportive ear. Be concise and empathetic."
        response = model.generate_content(prompt)
        
        # Log the user message and the bot's response
        logging.info("User message: '%s'", user_message)
        logging.info("ShantiView response: '%s'", response.text)

        return jsonify({"response": response.text})

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
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
