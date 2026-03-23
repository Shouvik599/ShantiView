---
title: ShantiView
emoji: 🧘
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
# ShantiView

A comprehensive mental health analysis platform that combines multiple emotion recognition techniques with AI-powered insights and personalized wellness recommendations.

## Features

- **Facial Emotion Detection**: Real-time emotion recognition from webcam feed using deep learning
- **Voice Emotion Recognition**: Analyze emotional states from audio files using neural networks
- **Mental Health Questionnaire**: Structured assessment of mental well-being
NVIDIA Llama 3.3 70B for personalized insights and recommendations
- **Wellness Dashboard**: Comprehensive view combining facial emotions, voice analysis, and questionnaire results
- **AI Chatbot**: Interactive chatbot for wellness support and guidance
- **Smart Recommendations**: Personalized wellness suggestions based on combined analysis

## Tech Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow, Keras, scikit-learn
- **Computer Vision**: OpenCV, DeepFace
- **Audio Processing**: Librosa, pydub
AI Integration**: NVIDIA NIM (Llama 3.3 70B Instruct)
- **Frontend**: HTML, CSS, JavaScript
- **Environment Manager**: python-dotenv

## Prerequisites

- Python 3.8+
- Webcam (for facial emotion detection)
- Microphone (for voice emotion detection)
- NVIDIA API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ShantiView
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
  NVIDIA_API_KEY=your_nvidia_api_key_here
   FLASK_SECRET_KEY=your_secret_key_here
   FLASK_ENV=development
   ```

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open in browser**
   Navigate to `http://localhost:5000`

3. **Features Available**
   - **Welcome Page**: Introduction and navigation
   - **User Details**: Enter name and location
   - **Facial Analysis**: Real-time emotion detection via webcam
   - **Voice Analysis**: Upload audio files for emotion recognition
   - **Questionnaire**: Mental health assessment
   - **Results**: View comprehensive analysis
   - **Chatbot**: Get AI-powered wellness support

## Project Structure

```
ShantiView/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── test.py                     # Test suite
├── .env                        # Environment variables (not committed)
├── models/                     # Machine learning models
│   ├── facial_emotion.py       # Facial emotion detection module
│   ├── mlp_emotion_model.joblib# Trained MLP model for voice
│   ├── scaler.joblib           # Data scaler for normalization
│   ├── voice_emotion_cnn.ipynb # CNN model notebook
│   └── voice_emotion_mlp.ipynb # MLP model notebook
├── templates/                  # HTML templates
│   ├── welcome.html            # Home page
│   ├── user_details.html       # User info form
│   ├── facial.html             # Facial analysis page
│   ├── voice.html              # Voice analysis page
│   ├── questionnaire.html      # Mental health questionnaire
│   ├── results.html            # Results dashboard
│   ├── chatbot.html            # AI chatbot interface
│   └── chatbot.html            # Combined analysis results
├── uploads/                    # User uploaded audio files
├── ffmpeg/                     # FFmpeg binaries for audio processing
└── __pycache__/                # Python cache files
```

## API Endpoints

### Core Endpoints
- `GET /` - Welcome page
- `GET /details` - User details form
- `POST /submit_data` - Save user information
- `GET /facial` - Facial emotion detection page
- `GET /video_feed` - Video stream endpoint
- `POST /stop_video` - Stop video stream

### Analysis Endpoints
- `POST /api/analyze-frame` - Analyze single frame for emotions
- `GET /api/get-emotions` - Get accumulated emotions from session
- `POST /api/reset-emotions` - Reset emotion data
- `POST /predict_audio` - Analyze audio file
- `GET /api/get-voice-emotion` - Get voice emotion results

### Questionnaire & Results
- `GET /questionnaire` - Mental health questionnaire
- `POST /api/get_question_results` - Process questionnaire results
- `GET /results` - View comprehensive results

### AI & Wellness
- `GET /api/wellness-snapshot` - Get wellness analysis
- `GET /api/news-snapshot` - Get wellness news
- `POST /api/combined-analysis` - Combined emotion analysis
- `POST /api/suggestions` - Get AI recommendations
- `POST /api/chat` - AI chatbot endpoint

## Configuration

### Environment Variables
- `NVIDIA_API_KEY`: Your NVIDIA API key
- `FLASK_SECRET_KEY`: Secret key for Flask session management

### Audio Settings
- Supported formats: WAV, MP3
- FFmpeg is included in the `ffmpeg/` directory for audio processing

## Development

### Running Tests
```bash
python test.py
```

### Model Details
- **Facial Emotion Model**: Uses DeepFace with TensorFlow backend
- **Voice Emotion Model**: MLP neural network trained on audio features (MFCC)
- **Scaler**: Scikit-learn StandardScaler for feature normalization

## Known Limitations

- Facial detection requires good lighting conditions
- Audio analysis works best with clear voice samples
- Real-time processing depends on system resources

## Future Enhancements

- Multi-user support with database persistence
- Mobile-friendly responsive design improvements
- Integration with wearable health devices
- Advanced analytics and trend analysis
- Export wellness reports

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions, please open an issue on the repository or contact the development team.

## Acknowledgments

- NVIDIA Llama 3.3 70B for AI capabilities
- DeepFace for facial recognition
- OpenCV for computer vision
- Librosa for audio processing
- TensorFlow/Keras for deep learning

## Demo
You can view the application demo at : https://shouvik99-shantiview.hf.space