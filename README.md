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

A comprehensive mental health analysis platform that combines multiple emotion recognition techniques with AI-powered insights and personalized wellness recommendations. Built with **FastAPI**, **React**, and **LangGraph Multi-Agent System**.

## Features

- **Facial Emotion Detection**: Real-time emotion recognition from webcam feed using deep learning
- **Voice Emotion Recognition**: Analyze emotional states from audio files using neural networks
- **Mental Health Questionnaire**: Structured assessment of mental well-being
- **LangGraph Multi-Agent System**: Parallel LLM execution for faster responses
- **AI-Powered Insights**: NVIDIA Llama 3.3 70B for personalized insights and recommendations
- **Wellness Dashboard**: Comprehensive view combining facial emotions, voice analysis, and questionnaire results
- **AI Chatbot**: Interactive chatbot for wellness support and guidance
- **Smart Recommendations**: Personalized wellness suggestions based on combined analysis

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **LangGraph**: Multi-agent orchestration with parallel execution
- **LangChain NVIDIA AI Endpoints**: Llama 3.3 70B integration
- **Machine Learning**: TensorFlow, Keras, scikit-learn
- **Computer Vision**: OpenCV, DeepFace
- **Audio Processing**: Librosa, pydub

### Frontend
- **React 19**: Modern UI library
- **Vite**: Fast build tool and dev server
- **TailwindCSS**: Utility-first CSS framework
- **React Router**: Client-side routing
- **Lucide React**: Beautiful icons

## Prerequisites

- **Node.js 18+** and **pnpm** (for frontend)
- **Python 3.10+** and **uv** (for backend)
- **NVIDIA API key** from [NVIDIA NIM](https://build.nvidia.com/)
- Webcam (for facial emotion detection)
- Microphone (for voice emotion detection)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd ShantiView
```

### 2. Set up environment variables

Create a `.env` file in the project root with your NVIDIA API key:

```
NVIDIA_API_KEY=nvapi-your-actual-api-key-here
PORT=7860
```

### 3. Install Backend Dependencies

```bash
cd backend
uv sync
```

### 4. Install Frontend Dependencies

```bash
cd frontend
pnpm install
```

## Application Flow

The application follows a sequential wellness journey:

1. **Welcome Page** → User starts here
2. **User Details Page** → User enters name and location
3. **Facial Analysis Page** → Real-time facial emotion detection
4. **Voice Analysis Page** → Voice emotion analysis
5. **Questionnaire Page** → Mental health questionnaire
6. **Results Page** → Comprehensive wellness report

## Running the Application

### Option 1: Local Development (Separate Servers)

For local development, the Vite dev server proxies API requests to the backend automatically.

**Start the Backend (Terminal 1):**

```bash
cd backend
uv run uvicorn app.main:app --reload --port 7860
```

The backend API will be available at `http://localhost:7860`
- API Documentation: `http://localhost:7860/api/docs`

**Start the Frontend (Terminal 2):**

```bash
cd frontend
pnpm dev
```

The frontend will be available at `http://localhost:3000`

### Option 2: Docker (Production)

Build and run using Docker:

```bash
# Build the Docker image
docker build -t shantiview .

# Run the container (port 7860)
docker run -p 7860:7860 --env-file .env shantiview
```

The application will be available at `http://localhost:7860`

### Option 3: Hugging Face Space

The application is configured to run on Hugging Face Spaces using Docker. The `app_port` in the README metadata is set to 7860.

## Project Structure

```
ShantiView/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI application
│   │   ├── routes.py         # API endpoints
│   │   └── agents.py         # LangGraph multi-agent system
│   ├── pyproject.toml        # Python dependencies
│   └── requirements.txt      # Package list
├── frontend/
│   ├── src/
│   │   ├── pages/            # React page components
│   │   ├── App.jsx           # Main app with routing
│   │   ├── main.jsx          # Entry point
│   │   └── index.css         # Global styles
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── models/                   # ML models
│   ├── facial_emotion.py
│   ├── mlp_emotion_model.joblib
│   └── scaler.joblib
├── uploads/                  # Audio uploads directory
├── .env                      # Environment variables
├── .env.example              # Environment template
└── Dockerfile                # Multi-stage Docker build
```

## API Endpoints

### Health & Dashboard
- `GET /` - API info
- `GET /api/health` - Health check
- `GET /api/wellness-snapshot` - Get wellness statistics
- `GET /api/news-snapshot` - Get wellness news

### Analysis
- `POST /api/analyze_questionnaire` - Analyze questionnaire responses
- `POST /api/combined-analysis` - Combined facial + voice analysis
- `POST /api/suggestions` - Get location-based suggestions
- `POST /api/chat` - Chat with AI assistant
- `POST /predict_audio` - Analyze audio file

### Batch Processing
- `POST /api/batch-analysis` - Run multiple analyses in parallel

## LangGraph Multi-Agent Architecture

The application uses a multi-agent system for parallel LLM execution:

- **QuestionnaireAgent**: Analyzes questionnaire responses
- **WellnessStatsAgent**: Generates wellness statistics
- **NewsAgent**: Creates wellness news content
- **CombinedAnalysisAgent**: Combines facial and voice emotion data
- **SuggestionsAgent**: Generates location-based recommendations
- **ChatAgent**: Handles chat conversations

Agents run in parallel using `asyncio.gather()` for faster responses when multiple analyses are needed.

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | Your NVIDIA API key (required) |

## Development

### Backend Development

```bash
cd backend
uv run uvicorn app.main:app --reload --port 7860
```

### Frontend Development

```bash
cd frontend
pnpm dev
```

The frontend Vite server (port 3000) automatically proxies API requests to the backend at `http://localhost:7860`.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA NIM for AI capabilities (Llama 3.3 70B)
- LangGraph for multi-agent orchestration
- DeepFace for facial recognition
- OpenCV for computer vision
- Librosa for audio processing
- TensorFlow/Keras for deep learning