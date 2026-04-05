"""
ShantiView - AI Corporate Wellness Assistant
FastAPI Main Application with LangGraph Multi-Agent System
"""

import os
import re
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

def train_model_if_missing():
    """Train the CNN model at startup if model files don't exist (Docker/HF only)."""
    # Only train in Docker environment, not locally
    # In Docker, the backend directory is at /app, locally it's at C:\...\backend
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    is_docker = backend_dir == "/app"
    
    if not is_docker:
        logger.info("Running locally, skipping model training. Using existing model files.")
        return
    
    # In Docker: check if model exists, if not train
    model_path = os.path.join(backend_dir, "models", "ravdess_cnn_model.h5")
    
    if not os.path.exists(model_path):
        logger.info("Model not found in Docker. Starting model training...")
        try:
            import subprocess
            env = os.environ.copy()
            env["PYTHONPATH"] = backend_dir
            result = subprocess.run(
                ["python", "models/train_cnn.py"],
                capture_output=True,
                text=True,
                env=env,
                cwd=backend_dir
            )
            if result.returncode == 0:
                logger.info("Model training completed successfully")
            else:
                logger.error(f"Model training failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
    else:
        logger.info("Model already exists in Docker, skipping training")

# Train model at startup if missing
train_model_if_missing()

# Import routes
from app.routes import router

# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="ShantiView API",
    description="AI-powered Corporate Wellness Assistant with LangGraph Multi-Agent System",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# ============================================================
# MIDDLEWARE
# ============================================================

# Setup CORS for React frontend (local development and Hugging Face Space)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https://.*\.hf\.space$",  # Match Hugging Face Spaces URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# INCLUDE ROUTES
# ============================================================

app.include_router(router)

# ============================================================
# STATIC FILES - Mount React frontend
# ============================================================

# Mount static files directory to serve React frontend
# This serves the built React app from the /static folder (matching Docker path /app/static)
static_dir = "static"
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    # Fallback for development when static folder doesn't exist
    @app.get("/")
    async def root():
        """Root endpoint serving the React frontend."""
        return {
            "message": "Welcome to ShantiView API",
            "version": "2.0.0",
            "docs": "/api/docs",
            "features": [
                "LangGraph Multi-Agent LLM System",
                "Parallel LLM Execution",
                "Facial Emotion Recognition",
                "Voice Emotion Recognition", 
                "Wellness Questionnaire Analysis",
                "Location-based Suggestions",
                "AI Chat Assistant"
            ]
        }

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)