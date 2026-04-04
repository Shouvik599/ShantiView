"""
ShantiView - AI Corporate Wellness Assistant
FastAPI Main Application with LangGraph Multi-Agent System
"""

import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

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
# ROOT ENDPOINT
# ============================================================

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