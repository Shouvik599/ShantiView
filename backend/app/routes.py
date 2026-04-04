"""
FastAPI Routes for ShantiView API
"""

import os
import logging
import time
import base64
import tempfile
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.agents import (
    run_questionnaire_analysis,
    run_combined_analysis,
    run_wellness_stats,
    run_news_snapshot,
    run_suggestions,
    run_chat,
    run_parallel_wellness_dashboard
)
from app.facial_analysis import analyze_frame_api
from app.voice_analysis import predict_voice_emotion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Cache for wellness stats and news
wellness_stats_cache = {"data": None, "timestamp": 0}
news_cache = {"data": None, "timestamp": 0}
CACHE_DURATION = 60 * 60  # 1 hour


# ============================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# ============================================================

class QuestionnaireData(BaseModel):
    stressLevel: Optional[float] = None
    moodLevel: Optional[float] = None
    energyLevel: Optional[float] = None
    feelingWord: Optional[str] = None
    sleepHours: Optional[float] = None
    sleepQuality: Optional[str] = None
    socialConnection: Optional[float] = None
    physicalActivity: Optional[str] = None
    postExerciseEnergy: Optional[float] = None
    workloadStress: Optional[float] = None
    workLifeBalance: Optional[float] = None
    managerSupport: Optional[str] = None
    corporateFeedback: Optional[str] = None


class ChatMessage(BaseModel):
    message: str


class SuggestionRequest(BaseModel):
    location: str


class CombinedAnalysisRequest(BaseModel):
    facial_emotions_list: List[Dict[str, Any]]
    vocal_emotion: Dict[str, Any]
    user_name: Optional[str] = "User"


class FrameAnalysisResult(BaseModel):
    detected: bool
    emotion: str
    score: float


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@router.get("/api/health")
async def health_check():
    """Health check endpoint to verify app status."""
    return {
        "status": "ok",
        "agents": {
            "questionnaire": True,
            "wellness_stats": True,
            "news": True,
            "combined_analysis": True,
            "suggestions": True,
            "chat": True
        }
    }


# ============================================================
# WELLNESS DASHBOARD ENDPOINTS
# ============================================================

@router.get("/api/wellness-snapshot")
async def wellness_snapshot():
    """Get wellness statistics - uses parallel execution."""
    global wellness_stats_cache
    now = time.time()

    # Check cache first
    if wellness_stats_cache["data"] is not None and (now - wellness_stats_cache["timestamp"] < CACHE_DURATION):
        return wellness_stats_cache["data"]

    try:
        result = await run_parallel_wellness_dashboard()
        stats_data = result.get("wellness_stats", [])
        
        # Update cache
        wellness_stats_cache["data"] = stats_data
        wellness_stats_cache["timestamp"] = now
        
        return stats_data
        
    except Exception as e:
        logger.error(f"Error in wellness-snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/news-snapshot")
async def news_snapshot_endpoint():
    """Get wellness news snapshot - uses parallel execution."""
    global news_cache
    now = time.time()

    # Check cache first
    if news_cache["data"] is not None and (now - news_cache["timestamp"] < CACHE_DURATION):
        return news_cache["data"]

    try:
        result = await run_parallel_wellness_dashboard()
        news_data = result.get("news", [])
        
        # Update cache
        news_cache["data"] = news_data
        news_cache["timestamp"] = now
        
        return news_data
        
    except Exception as e:
        logger.error(f"Error in news-snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# QUESTIONNAIRE ENDPOINTS
# ============================================================

@router.post("/api/analyze_questionnaire")
async def analyze_questionnaire(data: QuestionnaireData):
    """Analyze questionnaire responses using LangGraph agent."""
    try:
        questionnaire_dict = data.model_dump()
        result = await run_questionnaire_analysis(
            questionnaire_data=questionnaire_dict,
            user_name="User"
        )
        return result
        
    except Exception as e:
        logger.error(f"Error during questionnaire analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# COMBINED EMOTION ANALYSIS ENDPOINT
# ============================================================

@router.post("/api/combined-analysis")
async def combined_analysis(request: CombinedAnalysisRequest):
    """Perform combined analysis of facial and voice emotions."""
    try:
        result = await run_combined_analysis(
            facial_emotions_list=request.facial_emotions_list,
            vocal_emotion=request.vocal_emotion,
            user_name=request.user_name
        )
        return result
        
    except Exception as e:
        logger.error(f"Error during combined analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# FACIAL EMOTION ANALYSIS ENDPOINTS
# ============================================================

class FrameAnalysisRequest(BaseModel):
    image: str  # base64 encoded image


@router.post("/api/analyze-frame")
async def analyze_frame(request: FrameAnalysisRequest):
    """
    Analyze a single frame for facial emotion detection.
    Accepts a base64 encoded image and returns the detected emotion.
    """
    try:
        # Decode the base64 image
        image_data = request.image.split(",")[1] if "," in request.image else request.image
        image_bytes = base64.b64decode(image_data)
        
        # Save to temp file for analysis
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        
        try:
            result = analyze_frame_api(temp_path)
            return result
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop_video")
async def stop_video():
    """
    Endpoint to signal stopping the video stream.
    Currently just returns acknowledgment.
    """
    return {"status": "stopped", "message": "Video stream stopped"}


# ============================================================
# AUDIO EMOTION PREDICTION ENDPOINTS
# ============================================================

@router.post("/predict_audio")
async def predict_audio(audio_file: UploadFile = File(...)):
    """
    Endpoint to receive an audio file, make a prediction using the MLP model.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Get emotion prediction
            result = await predict_voice_emotion(temp_path)
            return result
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error predicting audio emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# SUGGESTIONS ENDPOINT
# ============================================================

@router.post("/api/suggestions")
async def get_suggestions(request: SuggestionRequest):
    """Get location-based wellness suggestions."""
    try:
        result = await run_suggestions(location=request.location)
        return result.get("data", [])
        
    except Exception as e:
        logger.error(f"Error in suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CHAT ENDPOINTS
# ============================================================

@router.post("/api/chat")
async def chat_endpoint(request: ChatMessage):
    """Chat endpoint using LangGraph agent."""
    try:
        result = await run_chat(message=request.message)
        return {"response": result.get("response")}
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# USER DATA ENDPOINTS
# ============================================================

@router.post("/api/submit_data")
async def submit_data(request: Request):
    """Handle submission of user details."""
    data = await request.json()
    return {
        "status": "success",
        "message": "User data received",
        "data": data
    }


# ============================================================
# PARALLEL BATCH ENDPOINT
# ============================================================

@router.post("/api/batch-analysis")
async def batch_analysis(request: Request):
    """
    Run multiple analyses in parallel for faster responses.
    This endpoint demonstrates the power of parallel LLM execution.
    """
    try:
        body = await request.json()
        tasks = []
        
        # Queue up parallel tasks based on provided data
        if "questionnaire" in body:
            tasks.append(asyncio.create_task(
                run_questionnaire_analysis(body["questionnaire"], body.get("user_name", "User")),
                name="questionnaire"
            ))
        
        if "combined" in body:
            tasks.append(asyncio.create_task(
                run_combined_analysis(
                    body["combined"].get("facial_emotions_list", []),
                    body["combined"].get("vocal_emotion", {}),
                    body.get("user_name", "User")
                ),
                name="combined"
            ))
            
        if "suggestions" in body:
            tasks.append(asyncio.create_task(
                run_suggestions(body["suggestions"]["location"]),
                name="suggestions"
            ))
            
        if "chat" in body:
            tasks.append(asyncio.create_task(
                run_chat(body["chat"]["message"]),
                name="chat"
            ))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build response
        response = {}
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                response[task.get_name()] = {"status": "error", "error": str(result)}
            else:
                response[task.get_name()] = result
                
        return response
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))