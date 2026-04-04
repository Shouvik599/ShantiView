"""
LangGraph Multi-Agent System for ShantiView
Implements parallel execution of LLM calls for faster responses
"""

import os
import time
import json
import logging
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, END
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the LLM client
def get_llm(temperature=0.2, top_p=0.7, max_tokens=1024):
    """Create a ChatNVIDIA instance"""
    return ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ============================================================
# STATE DEFINITIONS
# ============================================================

class AgentState(TypedDict):
    """Global state for the multi-agent system"""
    # User inputs
    user_name: str
    user_city: str
    
    # Questionnaire data
    questionnaire_data: Optional[Dict[str, Any]]
    questionnaire_analysis: Optional[Dict[str, Any]]
    
    # Emotion data
    facial_emotions_list: List[Dict[str, Any]]
    vocal_emotion: Dict[str, Any]
    combined_analysis: Optional[Dict[str, Any]]
    
    # Wellness stats
    wellness_stats: Optional[List[Dict[str, Any]]]
    
    # News
    news_data: Optional[List[Dict[str, Any]]]
    
    # Suggestions
    location: Optional[str]
    suggestions: Optional[List[Dict[str, Any]]]
    
    # Chat
    chat_message: Optional[str]
    chat_response: Optional[str]
    
    # Task routing
    task_type: str  # questionnaire, wellness_stats, news, suggestions, combined_analysis, chat
    
    # Results
    final_response: Optional[Dict[str, Any]]
    errors: List[str]


# ============================================================
# SPECIALIZED AGENT NODES
# ============================================================

class QuestionnaireAgent:
    """Analyzes questionnaire responses using LLM"""
    
    @staticmethod
    async def analyze(state: AgentState) -> AgentState:
        if not state.get("questionnaire_data"):
            state["errors"] = state.get("errors", []) + ["No questionnaire data provided"]
            return state
            
        try:
            llm = get_llm(temperature=0.3, max_tokens=1500)
            data = state["questionnaire_data"]
            
            prompt = f"""
            Based on the following emotional and mental well-being data from a user questionnaire, provide a direct, empathetic, and encouraging feedback response in JSON format. The JSON object should have two keys: "summary" and "suggestions". The "summary" should be a string that summarizes the user's current state based on their ratings and feelings. The "suggestions" should be an array of strings, with each string containing an actionable and personalized suggestion for improvement.
            User Data:
            - Current Stress Level: {data.get('stressLevel', 'N/A')} out of 10
            - Overall Mood: {data.get('moodLevel', 'N/A')} out of 10
            - Energy Levels: {data.get('energyLevel', 'N/A')} out of 10
            - Feeling in one word: {data.get('feelingWord', 'N/A')}
            - Hours of Sleep: {data.get('sleepHours', 'N/A')}
            - Sleep Quality: {data.get('sleepQuality', 'N/A')}
            - Social Connection: {data.get('socialConnection', 'N/A')} out of 5
            - Physical Activity Today: {data.get('physicalActivity', 'N/A')}
            - Post-exercise Energy: {data.get('postExerciseEnergy', 'N/A')} out of 5 (if applicable)
            - Workload Stress: {data.get('workloadStress', 'N/A')} out of 5
            - Work-Life Balance: {data.get('workLifeBalance', 'N/A')} out of 5
            - Manager Support: {data.get('managerSupport', 'N/A')}
            - Biggest Emotional Challenge at Work: {data.get('corporateFeedback', 'N/A')}
            
            Return only a valid JSON object with "summary" and "suggestions" keys.
            """
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content
            
            if response_text.strip().startswith('```json'):
                response_text = response_text.strip()[7:-3].strip()
            
            analysis_dict = json.loads(response_text)
            state["questionnaire_analysis"] = analysis_dict
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM response: {e}")
            state["errors"] = state.get("errors", []) + ["Invalid analysis format received from LLM"]
        except Exception as e:
            logger.error(f"Error during questionnaire analysis: {e}")
            state["errors"] = state.get("errors", []) + [str(e)]
            
        return state


class WellnessStatsAgent:
    """Generates wellness statistics using LLM"""
    
    @staticmethod
    async def generate_stats(state: AgentState) -> AgentState:
        fallback_data = [
            {"title": "Engagement", "value": "85%", "description": "Employee engagement remains high.", "color": "text-teal-400"},
            {"title": "Stress Levels", "value": "15%", "description": "Average stress levels have decreased this week.", "color": "text-blue-400"},
            {"title": "Burnout Risk", "value": "7%", "description": "Percentage of employees at high risk of burnout.", "color": "text-purple-400"},
            {"title": "Positive Sentiment", "value": "92%", "description": "Sentiment in team communications is positive.", "color": "text-amber-400"},
            {"title": "Wellness Sessions", "value": "4.8/5", "description": "Average rating for wellness workshops.", "color": "text-emerald-400"}
        ]
        
        state["wellness_stats"] = fallback_data  # Set fallback first
        
        try:
            llm = get_llm(temperature=0.5, max_tokens=1500)
            
            prompt = """
            Generate a list of 5 realistic but fictional daily corporate wellness statistics for a company dashboard.
            The statistics should be diverse, covering topics like engagement, stress, burnout, and positive sentiment.
            Provide the output as a valid JSON array of objects. Each object should have the following keys:
            - "title": The name of the metric (e.g., "Engagement", "Stress Levels").
            - "value": The main statistic value as a string (e.g., "85%", "15%", "7"). Should be a number or percentage or a rating.
            - "description": A short, one-sentence description of the statistic.
            - "color": A Tailwind CSS text color class (e.g., "text-teal-400", "text-blue-400", "text-purple-400", "text-amber-400", "text-emerald-400").

            Return only the JSON array, without any markdown formatting.
            """
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content
            
            logger.info(f"Wellness stats LLM response: {response_text[:200]}")
            
            cleaned_response = response_text.strip().replace('```json', '').replace('```', '').strip()
            
            if cleaned_response:
                stats_data = json.loads(cleaned_response)
                state["wellness_stats"] = stats_data
                logger.info(f"Successfully parsed {len(stats_data)} wellness stats")
            else:
                logger.warning("Empty response from LLM, using fallback data")
            
        except Exception as e:
            logger.error(f"Error in wellness stats generation: {e}")
            logger.warning("Using fallback wellness stats data")
            state["errors"] = state.get("errors", []) + [str(e)]
            
        return state


class NewsAgent:
    """Generates wellness news using LLM"""
    
    @staticmethod
    async def generate_news(state: AgentState) -> AgentState:
        fallback_data = [
            {"title": "Mindfulness at Work: A Guide", "description": "Learn simple techniques to integrate mindfulness into your daily routine for a more focused and productive day."},
            {"title": "The Importance of Digital Detox", "description": "Why taking breaks from screens can improve your mental health, reduce stress, and improve real-life social connections."},
            {"title": "Building a Resilient Team", "description": "Expert tips on fostering a supportive and resilient work environment through open communication and adaptability."},
            {"title": "The Power of a Five-Minute Walk", "description": "Discover how short, brisk walks can boost creativity, reduce stress, and improve brain power throughout the day."},
            {"title": "Nutrition for Mental Clarity", "description": "How a balanced diet rich in healthy fats, whole grains, and leafy greens can have a profound impact on cognitive function."}
        ]
        
        state["news_data"] = fallback_data  # Set fallback first
        
        try:
            llm = get_llm(temperature=0.6, max_tokens=1000)
            
            prompt = """
            Generate a list of 5 short, positive wellness news headlines and descriptions for a company dashboard.
            For each news item, provide:
            - "title": The news headline.
            - "description": A very short, one-sentence summary of the article.
            Return only the JSON array, without any markdown formatting.
            """
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content
            
            logger.info(f"News LLM response: {response_text[:200]}")
            
            cleaned_response = response_text.strip().replace('```json', '').replace('```', '').strip()
            
            if cleaned_response:
                news_data = json.loads(cleaned_response)
                state["news_data"] = news_data
                logger.info(f"Successfully parsed {len(news_data)} news items")
            else:
                logger.warning("Empty response from LLM, using fallback data")
            
        except Exception as e:
            logger.error(f"Error in news generation: {e}")
            logger.warning("Using fallback news data")
            state["errors"] = state.get("errors", []) + [str(e)]
            
        return state


class CombinedAnalysisAgent:
    """Performs combined emotion analysis using LLM"""
    
    @staticmethod
    async def analyze(state: AgentState) -> AgentState:
        facial_results = state.get("facial_emotions_list", [])
        voice_result = state.get("vocal_emotion", {})
        user_name = state.get("user_name", "there")
        
        if not facial_results and not voice_result:
            state["errors"] = state.get("errors", []) + ["No facial or voice analysis data found"]
            return state
        
        try:
            llm = get_llm(temperature=0.3, max_tokens=1500)
            
            facial_summary = ", ".join([f"{res['emotion']} ({res['score'] * 100:.0f}%)" for res in facial_results]) if facial_results else "No facial data provided."
            voice_emotion = voice_result if isinstance(voice_result, str) else 'No voice data provided.'
            
            prompt = f"""
            As a compassionate wellness assistant named ShantiView, analyze the following emotional data for a user named {user_name} and provide a holistic, empathetic summary and actionable suggestions.

            **User's Emotional Data:**
            - **Facial Expressions Detected (chronological):** {facial_summary}
            - **Vocal Tone Emotion:** {voice_emotion}

            **Your Task:**
            Based on this combined data, generate a response in a valid JSON format with two keys: "summary" and "suggestions".

            1. **"summary" (string):** Write a concise, empathetic paragraph (3-4 sentences) that synthesizes the data. Acknowledge the user's emotional state as indicated by both their facial expressions and voice. If there's a conflict (e.g., smiling face but stressed voice), gently point it out as a sign of potential emotional masking.
            2. **"suggestions" (array of strings):** Provide 3 concise, empathetic actionable, positive, and personalized suggestions. These should be practical tips for improving well-being, tailored to the detected emotions.

            Please provide only the raw JSON object in your response.
            """
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content
            
            if response_text.strip().startswith('```json'):
                response_text = response_text.strip()[7:-3].strip()
            
            analysis_dict = json.loads(response_text)
            state["combined_analysis"] = analysis_dict
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM response: {e}")
            state["errors"] = state.get("errors", []) + ["Invalid analysis format received"]
        except Exception as e:
            logger.error(f"Error during combined analysis: {e}")
            state["errors"] = state.get("errors", []) + [str(e)]
            
        return state


class SuggestionsAgent:
    """Generates location-based suggestions using LLM"""
    
    @staticmethod
    async def generate(state: AgentState) -> AgentState:
        location = state.get("location")
        
        if not location:
            state["errors"] = state.get("errors", []) + ["No location provided"]
            return state
        
        try:
            llm = get_llm(temperature=0.4, max_tokens=2000)
            
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
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content
            
            cleaned_response = response_text.strip().replace('```json', '').replace('```', '').strip()
            suggestions_data = json.loads(cleaned_response)
            state["suggestions"] = suggestions_data
            
        except Exception as e:
            logger.error(f"Error in suggestions generation: {e}")
            state["errors"] = state.get("errors", []) + [str(e)]
            
        return state


class ChatAgent:
    """Handles chat conversations using LLM"""
    
    @staticmethod
    async def respond(state: AgentState) -> AgentState:
        user_message = state.get("chat_message")
        
        if not user_message:
            state["errors"] = state.get("errors", []) + ["No message provided"]
            return state
        
        try:
            llm = get_llm(temperature=0.6, max_tokens=500)
            
            prompt = f"You are ShantiView, a friendly and supportive wellness assistant. A user is talking to you. User's message: '{user_message}'. Your response should be helpful and focused on mental wellness, mindfulness, or providing a supportive ear. Be concise and empathetic."
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            state["chat_response"] = response.content
            
            logger.info(f"User message: '{user_message}'")
            logger.info(f"ShantiView response: '{state['chat_response']}'")
            
        except Exception as e:
            logger.error(f"Error in chat response: {e}")
            state["errors"] = state.get("errors", []) + [str(e)]
            
        return state




# ============================================================
# CACHING HELPERS
# ============================================================

_cached_wellness_stats: Optional[List[Dict[str, Any]]] = None
_cached_news: Optional[List[Dict[str, Any]]] = None
_cache_timestamp: float = 0
CACHE_DURATION = 6 * 60 * 60  # 6 hours


def _get_cached_wellness_stats() -> Optional[List[Dict[str, Any]]]:
    """Get cached wellness stats if still valid"""
    global _cached_wellness_stats, _cache_timestamp
    if _cached_wellness_stats and (time.time() - _cache_timestamp) < CACHE_DURATION:
        logger.info("Returning cached wellness stats")
        return _cached_wellness_stats
    return None


def _get_cached_news() -> Optional[List[Dict[str, Any]]]:
    """Get cached news if still valid"""
    global _cached_news, _cache_timestamp
    if _cached_news and (time.time() - _cache_timestamp) < CACHE_DURATION:
        logger.info("Returning cached news")
        return _cached_news
    return None


def _set_cache(stats: Optional[List] = None, news: Optional[List] = None):
    """Update the cache with new data"""
    global _cached_wellness_stats, _cached_news, _cache_timestamp
    if stats is not None:
        _cached_wellness_stats = stats
    if news is not None:
        _cached_news = news
    _cache_timestamp = time.time()
    logger.info(f"Cache updated. TTL: {CACHE_DURATION} seconds ({CACHE_DURATION // 3600} hours)")


# ============================================================
# PARALLEL EXECUTABLE FUNCTIONS
# ============================================================

async def run_questionnaire_analysis(questionnaire_data: Dict[str, Any], user_name: str = "User") -> Dict[str, Any]:
    """Run questionnaire analysis agent"""
    state: AgentState = {
        "user_name": user_name,
        "user_city": "",
        "questionnaire_data": questionnaire_data,
        "questionnaire_analysis": None,
        "facial_emotions_list": [],
        "vocal_emotion": {},
        "combined_analysis": None,
        "wellness_stats": None,
        "news_data": None,
        "location": None,
        "suggestions": None,
        "chat_message": None,
        "chat_response": None,
        "task_type": "questionnaire",
        "final_response": None,
        "errors": []
    }
    
    result_state = await questionnaire_agent.invoke(state)
    return {
        "status": "success" if not result_state.get("errors") else "error",
        "analysis_and_suggestions": result_state.get("questionnaire_analysis"),
        "errors": result_state.get("errors", [])
    }


async def run_combined_analysis(facial_emotions_list: List[Dict], vocal_emotion: Dict, user_name: str = "User") -> Dict[str, Any]:
    """Run combined analysis agent"""
    state: AgentState = {
        "user_name": user_name,
        "user_city": "",
        "questionnaire_data": None,
        "questionnaire_analysis": None,
        "facial_emotions_list": facial_emotions_list,
        "vocal_emotion": vocal_emotion,
        "combined_analysis": None,
        "wellness_stats": None,
        "news_data": None,
        "location": None,
        "suggestions": None,
        "chat_message": None,
        "chat_response": None,
        "task_type": "combined_analysis",
        "final_response": None,
        "errors": []
    }
    
    result_state = await combined_analysis_agent.invoke(state)
    return {
        "status": "success" if not result_state.get("errors") else "error",
        "analysis": result_state.get("combined_analysis"),
        "errors": result_state.get("errors", [])
    }


async def run_wellness_stats() -> Dict[str, Any]:
    """Run wellness stats generation with caching"""
    # Check cache first
    cached = _get_cached_wellness_stats()
    if cached is not None:
        return {"status": "success", "data": cached, "cached": True}
    
    state: AgentState = {
        "user_name": "",
        "user_city": "",
        "questionnaire_data": None,
        "questionnaire_analysis": None,
        "facial_emotions_list": [],
        "vocal_emotion": {},
        "combined_analysis": None,
        "wellness_stats": None,
        "news_data": None,
        "location": None,
        "suggestions": None,
        "chat_message": None,
        "chat_response": None,
        "task_type": "wellness_stats",
        "final_response": None,
        "errors": []
    }
    
    result_state = await wellness_stats_agent.invoke(state)
    stats_data = result_state.get("wellness_stats", [])
    
    # Update cache
    _set_cache(stats=stats_data)
    
    return {
        "status": "success" if not result_state.get("errors") else "error",
        "data": stats_data,
        "errors": result_state.get("errors", [])
    }


async def run_news_snapshot() -> Dict[str, Any]:
    """Run news generation with caching"""
    # Check cache first
    cached = _get_cached_news()
    if cached is not None:
        return {"status": "success", "data": cached, "cached": True}
    
    state: AgentState = {
        "user_name": "",
        "user_city": "",
        "questionnaire_data": None,
        "questionnaire_analysis": None,
        "facial_emotions_list": [],
        "vocal_emotion": {},
        "combined_analysis": None,
        "wellness_stats": None,
        "news_data": None,
        "location": None,
        "suggestions": None,
        "chat_message": None,
        "chat_response": None,
        "task_type": "news",
        "final_response": None,
        "errors": []
    }
    
    result_state = await news_agent.invoke(state)
    news_data = result_state.get("news_data", [])
    
    # Update cache
    _set_cache(news=news_data)
    
    return {
        "status": "success" if not result_state.get("errors") else "error",
        "data": news_data,
        "errors": result_state.get("errors", [])
    }


async def run_suggestions(location: str) -> Dict[str, Any]:
    """Run suggestions generation"""
    state: AgentState = {
        "user_name": "",
        "user_city": "",
        "questionnaire_data": None,
        "questionnaire_analysis": None,
        "facial_emotions_list": [],
        "vocal_emotion": {},
        "combined_analysis": None,
        "wellness_stats": None,
        "news_data": None,
        "location": location,
        "suggestions": None,
        "chat_message": None,
        "chat_response": None,
        "task_type": "suggestions",
        "final_response": None,
        "errors": []
    }
    
    result_state = await suggestions_agent.invoke(state)
    return {
        "status": "success" if not result_state.get("errors") else "error",
        "data": result_state.get("suggestions", []),
        "errors": result_state.get("errors", [])
    }


async def run_chat(message: str) -> Dict[str, Any]:
    """Run chat agent"""
    state: AgentState = {
        "user_name": "",
        "user_city": "",
        "questionnaire_data": None,
        "questionnaire_analysis": None,
        "facial_emotions_list": [],
        "vocal_emotion": {},
        "combined_analysis": None,
        "wellness_stats": None,
        "news_data": None,
        "location": None,
        "suggestions": None,
        "chat_message": message,
        "chat_response": None,
        "task_type": "chat",
        "final_response": None,
        "errors": []
    }
    
    result_state = await chat_agent.invoke(state)
    return {
        "status": "success" if not result_state.get("errors") else "error",
        "response": result_state.get("chat_response"),
        "errors": result_state.get("errors", [])
    }


async def run_parallel_wellness_dashboard() -> Dict[str, Any]:
    """Run wellness stats and news generation in parallel for faster dashboard loading"""
    stats_task = asyncio.create_task(run_wellness_stats())
    news_task = asyncio.create_task(run_news_snapshot())
    
    stats_result, news_result = await asyncio.gather(stats_task, news_task, return_exceptions=True)
    
    # Handle any exceptions from the tasks
    if isinstance(stats_result, Exception):
        stats_result = {"status": "error", "data": [], "errors": [str(stats_result)]}
    if isinstance(news_result, Exception):
        news_result = {"status": "error", "data": [], "errors": [str(news_result)]}
    
    return {
        "wellness_stats": stats_result.get("data", []),
        "news": news_result.get("data", [])
    }


# ============================================================
# DIRECT ASYNC INVOKER (bypasses LangGraph graph issues)
# ============================================================

class AsyncAgentInvoker:
    """Wraps an agent class method and invokes it directly without LangGraph"""
    
    def __init__(self, agent_class, method_name: str):
        self.agent_class = agent_class
        self.method_name = method_name
    
    async def invoke(self, state: AgentState) -> AgentState:
        """Invoke the agent's static async method directly"""
        method = getattr(self.agent_class, self.method_name, None)
        if method:
            return await method(state)
        return state


# Create agent invokers (direct invocation, bypasses LangGraph compilation issues)
questionnaire_agent = AsyncAgentInvoker(QuestionnaireAgent, "analyze")
wellness_stats_agent = AsyncAgentInvoker(WellnessStatsAgent, "generate_stats")
news_agent = AsyncAgentInvoker(NewsAgent, "generate_news")
combined_analysis_agent = AsyncAgentInvoker(CombinedAnalysisAgent, "analyze")
suggestions_agent = AsyncAgentInvoker(SuggestionsAgent, "generate")
chat_agent = AsyncAgentInvoker(ChatAgent, "respond")
