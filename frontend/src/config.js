/**
 * API URL Configuration
 * Works in three environments:
 * 1. Local development with Vite (port 3000) - proxies all API calls to backend on port 7860
 * 2. Docker production (uses same origin since frontend is served by backend)
 * 3. Hugging Face Space (uses same origin on port 7860)
 */

// In Vite dev mode (localhost:3000), the Vite dev server proxies /api, /predict_audio, /stop_video
// to the backend on port 7860. So we can use empty string for relative paths.
const getApiUrl = () => {
  // Check if running in Vite dev mode (separate frontend/backend)
  const isDevMode = window.location.hostname === 'localhost' && window.location.port === '3000'
  
  if (isDevMode) {
    // Use empty string for relative paths when using Vite dev server
    // The Vite proxy will forward these to http://localhost:7860
    return ''
  }
  
  // For production (Docker/HuggingFace), use same origin (backend serves frontend)
  return window.location.origin
}

export const API_URL = getApiUrl()