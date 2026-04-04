import { useState, useRef, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { Camera, CameraOff, ArrowLeft, ArrowRight } from 'lucide-react'
import { API_URL } from '../config'

function Facial() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [emotions, setEmotions] = useState([])
  const [currentEmotion, setCurrentEmotion] = useState(null)
  const captureInterval = useRef(null)
  const streamRef = useRef(null)

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } 
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
      }
      setIsStreaming(true)
    } catch (err) {
      console.error('Error accessing camera:', err)
      alert('Unable to access camera. Please check your permissions.')
    }
  }

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
    if (captureInterval.current) {
      clearInterval(captureInterval.current)
      captureInterval.current = null
    }
    // Notify backend
    fetch(`${API_URL}/stop_video`, { method: 'POST' }).catch(console.error)
  }, [])

  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return
    
    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    context.drawImage(video, 0, 0)
    
    const imageData = canvas.toDataURL('image/jpeg', 0.8)
    
    try {
      const response = await fetch(`${API_URL}/api/analyze-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      })
      
      if (response.ok) {
        const data = await response.json()
        if (data.detected && data.emotion !== 'Uncertain') {
          setCurrentEmotion(data)
          setEmotions(prev => [...prev, data].slice(-20))
        }
      }
    } catch (err) {
      console.error('Error analyzing frame:', err)
    }
  }, [])

  useEffect(() => {
    if (isStreaming) {
      captureInterval.current = setInterval(captureAndAnalyze, 2000)
    }
    return () => {
      if (captureInterval.current) {
        clearInterval(captureInterval.current)
      }
    }
  }, [isStreaming, captureAndAnalyze])

  // Save emotions to localStorage when they change
  useEffect(() => {
    if (emotions.length > 0) {
      localStorage.setItem('facialEmotions', JSON.stringify(emotions))
    }
  }, [emotions])

  useEffect(() => {
    return () => stopCamera()
  }, [stopCamera])

  const emotionStats = emotions.length > 0 
    ? emotions.reduce((acc, curr) => {
        acc[curr.emotion] = (acc[curr.emotion] || 0) + 1
        return acc
      }, {})
    : {}
    
  const dominantEmotion = Object.entries(emotionStats).sort((a, b) => b[1] - a[1])[0]

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 md:p-8">
      <header className="flex items-center justify-center mb-8">
        <Link to="/" className="mr-4 text-gray-400 hover:text-white transition-colors">
          <ArrowLeft className="h-6 w-6" />
        </Link>
        <h1 className="text-3xl md:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-purple-500">
          Facial Emotion Analysis
        </h1>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl w-full">
        {/* Video Section */}
        <div className="bg-gray-800 rounded-2xl shadow-xl p-6">
          <div className="relative aspect-video bg-gray-900 rounded-xl overflow-hidden mb-4">
            <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" />
            <canvas ref={canvasRef} className="hidden" />
            {!isStreaming && (
              <div className="absolute inset-0 flex items-center justify-center text-gray-400">
                Click "Start Camera" to begin
              </div>
            )}
          </div>

          <div className="flex justify-center gap-4">
            {!isStreaming ? (
              <button
                onClick={startCamera}
                className="btn-gradient px-6 py-3 rounded-xl text-white font-bold flex items-center gap-2"
              >
                <Camera className="h-5 w-5" />
                Start Camera
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-xl text-white font-bold flex items-center gap-2 transition-colors"
              >
                <CameraOff className="h-5 w-5" />
                Stop Camera
              </button>
            )}
          </div>
        </div>

        {/* Results Section */}
        <div className="bg-gray-800 rounded-2xl shadow-xl p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Analysis Results</h2>
          
          {currentEmotion && (
            <div className="mb-4 p-4 bg-gray-700 rounded-xl">
              <p className="text-gray-300">Current Emotion</p>
              <p className="text-2xl font-bold text-purple-400">{currentEmotion.emotion}</p>
              <p className="text-sm text-gray-400">Confidence: {(currentEmotion.score * 100).toFixed(0)}%</p>
            </div>
          )}

          {dominantEmotion && (
            <div className="mb-4 p-4 bg-gray-700 rounded-xl">
              <p className="text-gray-300">Dominant Emotion</p>
              <p className="text-2xl font-bold text-teal-400">{dominantEmotion[0]}</p>
              <p className="text-sm text-gray-400">Detected {dominantEmotion[1]} times</p>
            </div>
          )}

          {emotionStats && Object.keys(emotionStats).length > 0 && (
            <div className="mt-4">
              <p className="text-gray-300 mb-2">All Emotions Detected</p>
              <div className="space-y-2">
                {Object.entries(emotionStats).sort((a, b) => b[1] - a[1]).map(([emotion, count]) => (
                  <div key={emotion} className="flex items-center gap-3">
                    <span className="text-sm text-gray-400 w-24">{emotion}</span>
                    <div className="flex-1 bg-gray-600 rounded-full h-2">
                      <div
                        className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${(count / emotions.length) * 100}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-400 w-12 text-right">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {emotions.length === 0 && (
            <p className="text-gray-400 text-center py-8">No emotions detected yet. Start the camera and look at the screen!</p>
          )}

          <div className="mt-6 flex justify-end">
            <Link
              to="/voice"
              state={{ emotions }}
              className="btn-gradient px-6 py-3 rounded-xl text-white font-bold flex items-center gap-2"
            >
              Continue to Voice Analysis
              <ArrowRight className="h-5 w-5" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Facial