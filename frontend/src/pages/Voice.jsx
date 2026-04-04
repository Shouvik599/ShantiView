import { useState, useRef, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { Mic, Square, ArrowLeft, ArrowRight, Upload } from 'lucide-react'
import { API_URL } from '../config'

function Voice() {
  const [isRecording, setIsRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState(null)
  const [emotion, setEmotion] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const fileInputRef = useRef(null)

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setAudioBlob(blob)
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
      setError(null)
    } catch (err) {
      console.error('Error accessing microphone:', err)
      setError('Unable to access microphone. Please check your permissions.')
    }
  }

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }, [])

  const analyzeAudio = async () => {
    if (!audioBlob) return
    
    setIsLoading(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('audio_file', audioBlob, 'recording.webm')

    try {
      const response = await fetch(`${API_URL}/predict_audio`, {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const data = await response.json()
        setEmotion(data.emotion)
      } else {
        const errorData = await response.json()
        setError(errorData.error || 'Failed to analyze audio')
      }
    } catch (err) {
      console.error('Error analyzing audio:', err)
      setError('Failed to analyze audio. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setIsLoading(true)
    setError(null)
    setEmotion(null)

    const formData = new FormData()
    formData.append('audio_file', file)

    try {
      const response = await fetch(`${API_URL}/predict_audio`, {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const data = await response.json()
        setEmotion(data.emotion)
      } else {
        const errorData = await response.json()
        setError(errorData.error || 'Failed to analyze audio')
      }
    } catch (err) {
      console.error('Error analyzing audio:', err)
      setError('Failed to analyze audio. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 md:p-8">
      <header className="flex items-center justify-center mb-8">
        <Link to="/" className="mr-4 text-gray-400 hover:text-white transition-colors">
          <ArrowLeft className="h-6 w-6" />
        </Link>
        <h1 className="text-3xl md:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-purple-500">
          Voice Emotion Analysis
        </h1>
      </header>

      <div className="max-w-2xl w-full">
        <div className="bg-gray-800 rounded-2xl shadow-xl p-8 text-center">
          <h2 className="text-xl font-semibold text-white mb-6">Record Your Voice</h2>

          {/* Recording Controls */}
          <div className="flex justify-center gap-4 mb-8">
            {!isRecording ? (
              <button
                onClick={startRecording}
                className="btn-gradient px-6 py-3 rounded-xl text-white font-bold flex items-center gap-2"
              >
                <Mic className="h-5 w-5" />
                Start Recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-xl text-white font-bold flex items-center gap-2 transition-colors"
              >
                <Square className="h-5 w-5" />
                Stop Recording
              </button>
            )}
          </div>

          {/* Audio Upload */}
          <div className="mb-8">
            <p className="text-gray-400 mb-4">Or upload an audio file</p>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-gray-700 hover:bg-gray-600 px-6 py-3 rounded-xl text-white font-bold flex items-center gap-2 mx-auto transition-colors"
            >
              <Upload className="h-5 w-5" />
              Upload Audio
            </button>
          </div>

          {/* Analyze Button */}
          {audioBlob && (
            <button
              onClick={analyzeAudio}
              disabled={isLoading}
              className="btn-gradient px-6 py-3 rounded-xl text-white font-bold mb-6 disabled:opacity-50"
            >
              {isLoading ? 'Analyzing...' : 'Analyze Emotion'}
            </button>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-red-900/50 border border-red-500 rounded-xl p-4 mb-6">
              <p className="text-red-400">{error}</p>
            </div>
          )}

          {/* Result */}
          {emotion && (
            <div className="bg-gray-700 rounded-xl p-6 mb-6">
              <p className="text-gray-300 mb-2">Detected Emotion</p>
              <p className="text-3xl font-bold text-purple-400">{emotion}</p>
            </div>
          )}

          {/* Status Indicator */}
          <div className="text-gray-400">
            {isRecording && (
              <div className="flex items-center justify-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                Recording...
              </div>
            )}
            {!audioBlob && !isRecording && (
              <p>Click "Start Recording" to begin</p>
            )}
          </div>
        </div>

        {/* Navigation */}
        <div className="mt-6 flex justify-end">
          <Link
            to="/questionnaire"
            state={{ voiceEmotion: emotion }}
            className="btn-gradient px-6 py-3 rounded-xl text-white font-bold flex items-center gap-2"
          >
            Continue to Questionnaire
            <ArrowRight className="h-5 w-5" />
          </Link>
        </div>
      </div>
    </div>
  )
}

export default Voice