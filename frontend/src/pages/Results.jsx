import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { ArrowLeft, Lightbulb, Sparkles, MessageSquare } from 'lucide-react'

const API_URL = 'http://localhost:8000'

function Results() {
  const location = useLocation()
  const [analysis, setAnalysis] = useState(null)
  const [suggestions, setSuggestions] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const userCity = localStorage.getItem('userCity') || ''
  const userName = localStorage.getItem('userName') || 'User'

  useEffect(() => {
    const facialEmotions = location.state?.emotions || []
    const voiceEmotion = location.state?.voiceEmotion || null
    const questionnaireAnalysis = location.state?.questionnaireAnalysis || null

    if (questionnaireAnalysis) {
      setAnalysis(questionnaireAnalysis)
    } else if (facialEmotions.length > 0 || voiceEmotion) {
      fetchCombinedAnalysis(facialEmotions, voiceEmotion)
    }

    if (userCity) {
      fetchSuggestions()
    }
  }, [])

  const fetchCombinedAnalysis = async (facialEmotions, voiceEmotion) => {
    setIsLoading(true)
    try {
      const response = await fetch(`${API_URL}/api/combined-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          facial_emotions_list: facialEmotions,
          vocal_emotion: voiceEmotion || {},
          user_name: userName
        })
      })

      if (response.ok) {
        const data = await response.json()
        setAnalysis(data.analysis)
      }
    } catch (err) {
      console.error('Error fetching combined analysis:', err)
      setError('Failed to get combined analysis')
    } finally {
      setIsLoading(false)
    }
  }

  const fetchSuggestions = async () => {
    try {
      const response = await fetch(`${API_URL}/api/suggestions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ location: userCity })
      })

      if (response.ok) {
        const data = await response.json()
        setSuggestions(data)
      }
    } catch (err) {
      console.error('Error fetching suggestions:', err)
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-purple-500"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 md:p-8">
      <header className="flex items-center justify-center mb-8">
        <Link to="/" className="mr-4 text-gray-400 hover:text-white transition-colors">
          <ArrowLeft className="h-6 w-6" />
        </Link>
        <h1 className="text-3xl md:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-purple-500">
          Your Wellness Report
        </h1>
      </header>

      <div className="max-w-4xl w-full space-y-8">
        {/* Analysis Summary */}
        {(analysis?.summary || analysis?.analysis_and_suggestions?.summary) && (
          <div className="bg-gray-800 rounded-2xl shadow-xl p-8">
            <div className="flex items-center gap-3 mb-4">
              <Sparkles className="h-6 w-6 text-purple-400" />
              <h2 className="text-xl font-semibold text-white">Your Personal Summary</h2>
            </div>
            <p className="text-gray-300 text-lg leading-relaxed">
              {typeof analysis?.summary === 'string' ? analysis.summary : analysis?.analysis_and_suggestions?.summary}
            </p>
          </div>
        )}

        {/* Suggestions */}
        {(analysis?.suggestions || analysis?.analysis_and_suggestions?.suggestions) && (
          <div className="bg-gray-800 rounded-2xl shadow-xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <Lightbulb className="h-6 w-6 text-yellow-400" />
              <h2 className="text-xl font-semibold text-white">Personalized Suggestions</h2>
            </div>
            <ul className="space-y-4">
              {(analysis.suggestions || analysis?.analysis_and_suggestions?.suggestions || []).map((suggestion, index) => (
                <li key={index} className="flex items-start gap-3">
                  <span className="text-purple-400 mt-1">•</span>
                  <span className="text-gray-300">{suggestion}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Location-based Suggestions */}
        {suggestions.length > 0 && (
          <div className="bg-gray-800 rounded-2xl shadow-xl p-8">
            <h2 className="text-xl font-semibold text-white mb-6">Wellness Activities in {userCity}</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {suggestions.map((category, catIndex) => (
                <div key={catIndex} className="bg-gray-700 rounded-xl p-6">
                  <h3 className="text-lg font-bold text-purple-400 mb-4">{category.category}</h3>
                  <ul className="space-y-3">
                    {category.suggestions?.map((item, itemIndex) => (
                      <li key={itemIndex} className="text-sm">
                        <p className="text-white font-medium">{item.title}</p>
                        <p className="text-gray-400 text-xs mt-1">{item.description}</p>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-900/50 border border-red-500 rounded-xl p-6">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {!analysis && !isLoading && !error && (
          <div className="bg-gray-800 rounded-2xl shadow-xl p-12 text-center">
            <MessageSquare className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg">No analysis data available yet.</p>
            <p className="text-gray-500 mt-2">Complete the facial, voice, or questionnaire analysis to see your personalized report.</p>
          </div>
        )}

        {/* Navigation */}
        <div className="flex flex-wrap gap-4 justify-center">
          <Link to="/facial" className="bg-gray-700 hover:bg-gray-600 px-6 py-3 rounded-xl text-white font-bold transition-colors">
            Facial Analysis
          </Link>
          <Link to="/voice" className="bg-gray-700 hover:bg-gray-600 px-6 py-3 rounded-xl text-white font-bold transition-colors">
            Voice Analysis
          </Link>
          <Link to="/questionnaire" className="bg-gray-700 hover:bg-gray-600 px-6 py-3 rounded-xl text-white font-bold transition-colors">
            Questionnaire
          </Link>
        </div>
      </div>
    </div>
  )
}

export default Results