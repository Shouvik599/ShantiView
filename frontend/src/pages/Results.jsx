import { useState, useEffect } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { Sparkles, MessageSquare, Brain, Heart, Music, Utensils, Users, Repeat } from 'lucide-react'
import { API_URL } from '../config'

function Results() {
  const location = useLocation()
  const navigate = useNavigate()
  const [combinedAnalysis, setCombinedAnalysis] = useState(null)
  const [questionnaireAnalysis, setQuestionnaireAnalysis] = useState(null)
  const [regionSuggestions, setRegionSuggestions] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)
  const userCity = localStorage.getItem('userCity') || ''
  const userName = localStorage.getItem('userName') || 'User'

  useEffect(() => {
    // Retrieve data from route state or localStorage
    const facialEmotions = location.state?.emotions || JSON.parse(localStorage.getItem('facialEmotions') || '[]')
    const voiceEmotion = location.state?.voiceEmotion || JSON.parse(localStorage.getItem('voiceEmotion') || 'null')
    const questionnaireAnalysis = location.state?.questionnaireAnalysis || null
    // Check localStorage for cached data (pre-fetched from Questionnaire page)
    const cachedSuggestions = JSON.parse(localStorage.getItem('regionSuggestions') || 'null')
    const cachedCombinedAnalysis = JSON.parse(localStorage.getItem('combinedAnalysis') || 'null')

    if (questionnaireAnalysis) {
      setQuestionnaireAnalysis(questionnaireAnalysis)
    }

    // If we have cached combined analysis, use it immediately
    if (cachedCombinedAnalysis) {
      setCombinedAnalysis(cachedCombinedAnalysis)
    }

    // If we have cached suggestions, use them immediately
    if (cachedSuggestions) {
      setRegionSuggestions(cachedSuggestions)
      setIsLoading(false)
    }

    // Start all API calls in parallel immediately (in background) - only if not cached
    const promises = []

    if (facialEmotions.length > 0 || voiceEmotion) {
      // Only fetch if not cached
      if (!cachedCombinedAnalysis) {
        promises.push(fetchCombinedAnalysisParallel(facialEmotions, voiceEmotion))
      }
    }

    // Only fetch if we don't have cached suggestions
    if (userCity && !cachedSuggestions) {
      promises.push(fetchRegionSuggestionsParallel())
    }

    // Wait for parallel requests but don't block UI if we have cached data
    Promise.allSettled(promises).finally(() => {
      setIsLoading(false)
    })
  }, [])

  // Parallel version - fires immediately on component mount
  const fetchCombinedAnalysisParallel = async (facialEmotions, voiceEmotion) => {
    const emotionsList = Array.isArray(facialEmotions) ? facialEmotions : []
    const vocalEmotion = (typeof voiceEmotion === 'string' && voiceEmotion) 
      ? { emotion: voiceEmotion } 
      : {}
    
    console.log('Combined analysis request:', {
      facial_emotions_list: emotionsList,
      vocal_emotion: vocalEmotion,
      user_name: userName
    })

    try {
      const response = await fetch(`${API_URL}/api/combined-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          facial_emotions_list: emotionsList,
          vocal_emotion: vocalEmotion,
          user_name: userName
        })
      })

      const responseBody = await response.json()
      console.log('Combined analysis response:', response.status, responseBody)

      if (response.ok) {
        if (responseBody.analysis) {
          setCombinedAnalysis(responseBody.analysis)
        }
      }
    } catch (err) {
      console.error('Error fetching combined analysis:', err)
    }
  }

  // Parallel version - fires immediately when we have user city
  const fetchRegionSuggestionsParallel = async () => {
    try {
      const response = await fetch(`${API_URL}/api/suggestions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ location: userCity })
      })

      if (response.ok) {
        const data = await response.json()
        setRegionSuggestions(data)
      }
    } catch (err) {
      console.error('Error fetching region suggestions:', err)
    }
  }

  const startNewSession = () => {
    // Clear all stored data
    localStorage.removeItem('userName')
    localStorage.removeItem('userCity')
    localStorage.removeItem('facialEmotions')
    localStorage.removeItem('voiceEmotion')
    localStorage.removeItem('questionnaireResults')
    localStorage.removeItem('regionSuggestions')
    localStorage.removeItem('combinedAnalysis')
    navigate('/')
  }

  const getCategoryIcon = (category) => {
    const cat = category?.toLowerCase() || ''
    if (cat.includes('food')) return <Utensils className="h-5 w-5" />
    if (cat.includes('music')) return <Music className="h-5 w-5" />
    if (cat.includes('community')) return <Users className="h-5 w-5" />
    if (cat.includes('mindfulness')) return <Brain className="h-5 w-5" />
    return <Heart className="h-5 w-5" />
  }

  const getCategoryColor = (category) => {
    const cat = category?.toLowerCase() || ''
    if (cat.includes('food')) return 'text-orange-400 bg-orange-400/10'
    if (cat.includes('music')) return 'text-purple-400 bg-purple-400/10'
    if (cat.includes('community')) return 'text-blue-400 bg-blue-400/10'
    if (cat.includes('mindfulness')) return 'text-teal-400 bg-teal-400/10'
    return 'text-pink-400 bg-pink-400/10'
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-purple-500"></div>
      </div>
    )
  }

  const hasAnyData = combinedAnalysis || questionnaireAnalysis || regionSuggestions.length > 0

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 md:p-8">
      <header className="flex items-center justify-center mb-8">
        <Link to="/" className="mr-4 text-gray-400 hover:text-white transition-colors">
          <span className="sr-only">Home</span>
          <svg className="h-8 w-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 21.5c-4.4 0-8-3.6-8-8 0-4.4 3.6-8 8-8s8 3.6 8 8c0 4.4-3.6 8-8 8z" />
            <path d="M12 10a2 2 0 100 4 2 2 0 000-4z" />
          </svg>
        </Link>
        <h1 className="text-3xl md:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-purple-500">
          Your Wellness Report
        </h1>
      </header>

      <div className="max-w-5xl w-full space-y-8">
        {/* Welcome Header with Username */}
        {userName && userName !== 'User' && (
          <div className="bg-gradient-to-r from-purple-500/10 to-teal-500/10 rounded-2xl shadow-xl p-8 border border-purple-500/20">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-teal-500 flex items-center justify-center text-white text-xl font-bold">
                {userName.charAt(0).toUpperCase()}
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">Welcome, {userName}!</h2>
                <p className="text-gray-400 text-sm">Here's your personalized wellness report</p>
              </div>
            </div>
          </div>
        )}

        {/* Combined Emotion Analysis Section */}
        {(combinedAnalysis?.summary || combinedAnalysis?.analysis_and_suggestions?.summary) && (
          <div className="bg-gray-800 rounded-2xl shadow-xl p-8 border border-purple-500/20">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
                <Brain className="h-5 w-5 text-purple-400" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-white">Combined Emotion Analysis</h2>
                <p className="text-sm text-gray-400">Facial + Voice Emotion Insights</p>
              </div>
            </div>
            
            <div className="bg-gray-700/50 rounded-xl p-6 mb-6">
              <p className="text-gray-300 text-lg leading-relaxed">
                {typeof combinedAnalysis?.summary === 'string' 
                  ? combinedAnalysis.summary 
                  : combinedAnalysis?.analysis_and_suggestions?.summary}
              </p>
            </div>

            {combinedAnalysis?.suggestions?.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Personalized Suggestions</h3>
                <ul className="space-y-3">
                  {combinedAnalysis.suggestions.map((suggestion, index) => (
                    <li key={index} className="flex items-start gap-3 bg-gray-700/30 rounded-lg p-3">
                      <span className="text-purple-400 mt-1 flex-shrink-0">•</span>
                      <span className="text-gray-300">{suggestion}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Questionnaire Analysis Section */}
        {(questionnaireAnalysis?.summary || questionnaireAnalysis?.analysis_and_suggestions?.summary) && (
          <div className="bg-gray-800 rounded-2xl shadow-xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-teal-500/20 flex items-center justify-center">
                <Sparkles className="h-5 w-5 text-teal-400" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-white">Wellness Questionnaire Results</h2>
                <p className="text-sm text-gray-400">Your Mental Health Assessment</p>
              </div>
            </div>
            
            <div className="bg-gray-700/50 rounded-xl p-6 mb-6">
              <p className="text-gray-300 text-lg leading-relaxed">
                {typeof questionnaireAnalysis?.summary === 'string' 
                  ? questionnaireAnalysis.summary 
                  : questionnaireAnalysis?.analysis_and_suggestions?.summary}
              </p>
            </div>

            {questionnaireAnalysis?.suggestions?.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Wellness Suggestions</h3>
                <ul className="space-y-3">
                  {questionnaireAnalysis.suggestions.map((suggestion, index) => (
                    <li key={index} className="flex items-start gap-3 bg-gray-700/30 rounded-lg p-3">
                      <span className="text-teal-400 mt-1 flex-shrink-0">•</span>
                      <span className="text-gray-300">{suggestion}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Region-Based Suggestions Section */}
        {regionSuggestions.length > 0 && (
          <div className="bg-gray-800 rounded-2xl shadow-xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                <Heart className="h-5 w-5 text-blue-400" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-white">Wellness in {userCity}</h2>
                <p className="text-sm text-gray-400">Culture, Food, Music & Community</p>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {regionSuggestions.map((category, catIndex) => (
                <div key={catIndex} className="bg-gray-700/50 rounded-xl p-6 border border-gray-600/50">
                  <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg mb-4 ${getCategoryColor(category.category)}`}>
                    {getCategoryIcon(category.category)}
                    <h3 className="text-lg font-bold">{category.category}</h3>
                  </div>
                  <ul className="space-y-3">
                    {category.suggestions?.map((item, itemIndex) => (
                      <li key={itemIndex} className="bg-gray-800/50 rounded-lg p-3">
                        <p className="text-white font-medium text-sm">{item.title}</p>
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

        {/* No Data State */}
        {!hasAnyData && !isLoading && !error && (
          <div className="bg-gray-800 rounded-2xl shadow-xl p-12 text-center">
            <MessageSquare className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg">No analysis data available yet.</p>
            <p className="text-gray-500 mt-2">Complete the facial, voice, or questionnaire analysis to see your personalized report.</p>
          </div>
        )}

        {/* New Session Button */}
        <div className="flex justify-center pt-4">
          <button
            onClick={startNewSession}
            className="btn-gradient px-8 py-4 rounded-xl text-white font-bold text-lg flex items-center gap-3 shadow-lg transition-all duration-200 hover:scale-105"
          >
            <Repeat className="h-5 w-5" />
            Start New Session
          </button>
        </div>
      </div>
    </div>
  )
}

export default Results