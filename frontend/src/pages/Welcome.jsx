import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { MessageCircle } from 'lucide-react'

function Welcome() {
  const [wellnessStats, setWellnessStats] = useState([])
  const [news, setNews] = useState([])
  const [statsLoading, setStatsLoading] = useState(true)
  const [newsLoading, setNewsLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const [statsRes, newsRes] = await Promise.all([
        fetch('/api/wellness-snapshot'),
        fetch('/api/news-snapshot')
      ])
      
      if (statsRes.ok) {
        const statsData = await statsRes.json()
        setWellnessStats(Array.isArray(statsData) ? statsData : [])
      }
      setStatsLoading(false)
      
      if (newsRes.ok) {
        const newsData = await newsRes.json()
        setNews(Array.isArray(newsData) ? newsData : [])
      }
      setNewsLoading(false)
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
      setStatsLoading(false)
      setNewsLoading(false)
    }
  }

  return (
    <div className="relative isolate">
      {/* Chatbot FAB */}
      <Link 
        to="/chatbot" 
        className="fixed bottom-6 right-6 bg-purple-600 hover:bg-purple-700 text-white p-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-110 z-50"
        title="Chat with Assistant"
      >
        <MessageCircle className="h-8 w-8" />
      </Link>

      {/* Background decorative blobs */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50vw] h-[50vw] bg-purple-600/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-[-20%] right-[-10%] w-[50vw] h-[50vw] bg-teal-500/20 rounded-full blur-3xl"></div>
      </div>

      {/* Main Container */}
      <div className="min-h-screen flex flex-col items-center justify-center p-4 md:p-8 gap-12 md:gap-16">
        {/* Header */}
        <header className="flex items-center justify-center">
          <svg className="h-10 w-10 text-purple-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 21.5c-4.4 0-8-3.6-8-8 0-4.4 3.6-8 8-8s8 3.6 8 8c0 4.4-3.6 8-8 8z" />
            <path d="M12 10a2 2 0 100 4 2 2 0 000-4z" />
            <path d="M16 16l-1 1" />
            <path d="M8 16l1 1" />
            <path d="M16 8l1-1" />
            <path d="M8 8l-1-1" />
          </svg>
          <h1 className="text-3xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 via-blue-500 to-purple-600 ml-4">
            ShantiView
          </h1>
        </header>

        {/* Welcome Section */}
        <main className="text-center max-w-4xl w-full">
          <h2 className="text-2xl md:text-4xl font-semibold mb-4 text-white">
            Empower Your Workforce, Elevate Your Culture.
          </h2>
          <p className="text-base md:text-lg mb-8 text-gray-400 leading-relaxed">
            ShantiView is the AI-powered assistant designed to support corporate mental and emotional wellness. Discover insights, promote healthy habits, and build a resilient team.
          </p>
          <Link 
            to="/details" 
            className="inline-block btn-gradient px-8 py-3 rounded-full text-lg font-bold text-white shadow-lg transition-transform"
          >
            Get Started
          </Link>
        </main>

        {/* Daily Corporate Statistics Section */}
        <section className="w-full max-w-6xl">
          <h3 className="text-xl md:text-2xl font-semibold mb-6 text-center text-white">
            Daily Corporate Wellness Snapshot
          </h3>
          <div className="flex overflow-x-auto no-scrollbar space-x-6 snap-x snap-mandatory py-2 px-1">
            {statsLoading ? (
              <div className="flex-none w-full flex justify-center items-center py-12">
                <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-purple-500"></div>
              </div>
            ) : (
              wellnessStats.map((stat, index) => (
                <div key={index} className="flex-none w-11/12 sm:w-1/2 md:w-1/3 lg:w-1/4 snap-center bg-gray-800 rounded-2xl shadow-xl p-6 md:p-8 flex flex-col justify-between">
                  <div>
                    <p className={`text-sm ${stat.color} font-medium uppercase mb-2`}>{stat.title}</p>
                    <p className="text-3xl md:text-4xl font-bold text-white">{stat.value}</p>
                  </div>
                  <p className="text-sm text-gray-400 mt-4">{stat.description}</p>
                </div>
              ))
            )}
          </div>
        </section>

        {/* Latest Wellness News Section */}
        <section className="w-full max-w-6xl">
          <h3 className="text-xl md:text-2xl font-semibold mb-6 text-center text-white">
            Latest Wellness News
          </h3>
          <div className="flex overflow-x-auto no-scrollbar space-x-6 snap-x snap-mandatory py-2 px-1">
            {newsLoading ? (
              <div className="flex-none w-full flex justify-center items-center py-12">
                <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-purple-500"></div>
              </div>
            ) : (
              news.map((article, index) => (
                <div key={index} className="flex-none w-11/12 sm:w-1/2 md:w-1/3 lg:w-1/4 snap-center bg-gradient-to-br from-gray-800 via-gray-900 to-gray-800 rounded-2xl shadow-xl transition-all duration-300 p-8 flex flex-col justify-between">
                  <h4 className="text-xl font-bold text-white mb-3 leading-tight">{article.title}</h4>
                  <p className="text-base text-gray-300 mb-2 flex-grow whitespace-normal break-words">{article.description}</p>
                </div>
              ))
            )}
          </div>
        </section>

        {/* Features Section */}
        <section className="w-full max-w-6xl">
          <h3 className="text-xl md:text-2xl font-semibold mb-6 text-center text-white">
            Explore Your Wellness Journey
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Facial Emotion Recognition Card */}
            <Link to="/facial" className="feature-card bg-gray-800 rounded-2xl shadow-xl p-6 md:p-8 flex flex-col items-center text-center transition-all duration-300">
              <span className="text-5xl mb-4">😀</span>
              <h4 className="text-lg md:text-xl font-bold text-white mb-2">Facial Emotion Recognition</h4>
              <p className="text-sm text-gray-400">
                Use your camera to understand your current emotional state.
              </p>
            </Link>

            {/* Voice Emotion Recognition Card */}
            <Link to="/voice" className="feature-card bg-gray-800 rounded-2xl shadow-xl p-6 md:p-8 flex flex-col items-center text-center transition-all duration-300">
              <span className="text-5xl mb-4">🗣️</span>
              <h4 className="text-lg md:text-xl font-bold text-white mb-2">Voice Emotion Recognition</h4>
              <p className="text-sm text-gray-400">
                Analyze your voice to get insights into your emotional tone.
              </p>
            </Link>

            {/* Wellness Questionnaire Card */}
            <Link to="/questionnaire" className="feature-card bg-gray-800 rounded-2xl shadow-xl p-6 md:p-8 flex flex-col items-center text-center transition-all duration-300">
              <span className="text-5xl mb-4">✍️</span>
              <h4 className="text-lg md:text-xl font-bold text-white mb-2">Wellness Questionnaire</h4>
              <p className="text-sm text-gray-400">
                Complete a quick questionnaire for a personalized wellness report.
              </p>
            </Link>
          </div>
        </section>
      </div>
    </div>
  )
}

export default Welcome