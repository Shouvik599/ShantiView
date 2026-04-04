import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowLeft, ArrowRight } from 'lucide-react'
import { API_URL } from '../config'

function Questionnaire() {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({
    stressLevel: 5,
    moodLevel: 5,
    energyLevel: 5,
    feelingWord: '',
    sleepHours: 7,
    sleepQuality: 'moderate',
    socialConnection: 3,
    physicalActivity: 'none',
    postExerciseEnergy: 3,
    workloadStress: 3,
    workLifeBalance: 3,
    managerSupport: 'neutral',
    corporateFeedback: ''
  })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/api/analyze_questionnaire`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })

      if (response.ok) {
        const data = await response.json()
        localStorage.setItem('questionnaireResults', JSON.stringify(data))
        navigate('/results', { state: { questionnaireAnalysis: data } })
      } else {
        const errorData = await response.json()
        setError(errorData.error || 'Failed to analyze questionnaire')
      }
    } catch (err) {
      console.error('Error submitting questionnaire:', err)
      setError('Failed to submit questionnaire. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const RatingSlider = ({ name, label, min = 0, max = 10, description }) => (
    <div className="mb-6">
      <label className="block text-sm font-medium text-gray-300 mb-2">
        {label}
      </label>
      <div className="flex items-center gap-4">
        <input
          type="range"
          name={name}
          min={min}
          max={max}
          value={formData[name]}
          onChange={handleChange}
          className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-purple-500"
        />
        <span className="text-xl font-bold text-purple-400 w-12 text-center">
          {formData[name]}
        </span>
      </div>
      {description && <p className="text-xs text-gray-500 mt-1">{description}</p>}
    </div>
  )

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 md:p-8">
      <header className="flex items-center justify-center mb-8">
        <Link to="/" className="mr-4 text-gray-400 hover:text-white transition-colors">
          <ArrowLeft className="h-6 w-6" />
        </Link>
        <h1 className="text-3xl md:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-purple-500">
          Wellness Questionnaire
        </h1>
      </header>

      <form onSubmit={handleSubmit} className="max-w-2xl w-full bg-gray-800 rounded-2xl shadow-xl p-8">
        <h2 className="text-xl font-semibold text-white mb-6">Tell us about your wellbeing</h2>

        {error && (
          <div className="bg-red-900/50 border border-red-500 rounded-xl p-4 mb-6">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        <div className="space-y-6">
          <RatingSlider name="stressLevel" label="Stress Level" description="0 = No stress, 10 = Extremely stressed" />
          <RatingSlider name="moodLevel" label="Overall Mood" description="0 = Very low, 10 = Excellent" />
          <RatingSlider name="energyLevel" label="Energy Levels" description="0 = Exhausted, 10 = Highly energetic" />

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">How are you feeling in one word?</label>
            <input
              type="text"
              name="feelingWord"
              value={formData.feelingWord}
              onChange={handleChange}
              placeholder="e.g., tired, happy, anxious"
              className="w-full px-4 py-3 rounded-xl bg-gray-700 border border-gray-600 text-white placeholder-gray-500 focus:outline-none input-glow transition-all"
            />
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">Hours of Sleep</label>
            <input
              type="number"
              name="sleepHours"
              value={formData.sleepHours}
              onChange={handleChange}
              min={0}
              max={24}
              step={0.5}
              className="w-full px-4 py-3 rounded-xl bg-gray-700 border border-gray-600 text-white focus:outline-none input-glow transition-all"
            />
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">Sleep Quality</label>
            <select
              name="sleepQuality"
              value={formData.sleepQuality}
              onChange={handleChange}
              className="w-full px-4 py-3 rounded-xl bg-gray-700 border border-gray-600 text-white focus:outline-none input-glow transition-all"
            >
              <option value="poor">Poor</option>
              <option value="fair">Fair</option>
              <option value="moderate">Moderate</option>
              <option value="good">Good</option>
              <option value="excellent">Excellent</option>
            </select>
          </div>

          <RatingSlider name="socialConnection" label="Social Connection" min={1} max={5} description="1 = Very isolated, 5 = Very connected" />

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">Physical Activity Today</label>
            <select
              name="physicalActivity"
              value={formData.physicalActivity}
              onChange={handleChange}
              className="w-full px-4 py-3 rounded-xl bg-gray-700 border border-gray-600 text-white focus:outline-none input-glow transition-all"
            >
              <option value="none">None</option>
              <option value="light">Light (walking, stretching)</option>
              <option value="moderate">Moderate (jogging, cycling)</option>
              <option value="intense">Intense (gym, running, sports)</option>
            </select>
          </div>

          <RatingSlider name="workloadStress" label="Workload Stress" min={1} max={5} description="1 = Very manageable, 5 = Overwhelming" />
          <RatingSlider name="workLifeBalance" label="Work-Life Balance" min={1} max={5} description="1 = Poor balance, 5 = Excellent balance" />

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">Manager Support</label>
            <select
              name="managerSupport"
              value={formData.managerSupport}
              onChange={handleChange}
              className="w-full px-4 py-3 rounded-xl bg-gray-700 border border-gray-600 text-white focus:outline-none input-glow transition-all"
            >
              <option value="very_unsupportive">Very Unsupportive</option>
              <option value="unsupportive">Unsupportive</option>
              <option value="neutral">Neutral</option>
              <option value="supportive">Supportive</option>
              <option value="very_supportive">Very Supportive</option>
            </select>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">Biggest Emotional Challenge at Work</label>
            <textarea
              name="corporateFeedback"
              value={formData.corporateFeedback}
              onChange={handleChange}
              placeholder="Share what's been on your mind..."
              rows={4}
              className="w-full px-4 py-3 rounded-xl bg-gray-700 border border-gray-600 text-white placeholder-gray-500 focus:outline-none input-glow transition-all resize-none"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full btn-gradient px-8 py-3 rounded-xl text-white font-bold text-lg disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {isLoading ? 'Analyzing...' : (
            <>
              Get My Wellness Report
              <ArrowRight className="h-5 w-5" />
            </>
          )}
        </button>
      </form>
    </div>
  )
}

export default Questionnaire