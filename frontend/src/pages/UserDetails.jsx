import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

function UserDetails() {
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [city, setCity] = useState('')
  const [locationLoading, setLocationLoading] = useState(false)

  const getLocation = async () => {
    setLocationLoading(true)
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          try {
            const { latitude, longitude } = position.coords
            const response = await fetch(
              `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}`
            )
            const data = await response.json()
            const cityName = data.address?.city || data.address?.town || data.address?.village || 'Unknown City'
            setCity(cityName)
          } catch (error) {
            console.error('Failed to reverse geocode:', error)
          }
          setLocationLoading(false)
        },
        (error) => {
          console.error('Geolocation error:', error)
          setLocationLoading(false)
          alert('Unable to get your location. Please type it manually.')
        }
      )
    } else {
      alert('Geolocation is not supported by your browser.')
      setLocationLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    // Save to localStorage (works on all environments)
    localStorage.setItem('userName', name)
    localStorage.setItem('userCity', city)
    console.log('User details saved to localStorage:', { name, city })
    navigate('/facial')
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 md:p-8">
      <header className="flex items-center justify-center mb-12">
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

      <main className="text-center max-w-xl w-full p-8 bg-gray-800 rounded-3xl shadow-2xl">
        <h2 className="text-2xl md:text-3xl font-semibold mb-6 text-white">
          Let's Get Started
        </h2>
        <p className="text-base md:text-lg mb-8 text-gray-400 leading-relaxed">
          Please provide your name and location to begin your wellness journey.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6 text-left">
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-2">Your Full Name</label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="John Doe"
              required
              className="w-full px-4 py-3 rounded-xl bg-gray-700 border border-gray-600 text-white placeholder-gray-500 focus:outline-none input-glow transition-all duration-200"
            />
          </div>

          <div className="relative">
            <label htmlFor="city" className="block text-sm font-medium text-gray-300 mb-2">Select Your City</label>
            <button
              type="button"
              onClick={getLocation}
              disabled={locationLoading}
              className="w-full btn-gradient px-4 py-3 rounded-xl text-lg font-bold text-white shadow-lg transition-transform mb-4 disabled:opacity-50"
            >
              {locationLoading ? 'Fetching...' : 'Get My Current Location'}
            </button>
            <input
              type="text"
              id="city"
              value={city}
              onChange={(e) => setCity(e.target.value)}
              placeholder="Click the button above to get your city..."
              required
              className="w-full px-4 py-3 rounded-xl bg-gray-700 border border-gray-600 text-white placeholder-gray-500 focus:outline-none input-glow transition-all duration-200"
            />
          </div>

          <div className="pt-4 text-center">
            <button
              type="submit"
              className="w-full btn-gradient px-8 py-3 rounded-full text-lg font-bold text-white shadow-lg transition-transform"
            >
              Proceed
            </button>
          </div>
        </form>
      </main>
    </div>
  )
}

export default UserDetails