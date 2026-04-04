import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Welcome from './pages/Welcome'
import UserDetails from './pages/UserDetails'
import Facial from './pages/Facial'
import Voice from './pages/Voice'
import Questionnaire from './pages/Questionnaire'
import Results from './pages/Results'
import Chatbot from './pages/Chatbot'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Welcome />} />
        <Route path="/details" element={<UserDetails />} />
        <Route path="/facial" element={<Facial />} />
        <Route path="/voice" element={<Voice />} />
        <Route path="/questionnaire" element={<Questionnaire />} />
        <Route path="/results" element={<Results />} />
        <Route path="/chatbot" element={<Chatbot />} />
      </Routes>
    </Router>
  )
}

export default App