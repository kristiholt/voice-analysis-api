import React, { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [apiStatus, setApiStatus] = useState('checking...')

  useEffect(() => {
    // Check API health
    fetch('/health')
      .then(res => res.json())
      .then(data => setApiStatus(data.status || 'healthy'))
      .catch(() => setApiStatus('offline'))
  }, [])

  return (
    <div className="App">
      <header className="hero">
        <div className="container">
          <h1>🎙️ X Voice API v2.0</h1>
          <p className="subtitle">Advanced Voice Analysis by Voxcentia</p>
          <div className="status">
            API Status: <span className={`status-${apiStatus}`}>{apiStatus}</span>
          </div>
        </div>
      </header>

      <main className="container">
        <section className="features">
          <h2>🚀 Patent-Validated Features</h2>
          <div className="feature-grid">
            <div className="feature">
              <h3>38 Emotions</h3>
              <p>Binary classification with 80th percentile thresholds</p>
            </div>
            <div className="feature">
              <h3>Enneagram Profiling</h3>
              <p>9-archetype personality analysis system</p>
            </div>
            <div className="feature">
              <h3>Stress Response</h3>
              <p>Fight/Flight/Freeze pattern recognition</p>
            </div>
            <div className="feature">
              <h3>Wellbeing Classification</h3>
              <p>Struggling/OK/Thriving status system</p>
            </div>
            <div className="feature">
              <h3>Vibrational Analysis</h3>
              <p>Quantum physics-based frequency analysis</p>
            </div>
            <div className="feature">
              <h3>Meta-Emotions</h3>
              <p>0-300+ scale advanced emotion scoring</p>
            </div>
          </div>
        </section>

        <section className="api-info">
          <h2>📋 API Documentation</h2>
          <div className="api-section">
            <h3>Quick Test</h3>
            <p>Upload an audio file (WAV, MP3, FLAC, M4A, OGG, AAC) to get comprehensive voice analysis.</p>
            <div className="api-example">
              <code>
                POST /analyze<br/>
                Content-Type: multipart/form-data<br/>
                Authorization: Bearer YOUR_API_KEY<br/>
                Body: audio file
              </code>
            </div>
          </div>

          <div className="api-section">
            <h3>Response Features</h3>
            <ul>
              <li><strong>38 Emotions:</strong> emo1-emo38 with binary classification</li>
              <li><strong>94 Traits:</strong> char1-char94 personality characteristics</li>
              <li><strong>Probability Scores:</strong> 0-1 scale for standardized comparison</li>
              <li><strong>Enneagram Types:</strong> 9 personality archetypes</li>
              <li><strong>Stress Analysis:</strong> Fight/Flight/Freeze responses</li>
              <li><strong>Wellbeing Status:</strong> Overall classification system</li>
              <li><strong>AI Insights:</strong> Generated recommendations and observations</li>
            </ul>
          </div>
        </section>

        <section className="demo">
          <h2>🎯 Voice Analysis Demo</h2>
          <div className="demo-section">
            <p>Upload an audio file to test the enhanced X Voice API:</p>
            <input type="file" accept="audio/*" onChange={handleFileUpload} />
            <button className="analyze-btn">Analyze Voice</button>
          </div>
        </section>
      </main>

      <footer>
        <div className="container">
          <p>&copy; 2025 Voxcentia - Advanced Voice Analysis Technology</p>
        </div>
      </footer>
    </div>
  )

  function handleFileUpload(event) {
    const file = event.target.files[0]
    if (file) {
      console.log('File selected:', file.name)
      // Demo functionality - in real implementation would call the API
      alert('Demo: File selected - ' + file.name + '\\nIn production, this would call the X Voice API for analysis.')
    }
  }
}

export default App