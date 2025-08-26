import React, { useState } from 'react';
import './App.css';
import AdminDashboard from './components/AdminDashboard';
import CustomerDashboard from './components/CustomerDashboard';
import LoginForm from './components/LoginForm';

type UserRole = 'admin' | 'customer' | null;
type ViewMode = 'landing' | 'login' | 'dashboard';

interface User {
  role: UserRole;
  customerId?: string;
  apiKey?: string;
  name?: string;
}

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('landing');

  const handleLogin = async (credentials: { role: UserRole; apiKey?: string; password?: string }) => {
    setIsLoading(true);
    
    try {
      if (credentials.role === 'admin') {
        // Mock admin login
        if (credentials.password === 'admin123') {
          setUser({ role: 'admin', name: 'Administrator' });
          setViewMode('dashboard');
        } else {
          throw new Error('Invalid admin password');
        }
      } else if (credentials.role === 'customer' && credentials.password) {
        // Customer password login
        const response = await fetch('/api/customer/login', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ password: credentials.password })
        });
        
        if (response.ok) {
          const data = await response.json();
          setUser({ 
            role: 'customer', 
            customerId: data.customer.id,
            name: data.customer.label 
          });
          setViewMode('dashboard');
        } else {
          throw new Error('Invalid password');
        }
      }
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    setUser(null);
    setViewMode('landing');
  };

  if (viewMode === 'login') {
    return <LoginForm onLogin={handleLogin} isLoading={isLoading} onBack={() => setViewMode('landing')} />;
  }

  if (viewMode === 'dashboard' && user) {
    return (
      <div className="App">
        <header className="dashboard-header">
          <div className="header-content">
            <div className="logo" onClick={() => setViewMode('landing')}>
              <span className="logo-text">Voxcentia</span>
            </div>
            <div className="user-info">
              <span>Welcome, {user.name}</span>
              <button onClick={handleLogout} className="logout-btn">Logout</button>
            </div>
          </div>
        </header>
        
        <main className="dashboard-main">
          {user.role === 'admin' ? (
            <AdminDashboard />
          ) : (
            <CustomerDashboard customerId={user.customerId!} />
          )}
        </main>
      </div>
    );
  }

  // Landing page
  return (
    <div className="App landing-page">
      {/* Navigation */}
      <nav className="main-nav">
        <div className="nav-content">
          <div className="logo">
            <span className="logo-text">Voxcentia</span>
          </div>
          <div className="nav-links">
            <a href="#features" className="nav-link">Features</a>
            <a href="#api" className="nav-link">API</a>
            <a href="#pricing" className="nav-link">Pricing</a>
            <button onClick={() => setViewMode('login')} className="btn btn-primary">
              Customer Login
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">
              The Future of 
              <span className="gradient-text"> Voice Intelligence</span>
            </h1>
            <p className="hero-description">
              Unlock emotions and personality insights from just 15 seconds of audio. 
              Our advanced AI analyzes voice patterns to reveal emotional states, 
              personality traits, and mental wellness indicators with unprecedented accuracy.
            </p>
            <div className="hero-actions">
              <button className="btn btn-primary btn-large">
                Start Free Trial
              </button>
              <button className="btn btn-secondary btn-large">
                View Documentation
              </button>
            </div>
            <div className="hero-stats">
              <div className="stat">
                <span className="stat-number">99.7%</span>
                <span className="stat-label">Accuracy</span>
              </div>
              <div className="stat">
                <span className="stat-number">15s</span>
                <span className="stat-label">Analysis Time</span>
              </div>
              <div className="stat">
                <span className="stat-number">26+</span>
                <span className="stat-label">Emotions Detected</span>
              </div>
              <div className="stat">
                <span className="stat-number">94+</span>
                <span className="stat-label">Personality Traits</span>
              </div>
            </div>
          </div>
          <div className="hero-visual">
            <div className="hero-image-container">
              <img 
                src="/voice-tech-illustration.png" 
                alt="Voice Analysis Technology Illustration" 
                className="hero-image"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features">
        <div className="section-content">
          <div className="section-header">
            <h2 className="section-title">Revolutionary Voice Analysis</h2>
            <p className="section-description">
              Built on decades of research and tested with over 11,000 case studies, 
              our proprietary voice resonance technology delivers insights that matter.
            </p>
            <div className="section-visual">
              <img 
                src="/emotion-analysis-viz.png" 
                alt="Emotion Analysis Data Visualization" 
                className="section-image"
              />
            </div>
          </div>
          
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">◉</div>
              <h3>Emotion Detection</h3>
              <p>Identify happiness, stress, confidence, fear, motivation, and 20+ other emotional states with clinical precision.</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">◈</div>
              <h3>Personality Analysis</h3>
              <p>Unlock 94 distinct personality traits using the Big Five model and advanced psychological profiling.</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">▲</div>
              <h3>Real-time Processing</h3>
              <p>Get instant results from just 15 seconds of audio. Perfect for live applications and user experiences.</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">◆</div>
              <h3>Enterprise Security</h3>
              <p>Bank-level encryption and HIPAA-compliant processing ensure your data stays protected.</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">◇</div>
              <h3>Easy Integration</h3>
              <p>Robust REST API with SDKs for popular languages. Get up and running in minutes, not weeks.</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">▣</div>
              <h3>Rich Analytics</h3>
              <p>Detailed insights, trending data, and customizable reports to understand your users better.</p>
            </div>
          </div>
        </div>
      </section>

      {/* API Section */}
      <section id="api" className="api-section">
        <div className="section-content">
          <div className="api-content">
            <div className="api-text">
              <h2 className="section-title">Simple, Powerful API</h2>
              <p className="section-description">
                Integrate voice intelligence into your applications with just a few lines of code. 
                Our RESTful API is designed for developers who value simplicity and reliability.
              </p>
              <div className="api-features">
                <div className="api-feature">
                  <span className="check-icon">●</span>
                  <span>99.9% uptime SLA</span>
                </div>
                <div className="api-feature">
                  <span className="check-icon">●</span>
                  <span>Global CDN for low latency</span>
                </div>
                <div className="api-feature">
                  <span className="check-icon">●</span>
                  <span>Comprehensive documentation</span>
                </div>
                <div className="api-feature">
                  <span className="check-icon">●</span>
                  <span>24/7 developer support</span>
                </div>
              </div>
            </div>
            <div className="api-demo">
              <div className="code-block">
                <div className="code-header">
                  <span className="code-title">Quick Start</span>
                </div>
                <pre className="code-content">
{`curl -X POST "https://api.voxcentia.com/v1/voice/analyze" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: multipart/form-data" \\
  -F "audio=@voice_sample.wav"

# Response
{
  "emotions": {
    "happiness": 0.85,
    "confidence": 0.72,
    "stress": 0.23
  },
  "personality": {
    "extroversion": 0.68,
    "openness": 0.91,
    "conscientiousness": 0.77
  }
}`}
                </pre>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-section">
            <div className="logo">
              <span className="logo-text">Voxcentia</span>
            </div>
            <p>The leader in voice intelligence technology</p>
          </div>
          <div className="footer-section">
            <h4>Product</h4>
            <a href="#features">Features</a>
            <a href="#api">API Documentation</a>
            <a href="#pricing">Pricing</a>
          </div>
          <div className="footer-section">
            <h4>Company</h4>
            <a href="#about">About Us</a>
            <a href="#careers">Careers</a>
            <a href="#contact">Contact</a>
          </div>
          <div className="footer-section">
            <h4>Support</h4>
            <a href="#docs">Documentation</a>
            <a href="#help">Help Center</a>
            <a href="#status">Status</a>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2025 Voxcentia. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;