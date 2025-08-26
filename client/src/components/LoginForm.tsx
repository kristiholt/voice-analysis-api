import React, { useState } from 'react';

interface LoginFormProps {
  onLogin: (credentials: { role: 'admin' | 'customer'; apiKey?: string; password?: string }) => Promise<void>;
  isLoading: boolean;
  onBack?: () => void;
}

const LoginForm: React.FC<LoginFormProps> = ({ onLogin, isLoading, onBack }) => {
  const [role, setRole] = useState<'admin' | 'customer'>('customer');
  const [apiKey, setApiKey] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (role === 'admin') {
      await onLogin({ role, password });
    } else {
      await onLogin({ role, password });
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        {onBack && (
          <button onClick={onBack} className="back-btn">
            ← Back to Home
          </button>
        )}
        <h2>🎙️ Voxcentia Portal</h2>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="role">Login as:</label>
            <select 
              id="role" 
              value={role} 
              onChange={(e) => setRole(e.target.value as 'admin' | 'customer')}
            >
              <option value="customer">Customer</option>
              <option value="admin">Administrator</option>
            </select>
          </div>

          {role === 'admin' ? (
            <div className="form-group">
              <label htmlFor="password">Admin Password:</label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter admin password"
                required
              />
              <small style={{ color: '#666', fontSize: '0.8rem' }}>
                Demo password: admin123
              </small>
            </div>
          ) : (
            <div className="form-group">
              <label htmlFor="password">Customer Password:</label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                required
              />
              <small style={{ color: '#666', fontSize: '0.8rem' }}>
                Use your assigned customer password
              </small>
            </div>
          )}

          <button 
            type="submit" 
            className="btn btn-primary" 
            disabled={isLoading}
            style={{ width: '100%', marginTop: '1rem' }}
          >
            {isLoading ? (
              <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                <div className="spinner"></div>
                Logging in...
              </span>
            ) : (
              'Login'
            )}
          </button>
        </form>

        <div style={{ marginTop: '2rem', textAlign: 'center', fontSize: '0.8rem', color: '#666' }}>
          <p><strong>Demo Customer Passwords:</strong></p>
          <p>Alpha Corp: alpha123</p>
          <p>Beta Ltd: beta123</p>
          <p>Gamma Inc: gamma123</p>
        </div>
      </div>
    </div>
  );
};

export default LoginForm;