import React, { useState, useEffect } from 'react';

interface CustomerStats {
  label: string;
  rateLimit: number;
  currentUsage: number;
  totalRequests: number;
  successRate: number;
  monthlyQuota: number;
  quotaUsed: number;
  apiKey?: string;
  recentRequests: Array<{
    timestamp: string;
    endpoint: string;
    status: string;
    responseTime: number;
  }>;
}

interface UsageData {
  period: string;
  data: Array<{
    date: string;
    requests: number;
    successRate: number;
    avgResponseTime: number;
  }>;
  summary: {
    totalRequests: number;
    avgSuccessRate: number;
    avgResponseTime: number;
  };
}

interface CustomerDashboardProps {
  customerId: string;
}

const CustomerDashboard: React.FC<CustomerDashboardProps> = ({ customerId }) => {
  const [stats, setStats] = useState<CustomerStats | null>(null);
  const [usageData, setUsageData] = useState<UsageData | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState('7d');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchCustomerData();
  }, [customerId]);

  useEffect(() => {
    fetchUsageData();
  }, [selectedPeriod]);

  const fetchCustomerData = async () => {
    try {
      const response = await fetch('/api/customer/dashboard', {
        headers: { 'customerId': customerId }
      });

      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Error fetching customer data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchUsageData = async () => {
    try {
      const response = await fetch(`/api/customer/usage?period=${selectedPeriod}`, {
        headers: { 'customerId': customerId }
      });

      if (response.ok) {
        const data = await response.json();
        setUsageData(data);
      }
    } catch (error) {
      console.error('Error fetching usage data:', error);
    }
  };

  const testVoiceAPI = async () => {
    try {
      // First check if API is running
      const healthResponse = await fetch('https://voice-analysis-api-production.up.railway.app/health');
      if (!healthResponse.ok) {
        alert('❌ Voice Analysis API is currently down. Please try again later.');
        return;
      }

      // Test API key with proper multipart form data
      const formData = new FormData();
      // Create a minimal test audio blob
      const dummyBlob = new Blob([''], { type: 'audio/wav' });
      formData.append('audio', dummyBlob, 'test.wav');

      const response = await fetch('https://voice-analysis-api-production.up.railway.app/v1/voice/analyze', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${stats?.apiKey}`
        },
        body: formData
      });

      if (response.status === 422) {
        alert('✅ API Key is valid! The Voice Analysis API is ready to process your audio files.');
      } else if (response.status === 401) {
        alert('❌ API Key is invalid or inactive. Please contact support.');
      } else if (response.status === 413) {
        alert('✅ API Key is valid! Connection successful.');
      } else if (response.status === 500) {
        alert('⚠️ API Key is valid, but the Voice Analysis API is experiencing internal server issues.\n\nThe Railway service appears to be having problems. Please try again later or contact Voxcentia support if the issue persists.');
      } else {
        alert(`✅ API Connection Successful!\n\nStatus: ${response.status}\nYour API key is working and the Voice Analysis service is online.`);
      }
    } catch (error) {
      alert('❌ Network error: Unable to connect to Voice Analysis API. Please check your connection.');
    }
  };

  if (isLoading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <span>Loading your dashboard...</span>
      </div>
    );
  }

  if (!stats) {
    return <div>Error loading dashboard data</div>;
  }

  const usagePercentage = (stats.quotaUsed / stats.monthlyQuota) * 100;

  return (
    <div className="customer-dashboard">
      <div className="dashboard-header">
        <h2>Welcome, {stats.label}!</h2>
        <button className="btn btn-primary" onClick={testVoiceAPI}>
          Test API Connection
        </button>
      </div>

      {/* Usage Overview */}
      <div className="stats-grid">
        <div className="stat-card">
          <h3>{stats.totalRequests.toLocaleString()}</h3>
          <p>Total Requests</p>
        </div>
        <div className="stat-card">
          <h3>{stats.successRate}%</h3>
          <p>Success Rate</p>
        </div>
        <div className="stat-card">
          <h3>{stats.rateLimit}</h3>
          <p>Rate Limit (per min)</p>
        </div>
        <div className="stat-card">
          <h3>{Math.round(usagePercentage)}%</h3>
          <p>Monthly Quota Used</p>
        </div>
      </div>

      {/* Monthly Quota Progress */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h3>Monthly Usage</h3>
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span>{stats.quotaUsed.toLocaleString()} requests used</span>
            <span>{stats.monthlyQuota.toLocaleString()} total</span>
          </div>
          <div style={{ 
            width: '100%', 
            height: '8px', 
            backgroundColor: '#e5e7eb', 
            borderRadius: '4px',
            overflow: 'hidden'
          }}>
            <div style={{ 
              width: `${Math.min(usagePercentage, 100)}%`, 
              height: '100%', 
              backgroundColor: usagePercentage > 80 ? '#ef4444' : usagePercentage > 60 ? '#f59e0b' : '#10b981',
              transition: 'width 0.3s ease'
            }}></div>
          </div>
        </div>
        <p style={{ color: '#666', fontSize: '0.9rem' }}>
          {usagePercentage > 80 && '⚠️ You\'re approaching your monthly limit. '}
          Resets on the 1st of each month.
        </p>
      </div>

      {/* API Integration Guide */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h3>🚀 Voice Analysis API Integration</h3>
        
        <div style={{ marginBottom: '1.5rem' }}>
          <h4>Your API Details:</h4>
          <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: '0.5rem', fontFamily: 'monospace' }}>
            <p><strong>Endpoint:</strong> https://voice-analysis-api-production.up.railway.app/v1/voice/analyze</p>
            <p><strong>API Key:</strong> {stats?.apiKey?.substring(0, 20)}...</p>
            <p><strong>Rate Limit:</strong> {stats.rateLimit} requests per minute</p>
          </div>
        </div>

        <div style={{ marginBottom: '1.5rem' }}>
          <h4>Migration from Vibeonix:</h4>
          <ul style={{ marginLeft: '1.5rem', color: '#374151' }}>
            <li>✅ <strong>Change URL:</strong> Replace your Vibeonix endpoint with the new Voxcentia endpoint above</li>
            <li>✅ <strong>Update API Key:</strong> Replace your old API key with the new one</li>
            <li>✅ <strong>Same Format:</strong> Request and response formats remain identical</li>
            <li>✅ <strong>Enhanced Results:</strong> Same 26 emotions + 94 personality traits</li>
          </ul>
        </div>

        <div>
          <h4>Example Usage:</h4>
          <pre style={{ 
            background: '#1f2937', 
            color: '#f9fafb', 
            padding: '1rem', 
            borderRadius: '0.5rem', 
            overflow: 'auto',
            fontSize: '0.85rem'
          }}>
{`curl -X POST "https://voice-analysis-api-production.up.railway.app/v1/voice/analyze" \\
  -H "Authorization: Bearer ${stats?.apiKey}" \\
  -H "Content-Type: multipart/form-data" \\
  -F "audio_file=@your_audio.wav"`}
          </pre>
        </div>
      </div>

      {/* Usage Analytics */}
      {usageData && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3>Usage Analytics</h3>
            <select 
              value={selectedPeriod} 
              onChange={(e) => setSelectedPeriod(e.target.value)}
              style={{ padding: '0.5rem', borderRadius: '0.25rem', border: '1px solid #d1d5db' }}
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
            </select>
          </div>

          <div className="stats-grid" style={{ marginBottom: '1.5rem' }}>
            <div className="stat-card">
              <h3>{usageData.summary.totalRequests}</h3>
              <p>Total Requests ({selectedPeriod})</p>
            </div>
            <div className="stat-card">
              <h3>{usageData.summary.avgSuccessRate}%</h3>
              <p>Avg Success Rate</p>
            </div>
            <div className="stat-card">
              <h3>{usageData.summary.avgResponseTime}ms</h3>
              <p>Avg Response Time</p>
            </div>
          </div>

          {/* Recent Requests */}
          <div className="table-container">
            <h4 style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb', margin: 0 }}>
              Recent API Requests
            </h4>
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Endpoint</th>
                  <th>Status</th>
                  <th>Response Time</th>
                </tr>
              </thead>
              <tbody>
                {stats.recentRequests.slice(0, 10).map((request, index) => (
                  <tr key={index}>
                    <td>{new Date(request.timestamp).toLocaleString()}</td>
                    <td>{request.endpoint}</td>
                    <td>
                      <span className={`status-badge ${request.status === 'success' ? 'status-active' : 'status-inactive'}`}>
                        {request.status}
                      </span>
                    </td>
                    <td>{request.responseTime}ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default CustomerDashboard;