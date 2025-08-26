import React, { useState, useEffect } from 'react';

interface Customer {
  id: string;
  label: string;
  isActive: boolean;
  rateLimit: number;
  createdAt: string;
  lastUsedAt?: string;
  totalRequests: number;
  successRate: number;
  monthlyUsage: number;
}

interface DashboardStats {
  totalCustomers: number;
  activeCustomers: number;
  totalRequests: number;
  avgResponseTime: number;
  recentActivity: Array<{
    date: string;
    requests: number;
  }>;
}

interface NewCustomerForm {
  label: string;
  rateLimit: number;
}

const AdminDashboard: React.FC = () => {
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showNewCustomerForm, setShowNewCustomerForm] = useState(false);
  const [newCustomer, setNewCustomer] = useState<NewCustomerForm>({ label: '', rateLimit: 1000 });
  const [selectedCustomer, setSelectedCustomer] = useState<Customer | null>(null);

  useEffect(() => {
    fetchCustomers();
    fetchDashboardStats();
  }, []);

  const fetchCustomers = async () => {
    try {
      const response = await fetch('/api/admin/customers', {
        headers: { 'Authorization': 'Bearer admin-secret' }
      });

      if (response.ok) {
        const data = await response.json();
        setCustomers(data.customers || []);
      }
    } catch (error) {
      console.error('Error fetching customers:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchDashboardStats = async () => {
    try {
      const response = await fetch('/api/admin/dashboard', {
        headers: { 'Authorization': 'Bearer admin-secret' }
      });

      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
    }
  };

  const createCustomer = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const response = await fetch('/api/admin/customers', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer admin-secret',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newCustomer)
      });

      if (response.ok) {
        const data = await response.json();
        alert(`✅ Customer created!\n\nAPI Key: ${data.apiKey}\n\nPlease save this key - it won't be shown again!`);
        setShowNewCustomerForm(false);
        setNewCustomer({ label: '', rateLimit: 1000 });
        fetchCustomers();
        fetchDashboardStats();
      } else {
        alert('Failed to create customer');
      }
    } catch (error) {
      console.error('Error creating customer:', error);
      alert('Error creating customer');
    }
  };

  const updateCustomer = async (customerId: string, updates: Partial<Customer>) => {
    try {
      const response = await fetch(`/api/admin/customers/${customerId}`, {
        method: 'PUT',
        headers: {
          'Authorization': 'Bearer admin-secret',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(updates)
      });

      if (response.ok) {
        fetchCustomers();
        alert('Customer updated successfully!');
      } else {
        alert('Failed to update customer');
      }
    } catch (error) {
      console.error('Error updating customer:', error);
      alert('Error updating customer');
    }
  };

  const testVoiceAPI = async (customerId: string) => {
    try {
      const response = await fetch('/api/admin/test-voice-api', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer admin-secret',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ customerId })
      });

      if (response.ok) {
        const data = await response.json();
        alert(`✅ Voice API Test Results:\n\nStatus: ${data.status}\nResponse Time: ${data.responseTime}ms\nAuthenticated: ${data.authenticated}\nRate Limit: ${data.rateLimit} requests/min`);
      } else {
        alert('API test failed');
      }
    } catch (error) {
      console.error('Error testing API:', error);
      alert('Error testing API');
    }
  };

  if (isLoading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <span>Loading admin dashboard...</span>
      </div>
    );
  }

  return (
    <div className="admin-dashboard">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h2>🎛️ Admin Dashboard</h2>
        <button 
          className="btn btn-primary" 
          onClick={() => setShowNewCustomerForm(true)}
        >
          + Add Customer
        </button>
      </div>

      {/* Dashboard Stats */}
      {stats && (
        <div className="stats-grid">
          <div className="stat-card">
            <h3>{stats.totalCustomers}</h3>
            <p>Total Customers</p>
          </div>
          <div className="stat-card">
            <h3>{stats.activeCustomers}</h3>
            <p>Active Customers</p>
          </div>
          <div className="stat-card">
            <h3>{stats.totalRequests.toLocaleString()}</h3>
            <p>Total API Requests</p>
          </div>
          <div className="stat-card">
            <h3>{stats.avgResponseTime}ms</h3>
            <p>Avg Response Time</p>
          </div>
        </div>
      )}

      {/* Voice Analysis API Info */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h3>🎙️ Voice Analysis API Status</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
          <div>
            <strong>Production Endpoint:</strong>
            <p style={{ fontFamily: 'monospace', background: '#f8fafc', padding: '0.5rem', borderRadius: '0.25rem', wordBreak: 'break-all' }}>
              https://voice-analysis-api-production.up.railway.app/v1/voice/analyze
            </p>
          </div>
          <div>
            <strong>Features:</strong>
            <ul style={{ marginTop: '0.5rem', color: '#374151' }}>
              <li>26 emotion scores (emo1-emo26)</li>
              <li>94 personality traits (char1-char94)</li>
              <li>Multiple audio formats supported</li>
              <li>Enterprise-grade performance</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Customers Table */}
      <div className="table-container">
        <div style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0 }}>Customer Management</h3>
          <span style={{ color: '#666', fontSize: '0.9rem' }}>
            {customers.length} total customers
          </span>
        </div>
        
        <table>
          <thead>
            <tr>
              <th>Customer</th>
              <th>Status</th>
              <th>Rate Limit</th>
              <th>Usage</th>
              <th>Success Rate</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {customers.map((customer) => (
              <tr key={customer.id}>
                <td>
                  <div>
                    <strong>{customer.label}</strong>
                    <br />
                    <small style={{ color: '#666' }}>ID: {customer.id.substring(0, 8)}...</small>
                  </div>
                </td>
                <td>
                  <span className={`status-badge ${customer.isActive ? 'status-active' : 'status-inactive'}`}>
                    {customer.isActive ? 'Active' : 'Inactive'}
                  </span>
                </td>
                <td>{customer.rateLimit}/min</td>
                <td>{customer.monthlyUsage || 0} requests</td>
                <td>{customer.successRate}%</td>
                <td>{new Date(customer.createdAt).toLocaleDateString()}</td>
                <td>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button 
                      className="btn btn-secondary" 
                      style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
                      onClick={() => setSelectedCustomer(customer)}
                    >
                      Edit
                    </button>
                    <button 
                      className="btn btn-primary" 
                      style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
                      onClick={() => testVoiceAPI(customer.id)}
                    >
                      Test API
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* New Customer Modal */}
      {showNewCustomerForm && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div className="card" style={{ width: '100%', maxWidth: '500px', margin: '2rem' }}>
            <h3>Add New Customer</h3>
            <form onSubmit={createCustomer}>
              <div className="form-group">
                <label>Customer Name:</label>
                <input
                  type="text"
                  value={newCustomer.label}
                  onChange={(e) => setNewCustomer({ ...newCustomer, label: e.target.value })}
                  placeholder="e.g., Acme Corporation API Access"
                  required
                />
              </div>
              <div className="form-group">
                <label>Rate Limit (requests per minute):</label>
                <input
                  type="number"
                  value={newCustomer.rateLimit}
                  onChange={(e) => setNewCustomer({ ...newCustomer, rateLimit: parseInt(e.target.value) })}
                  min="1"
                  max="10000"
                  required
                />
              </div>
              <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
                <button 
                  type="button" 
                  className="btn btn-secondary"
                  onClick={() => setShowNewCustomerForm(false)}
                >
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Create Customer
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit Customer Modal */}
      {selectedCustomer && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div className="card" style={{ width: '100%', maxWidth: '500px', margin: '2rem' }}>
            <h3>Edit Customer: {selectedCustomer.label}</h3>
            <form onSubmit={(e) => {
              e.preventDefault();
              const formData = new FormData(e.target as HTMLFormElement);
              updateCustomer(selectedCustomer.id, {
                label: formData.get('label') as string,
                rateLimit: parseInt(formData.get('rateLimit') as string),
                isActive: formData.get('isActive') === 'true'
              });
              setSelectedCustomer(null);
            }}>
              <div className="form-group">
                <label>Customer Name:</label>
                <input
                  type="text"
                  name="label"
                  defaultValue={selectedCustomer.label}
                  required
                />
              </div>
              <div className="form-group">
                <label>Rate Limit (requests per minute):</label>
                <input
                  type="number"
                  name="rateLimit"
                  defaultValue={selectedCustomer.rateLimit}
                  min="1"
                  max="10000"
                  required
                />
              </div>
              <div className="form-group">
                <label>Status:</label>
                <select name="isActive" defaultValue={selectedCustomer.isActive.toString()}>
                  <option value="true">Active</option>
                  <option value="false">Inactive</option>
                </select>
              </div>
              <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
                <button 
                  type="button" 
                  className="btn btn-secondary"
                  onClick={() => setSelectedCustomer(null)}
                >
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Update Customer
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminDashboard;