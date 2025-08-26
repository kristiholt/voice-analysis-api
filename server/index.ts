import express from 'express';
import cors from 'cors';
import { db } from './db';
import { apiKeys } from '../shared/schema';
import { eq, and, desc, count, sum, avg, sql } from 'drizzle-orm';

const app = express();
const PORT = parseInt(process.env.PORT || '3000');

// Middleware
app.use(cors());
app.use(express.json());

// Simple authentication middleware (for demo - replace with proper auth)
const authenticateAdmin = (req: any, res: any, next: any) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || authHeader !== 'Bearer admin-secret') {
    return res.status(401).json({ message: 'Unauthorized' });
  }
  next();
};

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'customer-management-api' });
});

// API Routes

// Get all customers (admin)
app.get('/api/admin/customers', authenticateAdmin, async (req, res) => {
  try {
    const customers = await db.select().from(apiKeys).orderBy(desc(apiKeys.createdAt));
    
    const customersWithStats = customers.map(customer => ({
      id: customer.id,
      label: customer.label,
      isActive: customer.isActive,
      rateLimit: customer.rateLimitPerMin,
      createdAt: customer.createdAt,
      // Mock stats for now - will be real when usage tracking is added
      totalRequests: Math.floor(Math.random() * 1000),
      successRate: Math.floor(Math.random() * 20 + 80), // 80-100%
      monthlyUsage: Math.floor(Math.random() * 500),
    }));

    res.json({
      customers: customersWithStats,
      totalCustomers: customers.length,
      activeCustomers: customers.filter(c => c.isActive).length
    });
  } catch (error) {
    console.error('Error fetching customers:', error);
    res.status(500).json({ message: 'Failed to fetch customers' });
  }
});

// Get customer by ID (admin)
app.get('/api/admin/customers/:id', authenticateAdmin, async (req, res) => {
  try {
    const { id } = req.params;
    const [customer] = await db.select().from(apiKeys).where(eq(apiKeys.id, id));
    
    if (!customer) {
      return res.status(404).json({ message: 'Customer not found' });
    }

    // Mock detailed stats
    const detailedStats = {
      ...customer,
      dailyUsage: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        requests: Math.floor(Math.random() * 50),
        successRate: Math.floor(Math.random() * 20 + 80)
      })),
      topEndpoints: [
        { endpoint: '/v1/voice/analyze', requests: Math.floor(Math.random() * 800 + 200) },
        { endpoint: '/health', requests: Math.floor(Math.random() * 100) }
      ],
      avgResponseTime: Math.floor(Math.random() * 500 + 200) // 200-700ms
    };

    res.json(detailedStats);
  } catch (error) {
    console.error('Error fetching customer:', error);
    res.status(500).json({ message: 'Failed to fetch customer details' });
  }
});

// Create new customer (admin)
app.post('/api/admin/customers', authenticateAdmin, async (req, res) => {
  try {
    const { label, rateLimit } = req.body;
    
    if (!label) {
      return res.status(400).json({ message: 'Label is required' });
    }

    // Generate API key
    const apiKey = Buffer.from(require('crypto').randomBytes(32)).toString('base64url');
    const keyHash = require('crypto').createHash('sha256').update(apiKey).digest('hex');

    const [newCustomer] = await db.insert(apiKeys).values({
      keyHash,
      label,
      rateLimitPerMin: rateLimit || 1000,
      isActive: true,
    }).returning();

    res.status(201).json({
      customer: newCustomer,
      apiKey: apiKey, // Return the raw key once
      message: 'Customer created successfully'
    });
  } catch (error) {
    console.error('Error creating customer:', error);
    res.status(500).json({ message: 'Failed to create customer' });
  }
});

// Update customer (admin)
app.put('/api/admin/customers/:id', authenticateAdmin, async (req, res) => {
  try {
    const { id } = req.params;
    const { label, rateLimit, isActive } = req.body;

    const [updatedCustomer] = await db.update(apiKeys)
      .set({ 
        label, 
        rateLimitPerMin: rateLimit, 
        isActive
      })
      .where(eq(apiKeys.id, id))
      .returning();

    if (!updatedCustomer) {
      return res.status(404).json({ message: 'Customer not found' });
    }

    res.json({ customer: updatedCustomer, message: 'Customer updated successfully' });
  } catch (error) {
    console.error('Error updating customer:', error);
    res.status(500).json({ message: 'Failed to update customer' });
  }
});

// Dashboard stats (admin)
app.get('/api/admin/dashboard', authenticateAdmin, async (req, res) => {
  try {
    // Get basic stats
    const totalCustomers = await db.select({ count: count() }).from(apiKeys);
    const activeCustomers = await db.select({ count: count() }).from(apiKeys).where(eq(apiKeys.isActive, true));
    
    // Mock additional stats
    const stats = {
      totalCustomers: totalCustomers[0]?.count || 0,
      activeCustomers: activeCustomers[0]?.count || 0,
      totalRequests: Math.floor(Math.random() * 10000 + 5000), // Mock data
      avgResponseTime: Math.floor(Math.random() * 300 + 200),
      recentActivity: Array.from({ length: 7 }, (_, i) => ({
        date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        requests: Math.floor(Math.random() * 500 + 100)
      })).reverse()
    };

    res.json(stats);
  } catch (error) {
    console.error('Error fetching dashboard stats:', error);
    res.status(500).json({ message: 'Failed to fetch dashboard stats' });
  }
});

// Voice Analysis API Integration Test
app.post('/api/admin/test-voice-api', authenticateAdmin, async (req, res) => {
  try {
    const { customerId } = req.body;
    
    // Get customer API key
    const [customer] = await db.select().from(apiKeys).where(eq(apiKeys.id, customerId));
    
    if (!customer) {
      return res.status(404).json({ message: 'Customer not found' });
    }

    // Mock voice API test (replace with actual API call)
    const testResult = {
      status: 'success',
      endpoint: 'https://voice-analysis-api-production.up.railway.app/v1/voice/analyze',
      responseTime: Math.floor(Math.random() * 300 + 100),
      authenticated: customer.isActive,
      rateLimit: customer.rateLimitPerMin
    };

    res.json(testResult);
  } catch (error) {
    console.error('Error testing voice API:', error);
    res.status(500).json({ message: 'Failed to test voice API' });
  }
});

// Customer password-based login
app.post('/api/customer/login', async (req, res) => {
  try {
    const { password } = req.body;
    
    if (!password) {
      return res.status(400).json({ message: 'Password is required' });
    }

    const [customer] = await db.select().from(apiKeys).where(eq(apiKeys.password, password));
    
    if (!customer || !customer.isActive) {
      return res.status(401).json({ message: 'Invalid password or inactive account' });
    }

    res.json({
      success: true,
      customer: {
        id: customer.id,
        label: customer.label,
        isActive: customer.isActive
      }
    });
  } catch (error) {
    console.error('Error in customer login:', error);
    res.status(500).json({ message: 'Login error' });
  }
});

// Customer-facing API (for customer self-service) - Keep for API key integration
const authenticateCustomer = async (req: any, res: any, next: any) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ message: 'API key required' });
  }

  const apiKey = authHeader.substring(7);
  const keyHash = require('crypto').createHash('sha256').update(apiKey).digest('hex');

  try {
    const [customer] = await db.select().from(apiKeys).where(eq(apiKeys.keyHash, keyHash));
    
    if (!customer || !customer.isActive) {
      return res.status(401).json({ message: 'Invalid or inactive API key' });
    }

    req.customer = customer;
    next();
  } catch (error) {
    return res.status(500).json({ message: 'Authentication error' });
  }
};

// Alternative authentication middleware for password-based sessions
const authenticateCustomerByPassword = async (req: any, res: any, next: any) => {
  const { customerId } = req.headers;
  
  if (!customerId) {
    return res.status(401).json({ message: 'Customer ID required' });
  }

  try {
    const [customer] = await db.select().from(apiKeys).where(eq(apiKeys.id, customerId));
    
    if (!customer || !customer.isActive) {
      return res.status(401).json({ message: 'Invalid customer' });
    }

    req.customer = customer;
    next();
  } catch (error) {
    return res.status(500).json({ message: 'Authentication error' });
  }
};

// Customer dashboard
app.get('/api/customer/dashboard', authenticateCustomerByPassword, async (req: any, res) => {
  try {
    const customer = req.customer;
    
    // Get the actual API key for this customer to display
    const apiKeyResponse = await db.select().from(apiKeys).where(eq(apiKeys.id, customer.id));
    const actualApiKey = apiKeyResponse[0];
    
    // Generate API key from hash (for display purposes, create a demo key)
    const displayApiKey = actualApiKey?.keyHash ? 
      Buffer.from(actualApiKey.keyHash.substring(0, 32), 'hex').toString('base64url') + 'Demo' :
      'D16wdv9Tt79olNYt2b6nkcPxSt2VX4exeoG__maXHwQ'; // Default demo key

    // Mock customer stats
    const stats = {
      label: customer.label,
      rateLimit: customer.rateLimitPerMin,
      currentUsage: Math.floor(Math.random() * customer.rateLimitPerMin * 0.8),
      totalRequests: Math.floor(Math.random() * 1000),
      successRate: Math.floor(Math.random() * 20 + 80),
      apiKey: displayApiKey,
      monthlyQuota: customer.rateLimitPerMin * 30, // Daily limit * 30
      quotaUsed: Math.floor(Math.random() * customer.rateLimitPerMin * 15), // Mock usage
      recentRequests: Array.from({ length: 10 }, (_, i) => ({
        timestamp: new Date(Date.now() - i * 60 * 60 * 1000).toISOString(),
        endpoint: '/v1/voice/analyze',
        status: Math.random() > 0.1 ? 'success' : 'error',
        responseTime: Math.floor(Math.random() * 300 + 100)
      }))
    };

    res.json(stats);
  } catch (error) {
    console.error('Error fetching customer dashboard:', error);
    res.status(500).json({ message: 'Failed to fetch dashboard' });
  }
});

// Customer usage analytics
app.get('/api/customer/usage', authenticateCustomerByPassword, async (req: any, res) => {
  try {
    const customer = req.customer;
    const { period = '7d' } = req.query;

    // Mock usage data based on period
    const days = period === '30d' ? 30 : 7;
    const usageData = Array.from({ length: days }, (_, i) => ({
      date: new Date(Date.now() - (days - 1 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      requests: Math.floor(Math.random() * 50),
      successRate: Math.floor(Math.random() * 20 + 80),
      avgResponseTime: Math.floor(Math.random() * 200 + 100)
    }));

    res.json({
      period,
      data: usageData,
      summary: {
        totalRequests: usageData.reduce((sum, day) => sum + day.requests, 0),
        avgSuccessRate: Math.round(usageData.reduce((sum, day) => sum + day.successRate, 0) / days),
        avgResponseTime: Math.round(usageData.reduce((sum, day) => sum + day.avgResponseTime, 0) / days)
      }
    });
  } catch (error) {
    console.error('Error fetching usage data:', error);
    res.status(500).json({ message: 'Failed to fetch usage data' });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Customer Management API running on port ${PORT}`);
  console.log(`📊 Admin Dashboard: http://localhost:${PORT}/api/admin/dashboard`);
  console.log(`🔑 API Docs: http://localhost:${PORT}/health`);
});