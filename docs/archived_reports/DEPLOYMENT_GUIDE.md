# MCTS Deployment Guide

## Overview

This guide covers deploying the enhanced MCTS system to both local development and Vercel production environments.

## 🚀 Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy development environment
cp env_development .env

# 3. Run locally
python app.py

# 4. Test the API
curl -X POST http://localhost:8000/api/mcts-calculate \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "initial_portfolio": 10000,
      "symbols": ["BTC", "ETH"],
      "max_depth": 5
    }
  }'
```

### Vercel Deployment

```bash
# 1. Install Vercel CLI
npm i -g vercel

# 2. Login to Vercel
vercel login

# 3. Deploy to production
vercel --prod

# 4. Set environment variables
vercel env add MCTS_JWT_SECRET
vercel env add KV_REST_API_URL
vercel env add KV_REST_API_TOKEN
```

## 📋 Configuration

### Environment Variables

#### Core MCTS Settings
- `MCTS_ITERATIONS`: Number of MCTS iterations (default: 1000)
- `MCTS_EXPLORATION`: UCB1 exploration constant (default: 1.4)
- `MCTS_SIM_DEPTH`: Maximum simulation depth (default: 10)
- `MCTS_TIMEOUT`: Operation timeout in seconds (default: 25 for Vercel)
- `MCTS_MAX_MEMORY_MB`: Memory limit in MB (default: 512)
- `MCTS_SIMULATION_STRATEGY`: `pure_random` or `weighted_random`

#### Security Settings
- `MCTS_SECURITY_LEVEL`: `development`, `testing`, `staging`, or `production`
- `MCTS_JWT_SECRET`: Secret key for JWT tokens (required in production)

#### Vercel-Specific
- `VERCEL_EDGE`: Enable Edge Runtime optimizations
- `KV_REST_API_URL`: Vercel KV store URL
- `KV_REST_API_TOKEN`: Vercel KV authentication token

### vercel.json Configuration

```json
{
  "functions": {
    "api/mcts-calculate.ts": {
      "runtime": "edge",
      "maxDuration": 30,
      "memory": 1024
    }
  }
}
```

## 🏗️ Architecture

### Local Development Setup

```
Local Development
├── Python MCTS Agent (full features)
├── Local Redis Cache
├── Background Monitoring
└── Full AsyncIO Support
```

### Vercel Production Setup

```
Vercel Edge Runtime
├── TypeScript Edge Function (api/mcts-calculate.ts)
├── Python Serverless Function (api/mcts-python.py)
├── Vercel KV Cache
└── No Background Tasks
```

## 🔧 Key Differences: Local vs Vercel

### Memory Management
- **Local**: Uses `psutil` for accurate memory tracking
- **Vercel**: Estimates memory based on tree size (psutil not available)

### Background Tasks
- **Local**: Continuous monitoring dashboard
- **Vercel**: Request-based metrics only (no background tasks)

### Timeouts
- **Local**: Configurable, can be long-running
- **Vercel**: Hard limit of 30 seconds for Edge Functions

### Caching
- **Local**: Redis or in-memory cache
- **Vercel**: Vercel KV store integration

## 🛡️ Security

### Development Mode
- Basic authentication (admin/admin123)
- Simple API keys
- Local JWT signing

### Production Mode
- Environment-based JWT secret
- Secure API key generation
- Rate limiting per user/key
- IP-based blocking

## 📊 Monitoring

### Local Development
```python
# Monitoring automatically starts
# Access dashboard data via API
GET /api/mcts/monitoring/health
GET /api/mcts/monitoring/metrics
```

### Vercel Production
```javascript
// Metrics sent to external service
// Configure in vercel.json
"env": {
  "MONITORING_ENDPOINT": "https://your-metrics-service.com"
}
```

## 🧪 Testing Deployment

### Local Tests
```bash
# Run all tests
pytest tests/

# Run deployment-specific tests
pytest tests/test_vercel_deployment.py
```

### Vercel Preview
```bash
# Deploy to preview
vercel

# Test preview deployment
curl https://your-preview-url.vercel.app/api/mcts-calculate
```

## 🚨 Common Issues

### 1. Memory Limit Errors
**Problem**: "Approaching Vercel memory limit"
**Solution**: 
- Reduce `MCTS_ITERATIONS`
- Enable more aggressive tree pruning
- Decrease `max_depth` parameter

### 2. Timeout Errors
**Problem**: "Operation approaching Vercel timeout limit"
**Solution**:
- Lower `MCTS_ITERATIONS` to 500-1000
- Enable early convergence detection
- Use cached results when possible

### 3. Import Errors
**Problem**: "Module not found" in Vercel
**Solution**:
- Check `includeFiles` in vercel.json
- Use absolute imports
- Verify requirements.txt is complete

### 4. Cache Connection Failed
**Problem**: "Cache operation failed"
**Solution**:
- Verify KV_REST_API_URL is set
- Check Vercel KV is enabled
- Use fallback to computation

## 📈 Performance Optimization

### For Vercel Deployment

1. **Optimize Iterations**
   ```python
   # Production settings for 30s limit
   MCTS_ITERATIONS=800
   MCTS_PARALLEL_SIMS=2
   ```

2. **Enable Aggressive Pruning**
   ```python
   # Prune at 3000 nodes instead of 5000
   if tree_size > 3000:
       self._prune_tree(root, keep_ratio=0.5)
   ```

3. **Use Cached Results**
   ```python
   # Increase cache TTL for stable markets
   MCTS_CACHE_TTL=1800  # 30 minutes
   ```

## 🔄 Continuous Deployment

### GitHub Actions Setup

```yaml
# .github/workflows/deploy.yml
name: Deploy to Vercel
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
```

## 📝 Deployment Checklist

- [ ] Environment variables configured
- [ ] vercel.json properly set up
- [ ] API authentication enabled
- [ ] Rate limiting configured
- [ ] Error handling tested
- [ ] Memory limits verified
- [ ] Timeout handling confirmed
- [ ] Cache integration working
- [ ] Security headers set
- [ ] CORS configured

## 🆘 Support

For deployment issues:
1. Check Vercel function logs
2. Verify environment variables
3. Test with reduced iterations
4. Review memory usage patterns

For production emergencies:
1. Revert to previous deployment
2. Increase timeout temporarily
3. Disable expensive features
4. Scale down iterations