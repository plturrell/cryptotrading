# MCP System Deployment Guide

## Overview

The Model Context Protocol (MCP) system for рекс.com crypto trading platform is now production-ready and deployed on Vercel with GitHub integration.

## Architecture

### Core Components
- **Enhanced MCP Server** (`src/mcp/enhanced_server.py`) - Production-ready server with all integrations
- **Authentication** (`src/mcp/auth.py`) - API key and JWT authentication
- **Caching** (`src/mcp/cache.py`) - Lightweight in-memory caching
- **Rate Limiting** (`src/mcp/rate_limiter.py`) - Request rate limiting
- **Metrics** (`src/mcp/metrics.py`) - Performance monitoring
- **Health Checks** (`src/mcp/health.py`) - System health monitoring
- **Multi-tenancy** (`src/mcp/multi_tenant.py`) - Tenant isolation
- **Event Streaming** (`src/mcp/events.py`) - Real-time event system

### Integrations
- **Strand Framework** (`src/mcp/strand_integration.py`) - Agent workflow orchestration
- **SAP Fiori** (`src/mcp/fiori_integration.py`) - Enterprise UI integration

## Deployment

### Vercel Configuration

The system is configured for Vercel serverless deployment:

```json
{
  "functions": {
    "api/mcp.py": {
      "runtime": "python3.9"
    }
  },
  "env": {
    "MCP_API_KEY": "@mcp_api_key",
    "JWT_SECRET": "@jwt_secret"
  }
}
```

### Environment Variables

Required environment variables in Vercel:
- `MCP_API_KEY` - API key for MCP authentication
- `JWT_SECRET` - Secret for JWT token signing

### API Endpoints

#### Main MCP Endpoint
```
POST /api/mcp
Authorization: Bearer <token> | ApiKey <key>
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "tools/list",
  "params": {}
}
```

#### Health Check
```
GET /api/mcp/health
```

#### Server Info
```
GET /api/mcp/info
```

#### Metrics
```
GET /api/mcp/metrics
```

## Authentication

### API Key Authentication
```bash
curl -X POST https://your-domain.vercel.app/api/mcp \
  -H "Authorization: ApiKey your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}'
```

### JWT Authentication
```bash
curl -X POST https://your-domain.vercel.app/api/mcp \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}'
```

## Available Methods

### Core MCP Methods
- `tools/list` - List available tools
- `tools/call` - Execute a tool
- `resources/list` - List available resources
- `resources/read` - Read a resource

### Strand Integration Methods
- `strand/workflow/execute` - Execute Strand workflow
- `strand/stats` - Get integration statistics

### Fiori Integration Methods
- `fiori/manifest` - Get Fiori manifest
- `fiori/launchpad/config` - Get launchpad configuration
- `fiori/navigate` - Handle navigation
- `fiori/stats` - Get integration statistics

## Multi-tenancy

The system supports multi-tenant architecture:

```json
{
  "tenant_id": "production",
  "name": "Production Tenant",
  "trading_enabled": true,
  "portfolio_limit": 100,
  "api_rate_limit": 1000
}
```

## Monitoring

### Health Checks
The system includes comprehensive health monitoring:
- Connection health
- Authentication system health
- Cache system health
- Rate limiter health
- Metrics system health

### Metrics Collection
Metrics are collected for:
- Tool execution times
- Request rates
- Error rates
- Cache hit rates
- Authentication success rates

## Security

### Authentication
- API key validation
- JWT token validation
- Request signing verification

### Rate Limiting
- Per-user rate limiting
- Burst protection
- Tenant-based limits

### Multi-tenancy
- Tenant isolation
- Permission-based access control
- Resource segregation

## GitHub Integration

### Automated Deployment
- Automatic deployment on push to main/master
- MCP-specific testing pipeline
- Vercel integration

### Workflow Triggers
- Changes to `src/mcp/**`
- Changes to `api/mcp.py`
- Changes to `requirements.txt`
- Changes to `vercel.json`

## Production Considerations

### Performance
- Lightweight, stateless design
- Efficient caching strategies
- Optimized for serverless execution

### Scalability
- Horizontal scaling via Vercel
- Stateless architecture
- Event-driven updates

### Reliability
- Comprehensive error handling
- Health monitoring
- Graceful degradation

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Check API key configuration
   - Verify JWT secret
   - Validate token format

2. **Rate Limiting**
   - Check rate limit configuration
   - Monitor request patterns
   - Adjust limits per tenant

3. **Health Check Failures**
   - Check component status
   - Review error logs
   - Verify dependencies

### Debugging

Use the health check endpoint to diagnose issues:
```bash
curl https://your-domain.vercel.app/api/mcp/health
```

Check metrics for performance insights:
```bash
curl https://your-domain.vercel.app/api/mcp/metrics
```

## Next Steps

1. Configure environment variables in Vercel
2. Set up GitHub secrets for deployment
3. Test API endpoints
4. Monitor health and metrics
5. Scale based on usage patterns

The MCP system is now production-ready and provides enterprise-grade capabilities for your crypto trading platform.
