# Service Directory - Crypto Trading Application

This directory contains service-related components for backend integration and service management.

## Components

### ServiceRegistry.js
Centralized service registry that manages all backend service configurations and instances.

**Features:**
- Service registration and discovery
- Configuration management
- Service health monitoring
- Instance lifecycle management
- Support for multiple service types (REST, WebSocket, Blockchain)

**Usage:**
```javascript
// Get service registry from component
var oServiceRegistry = this.getOwnerComponent().getServiceRegistry();

// Get a service instance
var oTradingService = oServiceRegistry.getService("trading");

// Make a request
oTradingService.request("POST", "orders", orderData)
    .then(function(result) {
        // Handle success
    });
```

## Registered Services

### Market Data Service
- **Type**: WebSocket with REST fallback
- **Purpose**: Real-time cryptocurrency market data
- **Endpoint**: `/ws/market-data`
- **Fallback**: `/api/v1/market-data`

### Trading Service
- **Type**: REST API
- **Purpose**: Order execution and management
- **Base URL**: `/api/v1/trading`
- **Endpoints**: orders, positions, history, balance

### Portfolio Service
- **Type**: REST API
- **Purpose**: Portfolio tracking and analysis
- **Base URL**: `/api/v1/portfolio`
- **Endpoints**: summary, holdings, performance, allocation

### Analytics Service
- **Type**: REST API
- **Purpose**: Technical analysis and indicators
- **Base URL**: `/api/v1/analytics`
- **Endpoints**: indicators, signals, backtesting, reports

### Risk Management Service
- **Type**: REST API
- **Purpose**: Risk assessment and monitoring
- **Base URL**: `/api/v1/risk`
- **Endpoints**: assessment, limits, monitoring, alerts

### Wallet Service
- **Type**: Blockchain integration
- **Purpose**: Cryptocurrency wallet management
- **Networks**: Ethereum, Polygon
- **Wallets**: MetaMask, WalletConnect, Coinbase

### News Service
- **Type**: REST API
- **Purpose**: Cryptocurrency news and sentiment
- **Base URL**: `/api/v1/news`
- **Endpoints**: latest, search, categories, sentiment

## Service Types

### REST Services
Standard HTTP REST API services with:
- Configurable endpoints
- Timeout and retry logic
- Error handling
- Response caching

### WebSocket Services
Real-time data services with:
- Automatic reconnection
- Connection health monitoring
- Fallback to REST endpoints
- Message queuing

### Blockchain Services
Cryptocurrency wallet integration with:
- Multiple network support
- Wallet provider abstraction
- Transaction management
- Gas optimization

## Configuration

Services are configured in the ServiceRegistry with:
- Endpoint URLs
- Timeout settings
- Retry policies
- Health check parameters
- Service-specific options

## Integration

The ServiceRegistry is initialized in Component.js and made available to all controllers through the BaseController pattern.

## Future Enhancements

- Service discovery from external registry
- Load balancing for multiple endpoints
- Circuit breaker pattern for fault tolerance
- Metrics collection and monitoring
- Service mesh integration
