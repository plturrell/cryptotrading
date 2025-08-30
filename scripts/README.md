# Crypto Trading Development Scripts

This directory contains scripts for orchestrating the complete crypto trading development environment.

## Available Scripts

### 🚀 Full Development Stack
```bash
npm run dev:full
```
Starts the complete development environment including:
- **Anvil Blockchain** (port 8545) - Local Ethereum development chain
- **A2A Registry** (port 3001) - Agent registry and orchestration
- **MCP Servers** (ports 3002+) - Model Context Protocol servers for trading, analytics, risk
- **Trading Agents** - Market monitor, risk analyzer, trade executor
- **CDS Backend** (port 4004) - SAP CAP services with OData APIs
- **UI5 Frontend** (port 8080) - Fiori application

### 🤖 Agents Only
```bash
npm run start:agents
```
Starts just the trading agents:
- **Market Monitor Agent** - Real-time market data collection
- **Risk Analyzer Agent** - Portfolio risk assessment
- **Trade Executor Agent** - Automated trade execution

### ⛓️ Blockchain Only
```bash
npm run start:anvil
```
Starts local Anvil blockchain with:
- 10 pre-funded accounts
- 10,000 ETH balance each
- JSON-RPC on port 8545

### 🔧 Individual Services
```bash
npm run watch          # CDS backend only
npm run start:ui       # UI5 frontend only
npm run dev            # Backend + Frontend
```

## Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UI5 Frontend  │    │   CDS Backend   │    │ Anvil Blockchain│
│   Port: 8080    │◄──►│   Port: 4004    │◄──►│   Port: 8545    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  A2A Registry   │    │   MCP Servers   │    │ Trading Agents  │
│   Port: 3001    │    │   Port: 3002+   │    │   Background    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Agent Details

### Market Monitor Agent
- **Purpose**: Collects real-time market data from multiple sources
- **Frequency**: Every 30 seconds
- **Outputs**: Price feeds, volume data, market indicators
- **Integration**: Posts data to CDS backend `/api/market-data`

### Risk Analyzer Agent
- **Purpose**: Analyzes portfolio risk metrics
- **Frequency**: Every 60 seconds
- **Outputs**: VaR calculations, drawdown analysis, Sharpe ratios
- **Integration**: Updates risk models in CDS backend

### Trade Executor Agent
- **Purpose**: Executes automated trading strategies
- **Frequency**: Continuous monitoring
- **Outputs**: Trade orders, execution reports
- **Integration**: Interacts with blockchain and CDS backend

## Configuration

Edit `scripts/start-dev-stack.js` to modify:
- Service ports
- Agent parameters
- Blockchain configuration
- A2A registry settings

## Troubleshooting

### Anvil Not Found
```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

### Port Conflicts
Check for running services:
```bash
lsof -i :4004  # CDS Backend
lsof -i :8080  # UI5 Frontend
lsof -i :8545  # Anvil Blockchain
```

### Agent Failures
Check agent logs in the console output for specific error messages. Agents will auto-restart on failure.
