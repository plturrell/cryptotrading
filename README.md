# рекс.com - Cryptocurrency AI Trading Platform

This repository contains the source code and documentation for рекс.com, a professional AI-powered cryptocurrency trading platform.

## Contents

### Strategic Planning Documents
- `plan.pdf` - Core trading system plan
- `plan0_30.pdf` - Initial 30-day implementation strategy

### Comprehensive Guides (Russian)
- `Комплексное руководство по системам ИИ-трейдинга криптовалют.pdf` - Complete guide to AI cryptocurrency trading systems
- `Комплексный отчет по криптовалютным системам AI-трейдинга_ Стратегический анализ 2024-2025.pdf` - Comprehensive report on AI trading systems with strategic analysis for 2024-2025
- `Продвинутая адаптация a2a для криптовалютной AI-системы_ Техническая спецификация.pdf` - Advanced A2A adaptation for cryptocurrency AI systems technical specification
- `Реалистичная стратегия роста капитала 5K→50K SGD_ AI + Human Supervision.pdf` - Realistic capital growth strategy from 5K to 50K SGD with AI and human supervision

## Overview

This documentation collection covers:
- AI trading system architecture
- Implementation strategies
- Risk management approaches
- Capital growth methodologies
- Technical specifications for automated trading systems

## Usage

These documents serve as reference materials for developing and implementing AI-powered cryptocurrency trading systems with proper risk management and strategic planning.

## Deployment

The platform is deployed on Vercel and accessible at:
- Production: https://рекс.com (requires authentication)
- Vercel URL: https://cryptotrading-4gqyiuua0-plturrells-projects.vercel.app

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the development server:
   ```bash
   python app.py
   ```

3. Access at http://localhost:5000

### API Endpoints

#### Core Endpoints
- `/health` - Health check
- `/api/` - API documentation (Swagger UI)
- `/api/trading/status` - Trading system status

#### Market Data Endpoints
- `/api/market/data?symbol={symbol}&network={network}` - Aggregated market data from multiple sources
- `/api/market/overview?symbols={symbol1,symbol2}` - Market overview for multiple symbols
- `/api/market/historical/{symbol}?days={days}` - Historical market data
- `/api/market/dex/trending?network={network}` - Trending DEX pools
- `/api/market/dex/opportunities?min_liquidity={amount}` - DEX trading opportunities
- `/api/market/dex/pool/{network}/{address}` - Specific DEX pool data

#### AI Analysis Endpoints
- `/api/ai/analyze` - AI market analysis using DeepSeek R1
- `/api/ai/news/{symbol}` - Real-time crypto news via Perplexity
- `/api/ai/signals/{symbol}` - AI trading signals

#### Wallet & DeFi Endpoints
- `/api/wallet/balance` - MetaMask wallet balance
- `/api/wallet/monitor` - Wallet monitoring
- `/api/defi/opportunities` - DeFi opportunities
- `/api/wallet/gas` - Gas price optimization

### Market Data Sources

The platform aggregates data from multiple sources with rate limiting:

#### Free Tier APIs:
- **GeckoTerminal**: DEX data from 1,600+ exchanges (30 calls/min limit)
- **CoinGecko**: Comprehensive crypto market data (10 calls/min on free tier)
- **Yahoo Finance**: Historical data via yfinance (unofficial limits)

#### API Key Required:
- **CoinMarketCap**: Market data and trending analysis (30 calls/min)
- **Bitquery**: GraphQL API for DEX trades (60 calls/min)

### Historical Data Sources:
- **CryptoDataDownload**: Free CSV downloads from 20+ exchanges
- **Yahoo Finance**: Major crypto pairs with technical indicators
- **Bitget**: Limited to 1 download per coin per day

### Rate Limiting

All API calls are managed through a centralized rate limiter to ensure compliance with free tier limits. Check current usage with `/api/limits`.

## Disclaimer

This repository contains educational and research materials. Any trading strategies discussed should be thoroughly tested and validated before implementation. Cryptocurrency trading involves significant risks, and past performance does not guarantee future results.