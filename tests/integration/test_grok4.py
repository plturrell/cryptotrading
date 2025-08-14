"""
Integration tests for Grok-4 client
"""

import pytest
import asyncio
from src.rex.a2a.grok4_client import get_grok4_client


@pytest.mark.integration
async def test_grok4_integration():
    """Test Grok-4 integration"""
    client = get_grok4_client()
    
    # Test connection
    assert client.test_connection(), "Grok-4 connection failed"
    
    # Test crypto analysis with real data
    test_data = {
        "symbol": "ETH-USD",
        "price": 3000,
        "volume": 500000,
        "rsi": 55,
        "date_range": "2024-01-01 to 2024-01-31"
    }
    
    result = await client.analyze_crypto_data(test_data)
    assert result["success"], f"Crypto analysis failed: {result.get('error')}"
    assert "content" in result
    assert len(result["content"]) > 0


@pytest.mark.integration
async def test_grok4_trading_decision():
    """Test Grok-4 trading decision making"""
    client = get_grok4_client()
    
    # Test trading decision
    market_data = {
        "symbol": "ETH-USD",
        "current_price": 3000,
        "24h_change": 5.2,
        "volume": 1000000,
        "rsi": 70,
        "macd": "bullish"
    }
    
    result = await client.make_trading_decision(market_data)
    assert result["success"], f"Trading decision failed: {result.get('error')}"
    assert "recommendation" in result
    assert result["recommendation"] in ["buy", "sell", "hold"]


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_grok4_integration())
    asyncio.run(test_grok4_trading_decision())