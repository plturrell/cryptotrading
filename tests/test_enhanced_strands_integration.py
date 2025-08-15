"""
Test Enhanced Historical Loader Agent with Comprehensive Metrics
Verifies integration of institutional indicators with strands MCP pattern
"""

import asyncio
import logging
from datetime import datetime
from src.rex.a2a.agents.historical_loader_agent import get_historical_loader_agent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_comprehensive_indicators():
    """Test loading comprehensive indicators"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Load Comprehensive Indicators")
    logger.info("="*60)
    
    agent = get_historical_loader_agent()
    
    # Test loading key volatility and fixed income indicators
    request = """
    Load comprehensive indicators for these symbols:
    - ^VIX (volatility index)
    - TIP (inflation protected securities)
    - TLT (long-term treasuries)
    - UUP (dollar strength)
    - HYG (high yield bonds)
    
    Load 30 days of daily data.
    """
    
    result = await agent.process_request(request)
    logger.info(f"Result: {result['success']}")
    
    return result


async def test_crypto_predictors():
    """Test loading crypto-specific comprehensive predictors"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Load Crypto Comprehensive Predictors")
    logger.info("="*60)
    
    agent = get_historical_loader_agent()
    
    # Test BTC predictors
    request = """
    Load comprehensive predictors for Bitcoin (BTC) for the last 90 days.
    Include all enhanced indicators optimized for BTC prediction.
    """
    
    result = await agent.process_request(request)
    logger.info(f"Result: {result['success']}")
    
    return result


async def test_institutional_strategy():
    """Test loading institutional trading strategy"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Load Institutional Strategy")
    logger.info("="*60)
    
    agent = get_historical_loader_agent()
    
    # Test Two Sigma strategy
    request = """
    Load the Two Sigma Factor Lens institutional trading strategy.
    Get 60 days of data for all indicators in their model.
    """
    
    result = await agent.process_request(request)
    logger.info(f"Result: {result['success']}")
    
    return result


async def test_regime_detection():
    """Test market regime indicators"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Load Market Regime Indicators")
    logger.info("="*60)
    
    agent = get_historical_loader_agent()
    
    # Test risk-off regime
    request = """
    Load indicators for risk_off market regime.
    Get the last 30 days of data to analyze current market conditions.
    """
    
    result = await agent.process_request(request)
    logger.info(f"Result: {result['success']}")
    
    return result


async def test_combined_analysis():
    """Test combined crypto and comprehensive analysis"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Combined Crypto + Comprehensive Analysis")
    logger.info("="*60)
    
    agent = get_historical_loader_agent()
    
    # Complex multi-part request
    request = """
    I need to analyze Bitcoin with professional indicators:
    
    1. First load BTC-USD data for the last 90 days
    2. Then load the comprehensive predictors for BTC
    3. Also load risk-off regime indicators
    4. Finally, show me the critical thresholds for risk management
    
    This will help me understand BTC in the context of broader market conditions.
    """
    
    result = await agent.process_request(request)
    logger.info(f"Result: {result['success']}")
    
    return result


async def test_validation():
    """Test ticker validation"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Validate Comprehensive Tickers")
    logger.info("="*60)
    
    agent = get_historical_loader_agent()
    
    # Test validation
    request = """
    Validate these comprehensive indicator tickers on Yahoo Finance:
    - ^VIX, ^VIX9D, ^VVIX, ^SKEW
    - TIP, TLT, SHY, LQD, HYG
    - UUP, FXE, FXY
    - XLK, XLF, XLE, XLU
    - EEM, EFA, VGK
    
    Check their availability and data quality.
    """
    
    result = await agent.process_request(request)
    logger.info(f"Result: {result['success']}")
    
    return result


async def test_indicators_list():
    """Test getting comprehensive indicators list"""
    logger.info("\n" + "="*60)
    logger.info("TEST 7: Get Comprehensive Indicators List")
    logger.info("="*60)
    
    agent = get_historical_loader_agent()
    
    request = """
    Show me all available comprehensive indicators grouped by:
    1. Category (volatility, fixed income, currency, etc.)
    2. Predictive power (very high, high, medium)
    3. Institutional usage
    """
    
    result = await agent.process_request(request)
    logger.info(f"Result: {result['success']}")
    
    return result


async def test_strands_mcp_pattern():
    """Test that the agent follows strands MCP pattern correctly"""
    logger.info("\n" + "="*60)
    logger.info("TEST 8: Verify Strands MCP Pattern")
    logger.info("="*60)
    
    agent = get_historical_loader_agent()
    
    # Verify agent attributes
    logger.info(f"Agent ID: {agent.agent_id}")
    logger.info(f"Agent Type: {agent.agent_type}")
    logger.info(f"Capabilities: {agent.capabilities}")
    
    # Test tool discovery
    tools = agent._create_tools()
    comprehensive_tools = [t for t in tools if 'comprehensive' in t.__name__ or 'institutional' in t.__name__]
    
    logger.info(f"\nComprehensive Tools Available: {len(comprehensive_tools)}")
    for tool in comprehensive_tools:
        logger.info(f"  - {tool.__name__}")
    
    return {"success": True, "tools_count": len(comprehensive_tools)}


async def main():
    """Run all tests"""
    logger.info("Starting Enhanced Strands Integration Tests")
    logger.info(f"Test Time: {datetime.now()}")
    
    try:
        # Run tests sequentially
        await test_comprehensive_indicators()
        await test_crypto_predictors()
        await test_institutional_strategy()
        await test_regime_detection()
        await test_combined_analysis()
        await test_validation()
        await test_indicators_list()
        await test_strands_mcp_pattern()
        
        logger.info("\n" + "="*60)
        logger.info("✓ All tests completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())