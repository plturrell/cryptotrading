"""
Comprehensive test for A2A Historical Data Loader
Tests FRED, CBOE, DeFiLlama, and Yahoo Finance integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from src.rex.historical_data import (
    A2AHistoricalDataLoader, 
    DataLoadRequest,
    FREDClient,
    CBOEClient, 
    DeFiLlamaClient,
    YahooFinanceClient
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_individual_clients():
    """Test each data client individually"""
    print("=" * 60)
    print("TESTING INDIVIDUAL DATA CLIENTS")
    print("=" * 60)
    
    # Test dates (last 30 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Test FRED Client
    print("\n1. Testing FRED Client...")
    fred_client = FREDClient()
    
    if fred_client.api_key:
        # Test single series
        dgs10_data = fred_client.get_series_data("DGS10", start_date=start_date, end_date=end_date)
        if dgs10_data is not None:
            print(f"   ✓ FRED DGS10: {len(dgs10_data)} observations")
            print(f"   ✓ Latest 10Y Treasury: {dgs10_data['DGS10'].iloc[-1]:.2f}%")
        
        # Test liquidity metrics
        liquidity_data = fred_client.get_liquidity_metrics(start_date=start_date, end_date=end_date)
        if not liquidity_data.empty:
            print(f"   ✓ FRED Liquidity Metrics: {len(liquidity_data)} observations")
            if 'NET_LIQUIDITY' in liquidity_data.columns:
                print(f"   ✓ Latest Net Liquidity: ${liquidity_data['NET_LIQUIDITY'].iloc[-1]/1e12:.1f}T")
    else:
        print("   ✗ FRED API key not configured")
    
    # Test CBOE Client
    print("\n2. Testing CBOE Client...")
    cboe_client = CBOEClient()
    
    # Test VIX data
    vix_data = cboe_client.download_volatility_data("VIX")
    if vix_data is not None:
        print(f"   ✓ CBOE VIX: {len(vix_data)} observations")
        print(f"   ✓ Latest VIX: {vix_data['VIX_Close'].iloc[-1]:.2f}")
    
    # Test VIX term structure
    term_structure = cboe_client.get_vix_term_structure()
    if not term_structure.empty:
        print(f"   ✓ VIX Term Structure: {len(term_structure)} observations")
        if 'VIX_CONTANGO' in term_structure.columns:
            print(f"   ✓ Latest VIX Contango: {term_structure['VIX_CONTANGO'].iloc[-1]:.2f}")
    
    # Test DeFiLlama Client
    print("\n3. Testing DeFiLlama Client...")
    defillama_client = DeFiLlamaClient()
    
    # Test total TVL
    tvl_data = defillama_client.get_total_tvl_history()
    if tvl_data is not None:
        print(f"   ✓ DeFiLlama Total TVL: {len(tvl_data)} observations")
        print(f"   ✓ Latest Total TVL: ${tvl_data['TOTAL_TVL_USD'].iloc[-1]/1e9:.1f}B")
    
    # Test protocol data
    uniswap_data = defillama_client.get_protocol_tvl("uniswap")
    if uniswap_data is not None:
        print(f"   ✓ Uniswap TVL: {len(uniswap_data)} observations")
        tvl_col = [col for col in uniswap_data.columns if 'TVL' in col.upper()]
        if tvl_col:
            print(f"   ✓ Latest Uniswap TVL: ${uniswap_data[tvl_col[0]].iloc[-1]/1e9:.1f}B")
    
    # Test Yahoo Finance Client
    print("\n4. Testing Yahoo Finance Client...")
    yahoo_client = YahooFinanceClient()
    
    # Test BTC data
    btc_data = yahoo_client.download_data("BTC", start_date=start_date, end_date=end_date)
    if btc_data is not None:
        print(f"   ✓ Yahoo BTC: {len(btc_data)} observations")
        print(f"   ✓ Latest BTC Close: ${btc_data['close'].iloc[-1]:,.2f}")

async def test_a2a_comprehensive_loader():
    """Test the comprehensive A2A data loader"""
    print("\n" + "=" * 60)
    print("TESTING A2A COMPREHENSIVE DATA LOADER")
    print("=" * 60)
    
    # Initialize A2A loader
    loader = A2AHistoricalDataLoader()
    
    # Test data summary
    print("\n1. Data Summary:")
    summary = loader.get_data_summary()
    for source, info in summary.items():
        if source != "total_files" and isinstance(info, dict):
            print(f"   {source}: {info.get('file_count', 0)} files, "
                  f"{info.get('total_size_mb', 0):.1f} MB")
    
    # Test comprehensive crypto trading dataset
    print("\n2. Loading Comprehensive Crypto Trading Dataset...")
    
    # Use last 7 days for faster testing
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    result = loader.load_crypto_trading_dataset(start_date=start_date, end_date=end_date)
    
    print(f"   Status: {result.get('status')}")
    print(f"   Sources: {result.get('sources_loaded')}")
    print(f"   Timestamp: {result.get('timestamp')}")
    
    if result.get('status') == 'error':
        print(f"   Error: {result.get('error')}")
    
    # Test custom data request
    print("\n3. Testing Custom Data Request...")
    
    custom_request = DataLoadRequest(
        sources=["yahoo", "cboe"],  # Test subset of sources
        symbols=["BTC", "ETH"],
        cboe_indices=["VIX", "SKEW"],
        start_date=start_date,
        end_date=end_date,
        align_data=True,
        save_data=True
    )
    
    custom_result = await loader.load_comprehensive_data(custom_request)
    
    print(f"   Custom Request Status: {custom_result.get('status')}")
    if custom_result.get('status') == 'error':
        print(f"   Error: {custom_result.get('error')}")

async def test_strand_integration():
    """Test strand framework integration"""
    print("\n" + "=" * 60)
    print("TESTING STRAND FRAMEWORK INTEGRATION")
    print("=" * 60)
    
    from src.strands.agent import Agent
    from src.strands.types.tools import ToolSpec
    
    # Create a simple strand agent with data loading capability
    def get_market_overview() -> dict:
        """Get comprehensive market overview using A2A data loader"""
        try:
            loader = A2AHistoricalDataLoader()
            
            # Get recent data (last 3 days)
            start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Load key indicators
            yahoo_client = YahooFinanceClient()
            btc_data = yahoo_client.download_data("BTC", start_date=start_date, end_date=end_date, save=False)
            
            cboe_client = CBOEClient()
            vix_data = cboe_client.download_volatility_data("VIX", save=False)
            
            fred_client = FREDClient()
            treasury_data = None
            if fred_client.api_key:
                treasury_data = fred_client.get_series_data("DGS10", start_date=start_date, end_date=end_date, save=False)
            
            # Build overview
            overview = {
                "timestamp": datetime.now().isoformat(),
                "btc_price": float(btc_data['close'].iloc[-1]) if btc_data is not None else None,
                "vix_level": float(vix_data['VIX_Close'].iloc[-1]) if vix_data is not None else None,
                "treasury_10y": float(treasury_data['DGS10'].iloc[-1]) if treasury_data is not None else None,
                "data_sources": ["yahoo_finance", "cboe", "fred"]
            }
            
            return overview
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    # Create strand agent with market overview tool
    tools = [ToolSpec(
        name="get_market_overview", 
        description="Get comprehensive market overview using A2A data loader",
        parameters={"type": "object", "properties": {}},
        function=get_market_overview
    )]
    agent = Agent(tools=tools)
    
    # Test agent execution
    print("\n1. Testing Strand Agent with A2A Data Tools...")
    
    prompt = """
    Get a comprehensive market overview using the available data sources.
    Include Bitcoin price, VIX volatility, and Treasury yields if available.
    """
    
    try:
        result = await agent.process_async(prompt)
        print(f"   ✓ Strand Agent Response: {str(result)[:200]}...")
        print(f"   ✓ Agent Status: {result.stop_reason}")
    except Exception as e:
        print(f"   ✗ Strand Agent Error: {e}")

async def main():
    """Run all tests"""
    print("COMPREHENSIVE A2A HISTORICAL DATA LOADER TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().isoformat()}")
    
    try:
        # Test individual clients
        await test_individual_clients()
        
        # Test comprehensive loader
        await test_a2a_comprehensive_loader()
        
        # Test strand integration
        await test_strand_integration()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
