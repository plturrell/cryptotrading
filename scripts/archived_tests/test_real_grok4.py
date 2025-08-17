"""
Test real Grok4 integration
Set GROK4_API_KEY environment variable to test with real API
"""
import asyncio
import os
from src.cryptotrading.core.ai.grok4_client import Grok4Client

async def test_real_grok4():
    print("=== TESTING REAL GROK4 INTEGRATION ===")
    print()
    
    # Check if API key is set
    api_key = os.getenv('GROK4_API_KEY')
    if not api_key:
        print("❌ GROK4_API_KEY not set - using mock mode")
        print("To test real AI:")
        print("export GROK4_API_KEY='your_x_ai_api_key'")
        print()
    else:
        print("✅ GROK4_API_KEY found - will use real X.AI API")
        print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
        print()
    
    # Initialize client
    client = Grok4Client()
    print(f"Client mode: {'REAL AI' if not client.use_mock else 'MOCK'}")
    print(f"Base URL: {client.base_url}")
    print()
    
    try:
        # Test sentiment analysis
        print("Testing market sentiment analysis...")
        insights = await client.analyze_market_sentiment(['BTC', 'ETH'])
        
        print(f"✅ Received {len(insights)} insights")
        for insight in insights:
            print(f"   {insight.symbol}: {insight.recommendation} (score: {insight.score:.2f}, confidence: {insight.confidence:.2f})")
            print(f"      Reasoning: {insight.reasoning}")
        print()
        
        # Test risk assessment
        print("Testing portfolio risk assessment...")
        portfolio = {'BTC': 5000, 'ETH': 3000, 'ADA': 1000}
        risk = await client.assess_trading_risk(portfolio)
        
        print(f"✅ Risk assessment completed")
        print(f"   Overall risk score: {risk.get('overall_risk_score', 'N/A')}")
        print(f"   Risk level: {risk.get('risk_level', 'N/A')}")
        print(f"   Confidence: {risk.get('confidence', 'N/A')}")
        print()
        
        # Test predictions
        print("Testing market predictions...")
        predictions = await client.predict_market_movement(['BTC'])
        
        print(f"✅ Market predictions completed")
        print(f"   Predictions for {len(predictions)} symbols")
        print()
        
        await client.close()
        
        if client.use_mock:
            print("🔄 MOCK MODE RESULTS:")
            print("   All responses are simulated for testing")
            print("   Set GROK4_API_KEY for real AI analysis")
        else:
            print("🚀 REAL AI RESULTS:")
            print("   All responses from actual Grok4 model")
            print("   Real market intelligence provided!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_real_grok4())
    if success:
        print("\n🎉 Grok4 integration test successful!")
    else:
        print("\n💥 Grok4 integration test failed!")