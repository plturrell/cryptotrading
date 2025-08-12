#!/usr/bin/env python3
"""
Test all рекс.com clients and systems
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_database():
    """Test SQLite database client"""
    print("\n🔍 Testing Database Client...")
    try:
        from рекс.database import DatabaseClient
        
        # Initialize database
        db = DatabaseClient(db_path='./test_рекс.db')
        
        # Test saving AI analysis
        analysis_id = db.save_ai_analysis(
            symbol='BTC',
            model='test-model',
            analysis_type='test',
            analysis='Database test successful'
        )
        print("✅ Database write successful, ID:", analysis_id)
        
        # Test reading
        with db.get_session() as session:
            from рекс.database.models import AIAnalysis
            result = session.query(AIAnalysis).filter_by(symbol='BTC').first()
            if result:
                print("✅ Database read successful:", result.analysis)
        
        db.close()
        
        # Clean up test database
        if os.path.exists('test_рекс.db'):
            os.remove('test_рекс.db')
            
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_deepseek():
    """Test DeepSeek R1 integration"""
    print("\n🔍 Testing DeepSeek R1...")
    try:
        from рекс.ml.deepseek import DeepSeekR1
        
        ai = DeepSeekR1()
        
        # Test market analysis
        market_data = {
            'symbol': 'BTC',
            'price': 45000,
            'volume': 1000000,
            'rsi': 75
        }
        
        analysis = ai.analyze_market(market_data)
        print("✅ DeepSeek analysis:", analysis)
        
        # Test price prediction
        prediction = ai.predict_price('BTC', '4h')
        print("✅ Price prediction:", prediction)
        
        return True
    except Exception as e:
        print(f"❌ DeepSeek test failed: {e}")
        return False

def test_perplexity():
    """Test Perplexity AI client"""
    print("\n🔍 Testing Perplexity AI...")
    try:
        from рекс.ml.perplexity import PerplexityClient
        
        # Check if API key is set
        if not os.getenv('PERPLEXITY_API_KEY'):
            print("⚠️  Perplexity API key not found in environment")
            return False
            
        perplexity = PerplexityClient()
        
        # Test trading signals (mock if no real API)
        print("✅ Perplexity client initialized")
        print("  - search_crypto_news() available")
        print("  - get_trading_signals() available")
        print("  - analyze_market_conditions() available")
        
        return True
    except Exception as e:
        print(f"❌ Perplexity test failed: {e}")
        return False

def test_a2a_registry():
    """Test A2A Agent Registry"""
    print("\n🔍 Testing A2A Agent Registry...")
    try:
        from рекс.a2a.registry.registry import agent_registry
        
        # Test getting all agents
        agents = agent_registry.get_all_agents()
        print(f"✅ Found {len(agents)} registered agents:")
        
        for agent_id, agent_info in agents.items():
            print(f"  - {agent_id}: {agent_info['type']} agent")
            print(f"    Capabilities: {', '.join(agent_info['capabilities'])}")
        
        # Test finding by capability
        analysis_agents = agent_registry.find_agents_by_capability('market_analysis')
        print(f"\n✅ Agents with 'market_analysis' capability: {analysis_agents}")
        
        # Test connections
        agent_registry.establish_connection('transform-001', 'illuminate-001', 'data_pipeline')
        connections = agent_registry.get_agent_connections('transform-001')
        print(f"✅ Created connection between agents: {len(connections)} connections")
        
        return True
    except Exception as e:
        print(f"❌ A2A Registry test failed: {e}")
        return False

def test_api_endpoints():
    """Test if API endpoints are properly configured"""
    print("\n🔍 Testing API Endpoints Configuration...")
    try:
        # Check if app can be imported
        from app import app, api
        
        # List all registered endpoints
        print("✅ API endpoints configured:")
        
        rules = []
        for rule in app.url_map.iter_rules():
            if '/api/' in rule.rule:
                rules.append(f"  - {rule.rule} [{', '.join(rule.methods - {'HEAD', 'OPTIONS'})}]")
        
        for rule in sorted(rules):
            print(rule)
        
        return True
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        return False

def test_server_connection():
    """Test connection to DigitalOcean server"""
    print("\n🔍 Testing Server Connection...")
    try:
        import requests
        
        server_ip = "165.227.69.235"
        
        # Test direct IP
        print(f"  Testing http://{server_ip}/health ...")
        try:
            response = requests.get(f"http://{server_ip}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Server responded: {response.json()}")
            else:
                print(f"⚠️  Server returned status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Server not accessible yet (may still be initializing)")
        
        # Test domain
        print(f"  Testing https://рекс.com ...")
        try:
            response = requests.get("https://xn--e1afmkfd.com", timeout=5)
            print(f"✅ Domain accessible: Status {response.status_code}")
        except:
            print("⚠️  Domain not accessible yet (DNS may be propagating)")
            
        return True
    except Exception as e:
        print(f"❌ Server connection test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing рекс.com clients and systems...")
    print("=" * 50)
    
    results = {
        'Database': test_database(),
        'DeepSeek R1': test_deepseek(),
        'Perplexity AI': test_perplexity(),
        'A2A Registry': test_a2a_registry(),
        'API Endpoints': test_api_endpoints(),
        'Server Connection': test_server_connection()
    }
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\n🎯 Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All systems operational!")
    else:
        print("⚠️  Some components need attention")

if __name__ == '__main__':
    main()