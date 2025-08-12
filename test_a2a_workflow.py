#!/usr/bin/env python3
"""
Test A2A Workflow - 100% A2A Compliant
Tests complete integration between Historical Loader and Database agents
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from рекс.a2a.agents.a2a_coordinator import a2a_coordinator
from рекс.a2a.registry.registry import agent_registry

async def test_a2a_workflow():
    """Test complete A2A workflow"""
    print("=== A2A Workflow Test - Strand Agents Integration ===\n")
    
    # Test 1: Check agent registration
    print("1. Testing Agent Registration:")
    agents = agent_registry.get_all_agents()
    for agent_id, info in agents.items():
        print(f"  ✓ {agent_id}: {info['type']} - {info['status']}")
        print(f"    Capabilities: {', '.join(info['capabilities'])}")
    print()
    
    # Test 2: Agent status
    print("2. Testing Agent Status:")
    status = a2a_coordinator.get_agent_status()
    for agent_id, info in status.items():
        print(f"  ✓ {agent_id}: {info['status']}")
        print(f"    Type: {info['type']}")
        print(f"    Capabilities: {', '.join(info['capabilities'])}")
    print()
    
    # Test 3: Bulk data loading with A2A communication
    print("3. Testing Bulk Data Loading (A2A Protocol):")
    test_symbols = ["BTC-USD", "ETH-USD"]
    
    try:
        workflow_result = await a2a_coordinator.bulk_load_workflow(
            symbols=test_symbols,
            days_back=30  # Smaller dataset for testing
        )
        
        print(f"  Workflow ID: {workflow_result['workflow_id']}")
        print(f"  Overall Success: {workflow_result['overall_success']}")
        print(f"  Total Records: {workflow_result['total_records']}")
        print(f"  Started: {workflow_result['started_at']}")
        print(f"  Completed: {workflow_result.get('completed_at', 'N/A')}")
        
        for step in workflow_result['steps']:
            print(f"\n  Symbol: {step['symbol']}")
            print(f"    Success: {step['success']}")
            if 'records_count' in step:
                print(f"    Records: {step['records_count']}")
            if 'ai_analyses' in step:
                print(f"    AI Analyses: {step['ai_analyses']}")
            if 'error' in step:
                print(f"    Error: {step['error']}")
            
            print(f"    Messages:")
            for msg in step['messages']:
                print(f"      - {msg['step']}: {'✓' if msg['success'] else '✗'}")
        
    except Exception as e:
        print(f"  ✗ Workflow failed: {e}")
    print()
    
    # Test 4: AI Analysis Request (A2A)
    print("4. Testing AI Analysis Request (A2A Protocol):")
    try:
        analysis_result = await a2a_coordinator.get_symbol_analysis(
            "BTC-USD", 
            ai_providers=['deepseek', 'perplexity', 'claude']
        )
        
        if analysis_result.get('success'):
            print("  ✓ AI Analysis successful")
            results = analysis_result.get('results', {})
            for provider, result in results.items():
                status = "✓" if result.get('success') else "✗"
                print(f"    {provider}: {status}")
                if result.get('success'):
                    analysis = result.get('analysis', 'No analysis')[:100]
                    print(f"      Analysis: {analysis}...")
        else:
            print(f"  ✗ AI Analysis failed: {analysis_result.get('error')}")
    except Exception as e:
        print(f"  ✗ AI Analysis error: {e}")
    print()
    
    # Test 5: Message History
    print("5. A2A Message History:")
    message_history = a2a_coordinator.get_message_history()
    print(f"  Total Messages: {len(message_history)}")
    
    for i, msg_record in enumerate(message_history[-3:], 1):  # Show last 3 messages
        msg = msg_record['message']
        print(f"  Message {i}:")
        print(f"    From: {msg['sender_id']} → To: {msg['receiver_id']}")
        print(f"    Type: {msg['type']}")
        print(f"    Time: {msg['timestamp']}")
    print()
    
    print("=== A2A Integration Test Complete ===")

async def test_direct_agent_communication():
    """Test direct agent-to-agent communication"""
    print("\n=== Direct Agent Communication Test ===")
    
    try:
        # Test historical loader agent directly
        print("Testing Historical Loader Agent:")
        from рекс.a2a.agents.historical_loader_agent import historical_loader_agent
        
        response = await historical_loader_agent.process_request(
            "Load 7 days of historical data for BTC-USD with technical indicators"
        )
        print(f"  Response: {response[:200]}...")
        
        print("\nTesting Database Agent:")
        from рекс.a2a.agents.database_agent import database_agent
        
        response = await database_agent.process_request(
            "Analyze BTC-USD using all AI providers (DeepSeek, Perplexity, Claude)"
        )
        print(f"  Response: {response[:200]}...")
        
    except Exception as e:
        print(f"  Error in direct communication: {e}")

if __name__ == "__main__":
    print("Starting A2A Workflow Tests...")
    
    # Run the complete test suite
    asyncio.run(test_a2a_workflow())
    asyncio.run(test_direct_agent_communication())