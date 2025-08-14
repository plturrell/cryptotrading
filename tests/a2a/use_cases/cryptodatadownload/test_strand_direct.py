#!/usr/bin/env python3
"""Test direct tool invocation with Strand"""

import os
import sys
from pathlib import Path

# Clear any API keys
os.environ.pop('OPENAI_API_KEY', None)
os.environ.pop('ANTHROPIC_API_KEY', None)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

print("Testing direct discovery without AI model...")

# Import our data management agent
from rex.a2a.agents.data_management_agent import DataManagementAgent

# Create agent
agent = DataManagementAgent()

# Test direct method call (bypassing Strand Agent)
print("\n1. Testing direct discovery method:")
result = agent._discover_cryptodatadownload_structure({
    "exchange": "binance",
    "pair": "BTCUSDT",
    "timeframe": "d"
})

print(f"Success: {result.get('success')}")
if result.get('success'):
    print(f"Columns found: {len(result.get('structure', {}).get('columns', {}))}")
    quality = result.get('sap_resource_discovery', {}).get('Governance', {}).get('QualityMetrics', {})
    print(f"Quality metrics calculated: Completeness={quality.get('Completeness', 0)}")
else:
    print(f"Error: {result.get('error')}")

# Test with our DeepSeek as the model provider
print("\n2. Testing with DeepSeek configuration:")
os.environ['DEEPSEEK_API_KEY'] = 'test-key'  # Our local AI

# Can we use DeepSeek with Strand?
print("DeepSeek integration would require custom Strand provider")