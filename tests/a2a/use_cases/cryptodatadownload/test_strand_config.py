#!/usr/bin/env python3
"""Check Strand Agent configuration"""

import os
from strands import Agent

# Test with different configurations
print("Testing Strand Agent configurations...")

# Test 1: No keys
print("\n1. No API keys:")
try:
    os.environ.pop('OPENAI_API_KEY', None)
    os.environ.pop('ANTHROPIC_API_KEY', None)
    agent = Agent()
    print("SUCCESS: Agent created without keys")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: With OpenAI key
print("\n2. With OpenAI key:")
try:
    os.environ['OPENAI_API_KEY'] = 'test-key'
    os.environ.pop('ANTHROPIC_API_KEY', None)
    agent = Agent()
    print("SUCCESS: Agent created with OpenAI key")
    print(f"Agent type: {type(agent)}")
    if hasattr(agent, 'model_provider'):
        print(f"Model provider: {agent.model_provider}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 3: With Anthropic key
print("\n3. With Anthropic key:")
try:
    os.environ.pop('OPENAI_API_KEY', None)
    os.environ['ANTHROPIC_API_KEY'] = 'test-key'
    agent = Agent()
    print("SUCCESS: Agent created with Anthropic key")
except Exception as e:
    print(f"ERROR: {e}")

# Test 4: Check tool execution without model
print("\n4. Testing tool execution:")
try:
    from strands import tool
    
    @tool
    def test_tool() -> str:
        """Test tool that doesn't need AI"""
        return "Tool executed successfully"
    
    agent = Agent(tools=[test_tool])
    # Try to call the tool directly
    print("Agent created with tool")
    
except Exception as e:
    print(f"ERROR: {e}")