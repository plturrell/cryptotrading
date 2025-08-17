"""
Blockchain integration for cryptotrading.com
Real blockchain integration using Anvil for A2A messaging and consensus
"""

from .anvil_client import AnvilA2AClient, A2AMessage, AgentRegistration, start_anvil_node, stop_anvil_node

__all__ = [
    'AnvilA2AClient',
    'A2AMessage', 
    'AgentRegistration',
    'start_anvil_node',
    'stop_anvil_node'
]