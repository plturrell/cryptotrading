#!/usr/bin/env python3
"""
Simple agent registration script using Agent Manager directly.
Registers agents without full initialization requirements.
"""

import asyncio
import logging
from typing import Dict, Any, List
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set minimal environment
os.environ['ENVIRONMENT'] = 'development'
os.environ['DATABASE_URL'] = 'sqlite:///test.db'
os.environ['JWT_SECRET'] = 'test_secret'

from src.cryptotrading.core.protocols.a2a.a2a_protocol import A2AAgentRegistry, A2A_CAPABILITIES


# Agents to register
AGENTS_TO_REGISTER = {
    "mcts_calculation_agent": {
        "type": "mcts_calculation",
        "capabilities": [
            "monte_carlo_simulation", "strategy_optimization",
            "risk_assessment", "portfolio_optimization",
            "backtesting", "performance_analysis",
            "calculation_metrics", "general_optimization",
            "algorithm_performance"
        ]
    },
    "technical_analysis_agent": {
        "type": "technical_analysis", 
        "capabilities": [
            "momentum_analysis", "volume_analysis", "pattern_recognition",
            "indicator_calculation", "trend_analysis", "support_resistance",
            "technical_indicators", "market_sentiment"
        ]
    },
    "ml_agent": {
        "type": "ml_agent",
        "capabilities": [
            "model_training", "prediction", "feature_engineering",
            "model_evaluation", "hyperparameter_tuning",
            "ml_training", "ml_calculations", "ensemble_methods"
        ]
    },
    "trading_algorithm_agent": {
        "type": "trading_algorithm",
        "capabilities": [
            "grid_trading", "dollar_cost_averaging", "arbitrage_detection",
            "momentum_trading", "mean_reversion", "signal_generation",
            "strategy_analysis", "backtesting", "risk_management",
            "portfolio_optimization", "scalping", "market_making",
            "breakout_trading", "ml_predictions", "multi_strategy_management"
        ]
    },
    "data_analysis_agent": {
        "type": "data_analysis",
        "capabilities": [
            "data_processing", "statistical_analysis", "pattern_recognition",
            "anomaly_detection", "correlation_analysis", "trend_analysis",
            "data_quality_assessment", "feature_extraction",
            "data_visualization", "report_generation"
        ]
    },
    "feature_store_agent": {
        "type": "feature_store",
        "capabilities": [
            "feature_storage", "feature_retrieval", "feature_versioning",
            "feature_validation", "feature_transformation", "feature_serving",
            "metadata_management", "lineage_tracking",
            "feature_monitoring", "feature_discovery"
        ]
    },
    "strands_glean_agent": {
        "type": "glean_agent",
        "capabilities": [
            "code_analysis", "dependency_tracking", "impact_analysis",
            "code_search", "documentation_generation",
            "code_indexing", "semantic_search"
        ]
    },
    "agent_manager": {
        "type": "agent_manager",
        "capabilities": [
            "agent_registration", "mcp_segregation_enforcement",
            "skill_card_compliance", "agent_discovery",
            "compliance_monitoring", "lifecycle_management"
        ]
    }
}


def register_agent_simple(agent_id: str, agent_info: Dict[str, Any]) -> bool:
    """Register an agent with the A2A registry."""
    try:
        # Get capabilities from A2A_CAPABILITIES or use provided ones
        capabilities = A2A_CAPABILITIES.get(agent_id, agent_info["capabilities"])
        
        # Register with A2A protocol
        success = A2AAgentRegistry.register_agent(
            agent_id=agent_id,
            capabilities=capabilities
        )
        
        if success:
            logger.info(f"✅ {agent_id} registered successfully")
            logger.info(f"   Type: {agent_info['type']}")
            logger.info(f"   Capabilities: {len(capabilities)}")
        else:
            logger.error(f"❌ {agent_id} registration failed")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Error registering {agent_id}: {e}")
        return False


def check_existing_registrations():
    """Check which agents are already registered."""
    logger.info("\n" + "="*60)
    logger.info("Checking Existing Registrations")
    logger.info("="*60)
    
    all_agents = A2AAgentRegistry.get_all_agents()
    
    if all_agents:
        logger.info(f"Found {len(all_agents)} registered agents:")
        for agent_id, info in all_agents.items():
            status = info.get('status')
            if hasattr(status, 'value'):
                status = status.value
            logger.info(f"  • {agent_id}")
            logger.info(f"    Status: {status}")
            logger.info(f"    Capabilities: {len(info.get('capabilities', []))}")
    else:
        logger.info("No agents currently registered")
    
    return all_agents


def register_all_agents():
    """Register all agents."""
    logger.info("\n" + "="*60)
    logger.info("Registering A2A Agents")
    logger.info("="*60)
    
    # Check existing registrations
    existing = check_existing_registrations()
    existing_ids = set(existing.keys())
    
    # Register each agent
    success_count = 0
    failed_count = 0
    
    for agent_id, agent_info in AGENTS_TO_REGISTER.items():
        logger.info(f"\n📝 Registering {agent_id}...")
        
        if agent_id in existing_ids:
            logger.info(f"   ⚠️  Already registered, updating...")
            # Update registration
            A2AAgentRegistry.update_agent_status(agent_id, "active")
        
        if register_agent_simple(agent_id, agent_info):
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Registration Summary")
    logger.info("="*60)
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"❌ Failed: {failed_count}")
    
    # Show all registered agents
    logger.info("\n" + "="*60)
    logger.info("All Registered Agents")
    logger.info("="*60)
    
    all_agents = A2AAgentRegistry.get_all_agents()
    for agent_id, info in all_agents.items():
        status = info.get('status')
        if hasattr(status, 'value'):
            status = status.value
        logger.info(f"  • {agent_id}")
        logger.info(f"    Status: {status}")
        logger.info(f"    Capabilities: {', '.join(info.get('capabilities', [])[:3])}...")
    
    # Test capability search
    logger.info("\n" + "="*60)
    logger.info("Testing Capability Search")
    logger.info("="*60)
    
    test_capabilities = [
        "grid_trading",
        "monte_carlo_simulation",
        "feature_storage",
        "agent_registration"
    ]
    
    for capability in test_capabilities:
        agents = A2AAgentRegistry.find_agents_by_capability(capability)
        logger.info(f"  {capability}: {', '.join(agents) if agents else 'No agents found'}")


def main():
    """Main function."""
    logger.info("="*60)
    logger.info("A2A Agent Registration (Simple)")
    logger.info("="*60)
    
    # Register all agents
    register_all_agents()
    
    logger.info("\n" + "="*60)
    logger.info("✅ Registration Complete!")
    logger.info("="*60)
    logger.info("\nNote: This is A2A protocol registration only.")
    logger.info("For blockchain registration, use the full Agent Manager CLI.")


if __name__ == "__main__":
    main()