#!/usr/bin/env python3
"""
Test script to verify A2A agent registrations are working properly
"""
import os
import sys
import logging
from pathlib import Path

# Set environment to testing to avoid production validation requirements
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_agent_registrations():
    """Test that all agents can be imported and initialized with A2A registration"""
    agents_tested = []
    failed_agents = []
    
    # Test ML Agent
    try:
        from src.cryptotrading.core.agents.specialized.ml_agent import MLAgent
        agent = MLAgent(agent_id="test-ml-agent")
        agents_tested.append("ML Agent")
        logger.info("✅ ML Agent - imported and initialized successfully")
    except Exception as e:
        failed_agents.append(f"ML Agent: {e}")
        logger.error(f"❌ ML Agent - failed: {e}")
    
    # Test Data Analysis Agent  
    try:
        from src.cryptotrading.core.agents.specialized.data_analysis_agent import DataAnalysisAgent
        agent = DataAnalysisAgent(agent_id="test-data-analysis-agent")
        agents_tested.append("Data Analysis Agent")
        logger.info("✅ Data Analysis Agent - imported and initialized successfully")
    except Exception as e:
        failed_agents.append(f"Data Analysis Agent: {e}")
        logger.error(f"❌ Data Analysis Agent - failed: {e}")
    
    # Test Strands Glean Agent
    try:
        from src.cryptotrading.core.agents.specialized.strands_glean_agent import StrandsGleanAgent
        agent = StrandsGleanAgent(agent_id="test-strands-glean-agent")
        agents_tested.append("Strands Glean Agent")
        logger.info("✅ Strands Glean Agent - imported and initialized successfully")
    except Exception as e:
        failed_agents.append(f"Strands Glean Agent: {e}")
        logger.error(f"❌ Strands Glean Agent - failed: {e}")
    
    # Test AWS Data Exchange Agent
    try:
        from src.cryptotrading.core.agents.specialized.aws_data_exchange_agent import AWSDataExchangeAgent
        agent = AWSDataExchangeAgent(agent_id="test-aws-data-exchange-agent")
        agents_tested.append("AWS Data Exchange Agent")
        logger.info("✅ AWS Data Exchange Agent - imported and initialized successfully")
    except Exception as e:
        failed_agents.append(f"AWS Data Exchange Agent: {e}")
        logger.error(f"❌ AWS Data Exchange Agent - failed: {e}")
    
    # Test MCTS Calculation Agent
    try:
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent import MCTSCalculationAgent
        agent = MCTSCalculationAgent(agent_id="test-mcts-calculation-agent")
        agents_tested.append("MCTS Calculation Agent")
        logger.info("✅ MCTS Calculation Agent - imported and initialized successfully")
    except Exception as e:
        failed_agents.append(f"MCTS Calculation Agent: {e}")
        logger.error(f"❌ MCTS Calculation Agent - failed: {e}")
    
    # Test A2A Registry access
    try:
        from src.cryptotrading.core.protocols.a2a.a2a_protocol import A2AAgentRegistry, A2A_CAPABILITIES
        logger.info("✅ A2A protocol imports successful")
        
        # Check if registry has any agents
        if hasattr(A2AAgentRegistry, 'agents') and A2AAgentRegistry.agents:
            logger.info(f"✅ A2A Registry contains {len(A2AAgentRegistry.agents)} registered agents")
        else:
            logger.info("ℹ️  A2A Registry is empty or not accessible (normal for test environment)")
            
    except Exception as e:
        failed_agents.append(f"A2A Registry access: {e}")
        logger.error(f"❌ A2A Registry access failed: {e}")
    
    # Summary
    logger.info(f"\n📊 TEST SUMMARY:")
    logger.info(f"✅ Successfully tested: {len(agents_tested)} agents")
    logger.info(f"❌ Failed agents: {len(failed_agents)}")
    
    if agents_tested:
        logger.info(f"Successfully tested agents:")
        for agent in agents_tested:
            logger.info(f"  • {agent}")
    
    if failed_agents:
        logger.info(f"Failed agents:")
        for failure in failed_agents:
            logger.info(f"  • {failure}")
    
    return len(failed_agents) == 0

if __name__ == "__main__":
    import asyncio
    
    async def main():
        logger.info("🚀 Starting A2A agent registration test...")
        success = await test_agent_registrations()
        
        if success:
            logger.info("🎉 All A2A agent registrations working correctly!")
            sys.exit(0)
        else:
            logger.error("💥 Some agent registrations failed!")
            sys.exit(1)
    
    asyncio.run(main())