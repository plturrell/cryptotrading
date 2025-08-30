#!/usr/bin/env python3
"""
Syntax validation script for A2A agent registrations
Verifies that the A2A registration code was added correctly without full agent initialization
"""
import ast
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_a2a_registration_syntax(file_path: Path, agent_name: str):
    """Check if A2A registration code is syntactically correct in the file"""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        tree = ast.parse(content)
        
        # Check for A2A imports
        has_a2a_import = 'from ...protocols.a2a.a2a_protocol import' in content
        has_registry_import = 'A2AAgentRegistry' in content
        has_capabilities_import = 'A2A_CAPABILITIES' in content
        
        # Check for registration call
        has_registration_call = 'A2AAgentRegistry.register_agent' in content
        has_capabilities_get = 'A2A_CAPABILITIES.get' in content
        
        logger.info(f"✅ {agent_name}: Syntax valid")
        logger.info(f"  📦 A2A Import: {'✅' if has_a2a_import else '❌'}")
        logger.info(f"  📦 Registry Import: {'✅' if has_registry_import else '❌'}")
        logger.info(f"  📦 Capabilities Import: {'✅' if has_capabilities_import else '❌'}")
        logger.info(f"  🔗 Registration Call: {'✅' if has_registration_call else '❌'}")
        logger.info(f"  🔧 Capabilities Get: {'✅' if has_capabilities_get else '❌'}")
        
        # Check if all required elements are present
        all_present = all([
            has_a2a_import,
            has_registry_import, 
            has_capabilities_import,
            has_registration_call,
            has_capabilities_get
        ])
        
        return all_present
        
    except SyntaxError as e:
        logger.error(f"❌ {agent_name}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        logger.error(f"❌ {agent_name}: Error checking syntax: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🔍 Verifying A2A registration syntax in all modified agents...")
    
    project_root = Path(__file__).parent.parent
    agent_files = [
        (project_root / "src/cryptotrading/core/agents/specialized/ml_agent.py", "ML Agent"),
        (project_root / "src/cryptotrading/core/agents/specialized/data_analysis_agent.py", "Data Analysis Agent"),
        (project_root / "src/cryptotrading/core/agents/specialized/strands_glean_agent.py", "Strands Glean Agent"),
        (project_root / "src/cryptotrading/core/agents/specialized/aws_data_exchange_agent.py", "AWS Data Exchange Agent"),
        (project_root / "src/cryptotrading/core/agents/specialized/mcts_calculation_agent.py", "MCTS Calculation Agent"),
    ]
    
    passed_count = 0
    failed_count = 0
    
    for file_path, agent_name in agent_files:
        if file_path.exists():
            if check_a2a_registration_syntax(file_path, agent_name):
                passed_count += 1
                logger.info(f"✅ {agent_name}: A2A registration code complete\n")
            else:
                failed_count += 1
                logger.error(f"❌ {agent_name}: A2A registration code incomplete\n")
        else:
            logger.error(f"❌ {agent_name}: File not found: {file_path}\n")
            failed_count += 1
    
    # Summary
    logger.info(f"📊 SYNTAX VALIDATION SUMMARY:")
    logger.info(f"✅ Passed: {passed_count} agents")
    logger.info(f"❌ Failed: {failed_count} agents")
    
    if failed_count == 0:
        logger.info("🎉 All A2A registration code syntax is valid!")
        return True
    else:
        logger.error("💥 Some agents have incomplete A2A registration code!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)