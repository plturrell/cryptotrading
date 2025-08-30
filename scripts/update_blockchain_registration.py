#!/usr/bin/env python3
"""
Script to update all remaining A2A agents with blockchain registration
"""

import os
import re
import sys
from pathlib import Path

def add_blockchain_import(file_path: Path) -> bool:
    """Add blockchain registration import to an agent file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if import already exists
        if 'from ...protocols.a2a.blockchain_registration import EnhancedA2AAgentRegistry' in content:
            print(f"✅ {file_path.name}: Blockchain import already exists")
            return True
        
        # Find the A2A protocol import line
        a2a_import_pattern = r'from \.\.\.protocols\.a2a\.a2a_protocol import A2A_CAPABILITIES, A2AAgentRegistry'
        match = re.search(a2a_import_pattern, content)
        
        if match:
            # Add the blockchain import right after
            new_import = match.group(0) + '\nfrom ...protocols.a2a.blockchain_registration import EnhancedA2AAgentRegistry'
            content = content.replace(match.group(0), new_import)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"✅ {file_path.name}: Added blockchain import")
            return True
        else:
            print(f"⚠️  {file_path.name}: Could not find A2A protocol import")
            return False
            
    except Exception as e:
        print(f"❌ {file_path.name}: Error adding import: {e}")
        return False

def update_blockchain_registration(file_path: Path, agent_type: str) -> bool:
    """Update the A2A registration to use blockchain"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if already updated
        if 'EnhancedA2AAgentRegistry.register_agent_with_blockchain' in content:
            print(f"✅ {file_path.name}: Blockchain registration already updated")
            return True
        
        # Find the A2AAgentRegistry.register_agent pattern
        old_pattern = r'(\s+)# Register with A2A Agent Registry\n(\s+)capabilities = A2A_CAPABILITIES\.get\(agent_id, \[\]\)\n(\s+)A2AAgentRegistry\.register_agent\(agent_id, capabilities, self\)\n(\s+)logger\.info\(f"([^"]*Agent) \{agent_id\} registered with A2A registry"\)'
        
        match = re.search(old_pattern, content, re.MULTILINE)
        
        if match:
            indent = match.group(1)
            agent_name = match.group(5)
            
            # Create the new blockchain registration code
            new_registration = f"""{indent}# Register with A2A Agent Registry (including blockchain)
{indent}capabilities = A2A_CAPABILITIES.get(agent_id, [])
{indent}mcp_tools = list(self.mcp_handlers.keys()) if hasattr(self, 'mcp_handlers') else []
{indent}
{indent}# Try blockchain registration, fallback to local only
{indent}try:
{indent}    import asyncio
{indent}    asyncio.create_task(
{indent}        EnhancedA2AAgentRegistry.register_agent_with_blockchain(
{indent}            agent_id=agent_id,
{indent}            capabilities=capabilities,
{indent}            agent_instance=self,
{indent}            agent_type="{agent_type}",
{indent}            mcp_tools=mcp_tools
{indent}        )
{indent}    )
{indent}    logger.info(f"{agent_name} {{agent_id}} blockchain registration initiated")
{indent}except Exception as e:
{indent}    # Fallback to local registration only
{indent}    A2AAgentRegistry.register_agent(agent_id, capabilities, self)
{indent}    logger.warning(f"{agent_name} {{agent_id}} registered locally only (blockchain failed: {{e}})")"""
            
            content = content.replace(match.group(0), new_registration)
            
            with open(file_path, 'w') as f:
                f.write(content)
                
            print(f"✅ {file_path.name}: Updated blockchain registration")
            return True
        else:
            print(f"⚠️  {file_path.name}: Could not find registration pattern")
            return False
            
    except Exception as e:
        print(f"❌ {file_path.name}: Error updating registration: {e}")
        return False

def main():
    """Main update function"""
    project_root = Path(__file__).parent.parent
    agents_dir = project_root / "src/cryptotrading/core/agents/specialized"
    
    # Agent files and their types
    agents_to_update = [
        ("strands_glean_agent.py", "strands-glean"),
        ("aws_data_exchange_agent.py", "aws_data_exchange"), 
        ("mcts_calculation_agent.py", "mcts_calculation")
    ]
    
    print("🔄 Updating A2A agents with blockchain registration...")
    
    updated_count = 0
    failed_count = 0
    
    for agent_file, agent_type in agents_to_update:
        file_path = agents_dir / agent_file
        
        if not file_path.exists():
            print(f"⚠️  {agent_file}: File not found")
            failed_count += 1
            continue
            
        print(f"\n📝 Processing {agent_file}...")
        
        # Add import
        import_success = add_blockchain_import(file_path)
        
        # Update registration
        registration_success = update_blockchain_registration(file_path, agent_type)
        
        if import_success and registration_success:
            updated_count += 1
        else:
            failed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"✅ Successfully updated: {updated_count} agents")
    print(f"❌ Failed: {failed_count} agents")
    
    if failed_count == 0:
        print("🎉 All agents successfully updated with blockchain registration!")
        return 0
    else:
        print("💥 Some agents failed to update")
        return 1

if __name__ == "__main__":
    sys.exit(main())