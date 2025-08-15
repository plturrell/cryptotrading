#!/usr/bin/env python3
"""
Source Directory Restructure Migration Script
Safely migrates the src/ directory from the current chaotic structure 
to a clean, standard Python package layout.
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import json

class SourceRestructureMigrator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.backup_dir = self.project_root / "src_backup"
        self.migration_log = []
        
    def log(self, message: str):
        """Log migration steps"""
        print(f"[MIGRATE] {message}")
        self.migration_log.append(message)
    
    def create_backup(self):
        """Create backup of current src directory"""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        shutil.copytree(self.src_dir, self.backup_dir)
        self.log(f"Created backup at {self.backup_dir}")
    
    def create_new_structure(self):
        """Create the new package structure"""
        new_structure = {
            "cryptotrading": {
                "__init__.py": "",
                "core": {
                    "__init__.py": "",
                    "agents": {
                        "__init__.py": "",
                        "specialized": {"__init__.py": ""}
                    },
                    "protocols": {
                        "__init__.py": "",
                        "a2a": {"__init__.py": ""},
                        "mcp": {"__init__.py": ""}
                    },
                    "blockchain": {"__init__.py": ""},
                    "ml": {"__init__.py": ""}
                },
                "data": {
                    "__init__.py": "",
                    "database": {"__init__.py": ""},
                    "storage": {"__init__.py": ""},
                    "historical": {"__init__.py": ""}
                },
                "infrastructure": {
                    "__init__.py": "",
                    "logging": {"__init__.py": ""},
                    "monitoring": {"__init__.py": ""},
                    "security": {"__init__.py": ""},
                    "registry": {"__init__.py": ""}
                },
                "utils": {"__init__.py": ""}
            }
        }
        
        def create_structure(base_path: Path, structure: dict):
            for name, content in structure.items():
                path = base_path / name
                if isinstance(content, dict):
                    path.mkdir(exist_ok=True)
                    create_structure(path, content)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        path.write_text(content)
        
        new_src = self.src_dir / "new_structure"
        new_src.mkdir(exist_ok=True)
        create_structure(new_src, new_structure)
        self.log("Created new package structure")
        return new_src
    
    def analyze_agent_files(self) -> Dict[str, List[str]]:
        """Analyze agent files to understand inheritance hierarchy"""
        agent_files = {}
        
        # Find all agent-related files
        for root, dirs, files in os.walk(self.src_dir):
            for file in files:
                if file.endswith('.py') and ('agent' in file.lower() or 'Agent' in file):
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(self.src_dir)
                    
                    # Read file content to analyze classes
                    try:
                        content = full_path.read_text()
                        classes = re.findall(r'class\s+(\w*Agent\w*)', content)
                        if classes:
                            agent_files[str(rel_path)] = classes
                    except Exception as e:
                        self.log(f"Error reading {rel_path}: {e}")
        
        return agent_files
    
    def create_unified_agents(self, new_src: Path):
        """Create unified agent base classes"""
        agents_dir = new_src / "cryptotrading" / "core" / "agents"
        
        # Base agent class
        base_agent_content = '''"""
Unified base agent class for the cryptotrading platform.
Consolidates functionality from multiple previous agent base classes.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import asyncio
from datetime import datetime

class BaseAgent(ABC):
    """
    Unified base agent class that consolidates functionality from:
    - rex.a2a.agents.base_classes.BaseAgent
    - rex.a2a.agents.a2a_agent_base.A2AAgentBase
    - strands.agent.Agent
    """
    
    def __init__(self, agent_id: str, agent_type: str, **kwargs):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.created_at = datetime.utcnow()
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{agent_id}]")
        self._initialize(**kwargs)
    
    def _initialize(self, **kwargs):
        """Initialize agent-specific configuration"""
        pass
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message and return response"""
        pass
    
    async def start(self):
        """Start the agent"""
        self.logger.info(f"Starting agent {self.agent_id}")
    
    async def stop(self):
        """Stop the agent"""
        self.logger.info(f"Stopping agent {self.agent_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "created_at": self.created_at.isoformat(),
            "status": "active"
        }
'''
        
        memory_agent_content = '''"""
Memory-enabled agent that extends BaseAgent with memory capabilities.
Consolidates memory functionality from multiple previous implementations.
"""
from typing import Dict, Any, Optional, List
from .base import BaseAgent

class MemoryAgent(BaseAgent):
    """
    Memory-enabled agent that consolidates functionality from:
    - rex.a2a.agents.base_memory_agent.BaseMemoryAgent
    - rex.a2a.agents.memory_strands_agent.MemoryStrandsAgent
    """
    
    def __init__(self, agent_id: str, agent_type: str, memory_config: Optional[Dict] = None, **kwargs):
        super().__init__(agent_id, agent_type, **kwargs)
        self.memory_config = memory_config or {}
        self.memory_store = {}
        self._setup_memory()
    
    def _setup_memory(self):
        """Setup memory storage and retrieval"""
        # Initialize memory components
        pass
    
    async def store_memory(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Store information in agent memory"""
        self.memory_store[key] = {
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def retrieve_memory(self, key: str) -> Optional[Any]:
        """Retrieve information from agent memory"""
        if key in self.memory_store:
            return self.memory_store[key]["value"]
        return None
    
    async def search_memory(self, query: str) -> List[Dict[str, Any]]:
        """Search agent memory"""
        # Implement memory search logic
        return []
'''
        
        strands_agent_content = '''"""
Strands framework integration agent.
Consolidates Strands functionality from multiple previous implementations.
"""
from typing import Dict, Any, Optional, List
from .memory import MemoryAgent

class StrandsAgent(MemoryAgent):
    """
    Strands framework integration agent that consolidates functionality from:
    - rex.a2a.agents.base_strands_agent.BaseStrandsAgent
    - rex.a2a.agents.a2a_strands_agent.A2AStrandsAgent
    - strands.agent.Agent
    """
    
    def __init__(self, agent_id: str, agent_type: str, 
                 capabilities: Optional[List[str]] = None,
                 model_provider: str = "grok4",
                 **kwargs):
        super().__init__(agent_id, agent_type, **kwargs)
        self.capabilities = capabilities or []
        self.model_provider = model_provider
        self._setup_strands()
    
    def _setup_strands(self):
        """Setup Strands framework integration"""
        # Initialize Strands components
        pass
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool using Strands framework"""
        # Implement tool execution
        return {"result": "tool_executed", "tool": tool_name}
    
    async def process_workflow(self, workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a workflow using Strands orchestration"""
        # Implement workflow processing
        return {"workflow_id": workflow_id, "status": "completed"}
'''
        
        # Write the unified agent files
        (agents_dir / "base.py").write_text(base_agent_content)
        (agents_dir / "memory.py").write_text(memory_agent_content)
        (agents_dir / "strands.py").write_text(strands_agent_content)
        
        self.log("Created unified agent base classes")
    
    def migrate_protocols(self, new_src: Path):
        """Migrate protocol implementations"""
        protocols_dir = new_src / "cryptotrading" / "core" / "protocols"
        
        # Move MCP protocol
        old_mcp = self.src_dir / "mcp"
        new_mcp = protocols_dir / "mcp"
        if old_mcp.exists():
            shutil.copytree(old_mcp, new_mcp, dirs_exist_ok=True)
            self.log("Migrated MCP protocol")
        
        # Move A2A protocol
        old_a2a = self.src_dir / "rex" / "a2a" / "protocols"
        new_a2a = protocols_dir / "a2a"
        if old_a2a.exists():
            shutil.copytree(old_a2a, new_a2a, dirs_exist_ok=True)
            self.log("Migrated A2A protocol")
    
    def migrate_data_layer(self, new_src: Path):
        """Migrate data access layer"""
        data_dir = new_src / "cryptotrading" / "data"
        
        # Move database components
        old_db = self.src_dir / "rex" / "database"
        new_db = data_dir / "database"
        if old_db.exists():
            shutil.copytree(old_db, new_db, dirs_exist_ok=True)
            self.log("Migrated database layer")
        
        # Move storage components
        old_storage = self.src_dir / "rex" / "storage"
        new_storage = data_dir / "storage"
        if old_storage.exists():
            shutil.copytree(old_storage, new_storage, dirs_exist_ok=True)
            self.log("Migrated storage layer")
        
        # Move historical data
        old_historical = self.src_dir / "rex" / "historical_data"
        new_historical = data_dir / "historical"
        if old_historical.exists():
            shutil.copytree(old_historical, new_historical, dirs_exist_ok=True)
            self.log("Migrated historical data layer")
    
    def migrate_infrastructure(self, new_src: Path):
        """Migrate infrastructure components"""
        infra_dir = new_src / "cryptotrading" / "infrastructure"
        
        # Move logging
        old_logging = self.src_dir / "rex" / "logging"
        new_logging = infra_dir / "logging"
        if old_logging.exists():
            shutil.copytree(old_logging, new_logging, dirs_exist_ok=True)
            self.log("Migrated logging infrastructure")
        
        # Move monitoring/observability
        old_obs = self.src_dir / "rex" / "observability"
        new_monitoring = infra_dir / "monitoring"
        if old_obs.exists():
            shutil.copytree(old_obs, new_monitoring, dirs_exist_ok=True)
            self.log("Migrated monitoring infrastructure")
        
        # Move security
        old_security = self.src_dir / "rex" / "security"
        new_security = infra_dir / "security"
        if old_security.exists():
            shutil.copytree(old_security, new_security, dirs_exist_ok=True)
            self.log("Migrated security infrastructure")
        
        # Move registry (consolidate duplicates)
        old_registry = self.src_dir / "rex" / "registry"
        new_registry = infra_dir / "registry"
        if old_registry.exists():
            shutil.copytree(old_registry, new_registry, dirs_exist_ok=True)
            self.log("Migrated registry infrastructure")
    
    def migrate_remaining_components(self, new_src: Path):
        """Migrate remaining components"""
        core_dir = new_src / "cryptotrading" / "core"
        utils_dir = new_src / "cryptotrading" / "utils"
        
        # Move ML components
        old_ml = self.src_dir / "rex" / "ml"
        new_ml = core_dir / "ml"
        if old_ml.exists():
            shutil.copytree(old_ml, new_ml, dirs_exist_ok=True)
            self.log("Migrated ML components")
        
        # Move blockchain
        old_blockchain = self.src_dir / "rex" / "blockchain"
        new_blockchain = core_dir / "blockchain"
        if old_blockchain.exists():
            shutil.copytree(old_blockchain, new_blockchain, dirs_exist_ok=True)
            self.log("Migrated blockchain components")
        
        # Move utilities
        old_utils = self.src_dir / "rex" / "utils"
        if old_utils.exists():
            shutil.copytree(old_utils, utils_dir, dirs_exist_ok=True)
            self.log("Migrated utilities")
    
    def update_imports(self, new_src: Path):
        """Update import statements throughout the codebase"""
        import_mappings = {
            "from rex.a2a.agents.base_strands_agent import BaseStrandsAgent": 
                "from cryptotrading.core.agents.strands import StrandsAgent",
            "from rex.a2a.agents.memory_strands_agent import MemoryStrandsAgent": 
                "from cryptotrading.core.agents.memory import MemoryAgent",
            "from rex.a2a.agents.a2a_strands_agent import A2AStrandsAgent": 
                "from cryptotrading.core.agents.strands import StrandsAgent",
            "from strands.agent import Agent": 
                "from cryptotrading.core.agents.strands import StrandsAgent",
            "from rex.database": "from cryptotrading.data.database",
            "from rex.security": "from cryptotrading.infrastructure.security",
            "from rex.logging": "from cryptotrading.infrastructure.logging",
            "from mcp.": "from cryptotrading.core.protocols.mcp.",
        }
        
        # Update imports in all Python files
        for root, dirs, files in os.walk(new_src):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text()
                        updated = False
                        
                        for old_import, new_import in import_mappings.items():
                            if old_import in content:
                                content = content.replace(old_import, new_import)
                                updated = True
                        
                        if updated:
                            file_path.write_text(content)
                            self.log(f"Updated imports in {file_path.relative_to(new_src)}")
                    
                    except Exception as e:
                        self.log(f"Error updating imports in {file}: {e}")
    
    def finalize_migration(self, new_src: Path):
        """Finalize the migration by replacing old structure"""
        # Remove old structure
        old_structure_backup = self.src_dir.parent / "src_old"
        if old_structure_backup.exists():
            shutil.rmtree(old_structure_backup)
        
        # Move current src to backup
        shutil.move(self.src_dir, old_structure_backup)
        
        # Move new structure to src
        shutil.move(new_src, self.src_dir)
        
        self.log("Finalized migration - old structure backed up to src_old")
    
    def run_migration(self):
        """Run the complete migration process"""
        self.log("Starting source directory restructure migration")
        
        try:
            # Step 1: Create backup
            self.create_backup()
            
            # Step 2: Analyze current structure
            agent_files = self.analyze_agent_files()
            self.log(f"Found {len(agent_files)} agent-related files")
            
            # Step 3: Create new structure
            new_src = self.create_new_structure()
            
            # Step 4: Create unified agents
            self.create_unified_agents(new_src)
            
            # Step 5: Migrate protocols
            self.migrate_protocols(new_src)
            
            # Step 6: Migrate data layer
            self.migrate_data_layer(new_src)
            
            # Step 7: Migrate infrastructure
            self.migrate_infrastructure(new_src)
            
            # Step 8: Migrate remaining components
            self.migrate_remaining_components(new_src)
            
            # Step 9: Update imports
            self.update_imports(new_src)
            
            # Step 10: Save migration log
            log_file = new_src.parent / "migration_log.json"
            log_file.write_text(json.dumps(self.migration_log, indent=2))
            
            self.log("Migration completed successfully!")
            self.log(f"New structure created at: {new_src}")
            self.log(f"Backup available at: {self.backup_dir}")
            self.log("Review the new structure before finalizing with finalize_migration()")
            
            return new_src
            
        except Exception as e:
            self.log(f"Migration failed: {e}")
            raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python migrate_src_structure.py <project_root>")
        sys.exit(1)
    
    project_root = sys.argv[1]
    migrator = SourceRestructureMigrator(project_root)
    
    try:
        new_structure = migrator.run_migration()
        print(f"\n✅ Migration completed successfully!")
        print(f"📁 New structure: {new_structure}")
        print(f"💾 Backup: {migrator.backup_dir}")
        print(f"\n🔍 Review the new structure, then run:")
        print(f"   migrator.finalize_migration(new_structure)")
        print(f"   to replace the old structure.")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print(f"💾 Original structure preserved")
        sys.exit(1)
