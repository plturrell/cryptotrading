#!/usr/bin/env python3
"""
Import Fix Script for New src/cryptotrading/ Structure
Updates all import statements throughout the codebase to work with the new structure.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

class ImportFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixes_applied = []
        
        # Define import mappings from old to new structure
        self.import_mappings = {
            # Old rex imports to new cryptotrading structure
            "from rex.database": "from cryptotrading.data.database",
            "from rex.a2a.agents": "from cryptotrading.core.agents.specialized",
            "from rex.a2a.protocols": "from cryptotrading.core.protocols.a2a",
            "from rex.a2a.orchestration": "from cryptotrading.core.protocols.a2a.orchestration",
            "from rex.a2a.registry": "from cryptotrading.infrastructure.registry",
            "from rex.observability": "from cryptotrading.infrastructure.monitoring",
            "from rex.logging": "from cryptotrading.infrastructure.logging",
            "from rex.security": "from cryptotrading.infrastructure.security",
            "from rex.storage": "from cryptotrading.data.storage",
            "from rex.historical_data": "from cryptotrading.data.historical",
            "from rex.ml": "from cryptotrading.core.ml",
            "from rex.ai": "from cryptotrading.core.ai",
            "from rex.market_data": "from cryptotrading.data.market_data",
            "from rex.utils": "from cryptotrading.utils",
            "from rex.memory": "from cryptotrading.infrastructure.memory",
            "from rex.monitoring": "from cryptotrading.infrastructure.monitoring",
            "from rex.registry": "from cryptotrading.infrastructure.registry",
            
            # Old src.rex imports to new structure
            "from src.rex.database": "from cryptotrading.data.database",
            "from src.rex.a2a.agents": "from cryptotrading.core.agents.specialized",
            "from src.rex.a2a.protocols": "from cryptotrading.core.protocols.a2a",
            "from src.rex.a2a.orchestration": "from cryptotrading.core.protocols.a2a.orchestration",
            "from src.rex.observability": "from cryptotrading.infrastructure.monitoring",
            "from src.rex.logging": "from cryptotrading.infrastructure.logging",
            "from src.rex.security": "from cryptotrading.infrastructure.security",
            "from src.rex.storage": "from cryptotrading.data.storage",
            "from src.rex.historical_data": "from cryptotrading.data.historical",
            "from src.rex.ml": "from cryptotrading.core.ml",
            "from src.rex.ai": "from cryptotrading.core.ai",
            "from src.rex.market_data": "from cryptotrading.data.market_data",
            "from src.rex.utils": "from cryptotrading.utils",
            "from src.rex.memory": "from cryptotrading.infrastructure.memory",
            "from src.rex.monitoring": "from cryptotrading.infrastructure.monitoring",
            "from src.rex.registry": "from cryptotrading.infrastructure.registry",
            
            # Old MCP imports to new structure
            "from mcp.": "from cryptotrading.core.protocols.mcp.",
            "from src.mcp.": "from cryptotrading.core.protocols.mcp.",
            
            # Old strands imports to new structure
            "from strands.": "from cryptotrading.core.agents.",
            "from src.strands.": "from cryptotrading.core.agents.",
            
            # Import statements without from
            "import rex.": "import cryptotrading.",
            "import mcp.": "import cryptotrading.core.protocols.mcp.",
            "import strands.": "import cryptotrading.core.agents.",
        }
        
        # Specific class mappings for agent consolidation
        self.class_mappings = {
            "BaseStrandsAgent": "StrandsAgent",
            "A2AStrandsAgent": "StrandsAgent", 
            "MemoryStrandsAgent": "MemoryAgent",
            "BaseMemoryAgent": "MemoryAgent",
            "A2AAgentBase": "BaseAgent",
            "BaseAgent": "BaseAgent",  # Keep as is
        }
    
    def log(self, message: str):
        """Log fix operations"""
        print(f"[FIX] {message}")
        self.fixes_applied.append(message)
    
    def fix_file_imports(self, file_path: Path) -> bool:
        """Fix imports in a single file"""
        if not file_path.exists() or not file_path.suffix == '.py':
            return False
        
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # Apply import mappings
            for old_import, new_import in self.import_mappings.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    self.log(f"Fixed import in {file_path.relative_to(self.project_root)}: {old_import} -> {new_import}")
            
            # Apply class name mappings
            for old_class, new_class in self.class_mappings.items():
                if old_class != new_class:  # Don't replace with itself
                    # Replace class imports
                    pattern = rf'\bfrom\s+[\w.]+\s+import\s+.*\b{old_class}\b'
                    if re.search(pattern, content):
                        content = re.sub(rf'\b{old_class}\b', new_class, content)
                        self.log(f"Fixed class name in {file_path.relative_to(self.project_root)}: {old_class} -> {new_class}")
            
            # Write back if changed
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                return True
                
        except Exception as e:
            self.log(f"Error fixing {file_path}: {e}")
            return False
        
        return False
    
    def fix_directory_imports(self, directory: Path, exclude_dirs: List[str] = None) -> int:
        """Fix imports in all Python files in a directory"""
        if exclude_dirs is None:
            exclude_dirs = ['node_modules', '.git', '.vercel', '__pycache__', '.pytest_cache']
        
        fixed_count = 0
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if self.fix_file_imports(file_path):
                        fixed_count += 1
        
        return fixed_count
    
    def create_missing_init_files(self):
        """Create missing __init__.py files for the new structure"""
        init_paths = [
            "src/cryptotrading/core/agents/specialized/__init__.py",
            "src/cryptotrading/core/protocols/a2a/orchestration/__init__.py",
            "src/cryptotrading/data/market_data/__init__.py",
            "src/cryptotrading/infrastructure/memory/__init__.py",
        ]
        
        for init_path in init_paths:
            full_path = self.project_root / init_path
            if not full_path.exists():
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text('"""Module initialization."""\n')
                self.log(f"Created missing __init__.py: {init_path}")
    
    def fix_specific_import_issues(self):
        """Fix specific import issues found in the analysis"""
        specific_fixes = [
            # API directory fixes
            {
                "file": "api/mcp.py",
                "old": "from src.mcp.enhanced_server import create_enhanced_mcp_server",
                "new": "from cryptotrading.core.protocols.mcp.enhanced_server import create_enhanced_mcp_server"
            },
            {
                "file": "api/mcp.py", 
                "old": "from src.mcp.auth import AuthMiddleware, AuthContext",
                "new": "from cryptotrading.core.protocols.mcp.auth import AuthMiddleware, AuthContext"
            },
            {
                "file": "api/mcp.py",
                "old": "from src.mcp.metrics import mcp_metrics",
                "new": "from cryptotrading.core.protocols.mcp.metrics import mcp_metrics"
            },
            {
                "file": "api/agents/routes.py",
                "old": "from rex.a2a.registry.registry import agent_registry",
                "new": "from cryptotrading.infrastructure.registry.registry import agent_registry"
            },
            
            # Scripts directory fixes
            {
                "file": "scripts/init_db.py",
                "old": "from rex.database import DatabaseClient, User",
                "new": "from cryptotrading.data.database import DatabaseClient, User"
            },
            {
                "file": "scripts/init_db.py",
                "old": "from rex.database.models import MarketData",
                "new": "from cryptotrading.data.database.models import MarketData"
            },
            {
                "file": "scripts/daily_progress.py",
                "old": "from rex.database import DatabaseClient",
                "new": "from cryptotrading.data.database import DatabaseClient"
            },
        ]
        
        for fix in specific_fixes:
            file_path = self.project_root / fix["file"]
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if fix["old"] in content:
                        content = content.replace(fix["old"], fix["new"])
                        file_path.write_text(content, encoding='utf-8')
                        self.log(f"Applied specific fix to {fix['file']}: {fix['old']} -> {fix['new']}")
                except Exception as e:
                    self.log(f"Error applying specific fix to {fix['file']}: {e}")
    
    def run_fixes(self):
        """Run all import fixes"""
        self.log("Starting comprehensive import fixes for new src/cryptotrading/ structure")
        
        # Create missing __init__.py files
        self.create_missing_init_files()
        
        # Apply specific fixes first
        self.fix_specific_import_issues()
        
        # Fix imports in all directories
        directories_to_fix = [
            "api",
            "scripts", 
            "tests",
            "app.py",
            "app_vercel.py"
        ]
        
        total_fixed = 0
        
        for item in directories_to_fix:
            path = self.project_root / item
            if path.is_dir():
                fixed = self.fix_directory_imports(path)
                total_fixed += fixed
                self.log(f"Fixed {fixed} files in {item}/")
            elif path.is_file() and path.suffix == '.py':
                if self.fix_file_imports(path):
                    total_fixed += 1
                    self.log(f"Fixed imports in {item}")
        
        # Also fix any remaining imports in src/ (internal references)
        src_fixed = self.fix_directory_imports(self.project_root / "src")
        total_fixed += src_fixed
        self.log(f"Fixed {src_fixed} internal imports in src/")
        
        self.log(f"Import fixes completed! Total files fixed: {total_fixed}")
        return total_fixed

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python fix_imports.py <project_root>")
        sys.exit(1)
    
    project_root = sys.argv[1]
    fixer = ImportFixer(project_root)
    
    try:
        total_fixed = fixer.run_fixes()
        print(f"\n✅ Import fixes completed successfully!")
        print(f"📊 Total files fixed: {total_fixed}")
        print(f"📝 Fixes applied: {len(fixer.fixes_applied)}")
        
        if total_fixed > 0:
            print(f"\n🔧 Run the following to commit the fixes:")
            print(f"   git add -A && git commit -m 'fix: Update all imports for new src/cryptotrading/ structure'")
        
    except Exception as e:
        print(f"\n❌ Import fixes failed: {e}")
        sys.exit(1)
