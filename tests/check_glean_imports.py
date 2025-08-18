#!/usr/bin/env python3
"""
Import validation script for Glean agent extension
Checks for unused, missing, and error imports
"""
import ast
import sys
from pathlib import Path
import importlib.util

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def analyze_imports(file_path: str):
    """Analyze imports in a Python file"""
    print(f"\nAnalyzing: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Extract imports
        imports = []
        from_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    from_imports.append(f"{module}.{alias.name}")
        
        # Extract used names
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Get the base name
                current = node
                while isinstance(current, ast.Attribute):
                    current = current.value
                if isinstance(current, ast.Name):
                    used_names.add(current.id)
        
        print(f"  Imports: {len(imports)} direct, {len(from_imports)} from imports")
        
        # Check for unused imports (simple heuristic)
        unused = []
        for imp in imports:
            base_name = imp.split('.')[0]
            if base_name not in used_names and base_name not in ['typing', 'dataclasses']:
                unused.append(imp)
        
        if unused:
            print(f"  ⚠️  Potentially unused imports: {unused}")
        else:
            print(f"  ✓ All imports appear to be used")
        
        # Try to compile
        try:
            compile(content, file_path, 'exec')
            print(f"  ✓ Compiles successfully")
        except SyntaxError as e:
            print(f"  ✗ Syntax error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Analysis failed: {e}")
        return False

def check_import_errors():
    """Check for import errors in the new Glean files"""
    
    glean_files = [
        "src/cryptotrading/infrastructure/analysis/glean_data_schemas.py",
        "src/cryptotrading/infrastructure/analysis/scip_data_flow_indexer.py", 
        "src/cryptotrading/infrastructure/analysis/glean_runtime_collectors.py",
        "src/cryptotrading/infrastructure/analysis/crypto_angle_queries.py",
        "src/cryptotrading/core/agents/specialized/strands_glean_agent.py"
    ]
    
    print("🔍 CHECKING GLEAN AGENT IMPORTS")
    print("=" * 50)
    
    results = []
    
    for file_path in glean_files:
        if Path(file_path).exists():
            success = analyze_imports(file_path)
            results.append((file_path, success))
        else:
            print(f"\n⚠️  File not found: {file_path}")
            results.append((file_path, False))
    
    # Try importing each module
    print(f"\n{'='*50}")
    print("TESTING ACTUAL IMPORTS")
    print("=" * 50)
    
    import_tests = [
        ("glean_data_schemas", "src.cryptotrading.infrastructure.analysis.glean_data_schemas"),
        ("scip_data_flow_indexer", "src.cryptotrading.infrastructure.analysis.scip_data_flow_indexer"),
        ("glean_runtime_collectors", "src.cryptotrading.infrastructure.analysis.glean_runtime_collectors"),
        ("crypto_angle_queries", "src.cryptotrading.infrastructure.analysis.crypto_angle_queries"),
    ]
    
    for name, module_path in import_tests:
        try:
            spec = importlib.util.spec_from_file_location(
                name, 
                module_path.replace('.', '/') + '.py'
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"✓ {name} imports successfully")
            else:
                print(f"✗ {name} spec creation failed")
        except Exception as e:
            print(f"✗ {name} import failed: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("IMPORT VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for file_path, success in results:
        status = "✓" if success else "✗"
        filename = Path(file_path).name
        print(f"{status} {filename}")
    
    print(f"\nFiles analyzed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All Glean agent files have clean imports!")
    else:
        print("⚠️  Some files have import issues")
    
    return passed == total

if __name__ == "__main__":
    success = check_import_errors()
    sys.exit(0 if success else 1)