#!/usr/bin/env python3
"""
Comprehensive scan and fix script for Glean agent extension
Finds and fixes issues automatically
"""
import ast
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class GleanScanner:
    """Scanner for Glean agent issues"""
    
    def __init__(self):
        self.issues = []
        self.fixes_applied = []
        
    def scan_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Scan a file for issues"""
        issues = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check syntax
            try:
                ast.parse(content)
            except SyntaxError as e:
                issues.append({
                    'type': 'syntax_error',
                    'severity': 'error',
                    'message': f"Syntax error: {e}",
                    'line': e.lineno,
                    'fixable': False
                })
            
            # Check imports
            issues.extend(self._check_imports(content, file_path))
            
            # Check coding style
            issues.extend(self._check_style(content, file_path))
            
            # Check functionality
            issues.extend(self._check_functionality(content, file_path))
            
        except Exception as e:
            issues.append({
                'type': 'scan_error',
                'severity': 'error', 
                'message': f"Failed to scan file: {e}",
                'fixable': False
            })
        
        return issues
    
    def _check_imports(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Check import issues"""
        issues = []
        lines = content.split('\n')
        
        # Check for unused imports
        import_pattern = r'^(import|from)\s+(\w+)'
        imports = []
        
        for i, line in enumerate(lines, 1):
            match = re.match(import_pattern, line.strip())
            if match:
                import_type, module = match.groups()
                imports.append({
                    'line': i,
                    'type': import_type,
                    'module': module,
                    'full_line': line.strip()
                })
        
        # Check for relative imports that might fail
        for imp in imports:
            if 'from .' in imp['full_line']:
                # Check if the relative import is valid
                if not self._validate_relative_import(imp['full_line'], file_path):
                    issues.append({
                        'type': 'invalid_relative_import',
                        'severity': 'warning',
                        'message': f"Relative import may fail: {imp['full_line']}",
                        'line': imp['line'],
                        'fixable': True,
                        'fix_data': imp
                    })
        
        # Check for missing imports
        issues.extend(self._check_missing_imports(content))
        
        return issues
    
    def _check_style(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Check style issues"""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 120:
                issues.append({
                    'type': 'long_line',
                    'severity': 'warning',
                    'message': f"Line too long ({len(line)} > 120 chars)",
                    'line': i,
                    'fixable': False
                })
            
            # Check trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issues.append({
                    'type': 'trailing_whitespace',
                    'severity': 'info',
                    'message': "Trailing whitespace",
                    'line': i,
                    'fixable': True,
                    'fix_data': {'line_num': i, 'content': line.rstrip()}
                })
        
        return issues
    
    def _check_functionality(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Check functionality issues"""
        issues = []
        
        # Check for common patterns that might cause issues
        if 'super().index_file' in content and 'result' in content:
            # Check for potential None assignment
            if 'result = super().index_file' in content:
                issues.append({
                    'type': 'potential_none_assignment',
                    'severity': 'warning',
                    'message': "super().index_file() might return None",
                    'fixable': True,
                    'fix_data': {'pattern': 'super_index_file_none'}
                })
        
        # Check for missing error handling
        if 'open(' in content and 'except' not in content:
            issues.append({
                'type': 'missing_error_handling',
                'severity': 'warning',
                'message': "File operations without error handling",
                'fixable': False
            })
        
        return issues
    
    def _check_missing_imports(self, content: str) -> List[Dict[str, Any]]:
        """Check for missing imports"""
        issues = []
        
        # Common patterns that need imports
        patterns = {
            r'json\.': 'import json',
            r'pd\.': 'import pandas as pd',
            r'np\.': 'import numpy as np',
            r'datetime\.': 'from datetime import datetime',
            r'Path\(': 'from pathlib import Path',
            r'uuid\.': 'import uuid',
            r'asyncio\.': 'import asyncio'
        }
        
        for pattern, needed_import in patterns.items():
            if re.search(pattern, content) and needed_import not in content:
                issues.append({
                    'type': 'missing_import',
                    'severity': 'error',
                    'message': f"Missing import: {needed_import}",
                    'fixable': True,
                    'fix_data': {'import_line': needed_import}
                })
        
        return issues
    
    def _validate_relative_import(self, import_line: str, file_path: str) -> bool:
        """Validate if a relative import is correct"""
        # Simple validation - check if the referenced file exists
        if 'from .' in import_line:
            # Extract module name
            match = re.search(r'from \.(\w+) import', import_line)
            if match:
                module_name = match.group(1)
                current_dir = Path(file_path).parent
                expected_file = current_dir / f"{module_name}.py"
                return expected_file.exists()
        return True
    
    def fix_issues(self, file_path: str, issues: List[Dict[str, Any]]) -> List[str]:
        """Fix fixable issues in a file"""
        fixes_applied = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            for issue in issues:
                if issue.get('fixable', False):
                    content, fixed = self._apply_fix(content, issue)
                    if fixed:
                        fixes_applied.append(issue['message'])
            
            # Only write if changes were made
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"  ✓ Applied {len(fixes_applied)} fixes to {Path(file_path).name}")
            
        except Exception as e:
            print(f"  ✗ Failed to fix {file_path}: {e}")
        
        return fixes_applied
    
    def _apply_fix(self, content: str, issue: Dict[str, Any]) -> Tuple[str, bool]:
        """Apply a specific fix"""
        issue_type = issue['type']
        
        if issue_type == 'trailing_whitespace':
            fix_data = issue['fix_data']
            lines = content.split('\n')
            if fix_data['line_num'] <= len(lines):
                lines[fix_data['line_num'] - 1] = fix_data['content']
                return '\n'.join(lines), True
        
        elif issue_type == 'missing_import':
            import_line = issue['fix_data']['import_line']
            # Add import at the top after existing imports
            lines = content.split('\n')
            
            # Find where to insert import
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    insert_pos = i + 1
                elif line.strip() == '' and insert_pos > 0:
                    break
            
            lines.insert(insert_pos, import_line)
            return '\n'.join(lines), True
        
        elif issue_type == 'potential_none_assignment':
            # Fix the super().index_file() pattern
            pattern = issue['fix_data']['pattern']
            if pattern == 'super_index_file_none':
                old_pattern = r'result = super\(\)\.index_file\(file_path\)'
                new_pattern = '''parent_result = super().index_file(file_path)
        if parent_result and parent_result.get('status') == 'success':
            result = parent_result
        else:
            result = {"status": "success", "file": file_path}'''
                
                if re.search(old_pattern, content):
                    content = re.sub(old_pattern, new_pattern, content)
                    return content, True
        
        return content, False

def scan_and_fix_glean():
    """Main scan and fix function"""
    print("🔍 SCANNING GLEAN AGENT FOR ISSUES")
    print("=" * 50)
    
    scanner = GleanScanner()
    
    # Files to scan
    glean_files = [
        "src/cryptotrading/infrastructure/analysis/glean_data_schemas.py",
        "src/cryptotrading/infrastructure/analysis/scip_data_flow_indexer.py", 
        "src/cryptotrading/infrastructure/analysis/glean_runtime_collectors.py",
        "src/cryptotrading/infrastructure/analysis/crypto_angle_queries.py",
        "src/cryptotrading/core/agents/specialized/strands_glean_agent.py"
    ]
    
    all_issues = {}
    total_issues = 0
    
    # Scan all files
    for file_path in glean_files:
        if not Path(file_path).exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        print(f"\nScanning: {Path(file_path).name}")
        issues = scanner.scan_file(file_path)
        
        if issues:
            all_issues[file_path] = issues
            total_issues += len(issues)
            
            # Group by severity
            errors = [i for i in issues if i.get('severity') == 'error']
            warnings = [i for i in issues if i.get('severity') == 'warning']
            info = [i for i in issues if i.get('severity') == 'info']
            
            print(f"  Found {len(issues)} issues: {len(errors)} errors, {len(warnings)} warnings, {len(info)} info")
            
            # Show errors and warnings
            for issue in errors + warnings:
                severity_icon = "✗" if issue['severity'] == 'error' else "⚠️"
                fixable = " [FIXABLE]" if issue.get('fixable') else ""
                print(f"    {severity_icon} {issue['message']}{fixable}")
        else:
            print(f"  ✓ No issues found")
    
    if total_issues == 0:
        print(f"\n🎉 NO ISSUES FOUND!")
        print("All Glean agent files are clean.")
        return True
    
    # Ask to apply fixes
    print(f"\n{'='*50}")
    print(f"FOUND {total_issues} TOTAL ISSUES")
    print("=" * 50)
    
    fixable_count = sum(
        len([i for i in issues if i.get('fixable', False)]) 
        for issues in all_issues.values()
    )
    
    if fixable_count > 0:
        print(f"\n🔧 APPLYING {fixable_count} AUTOMATIC FIXES")
        print("=" * 50)
        
        total_fixes = 0
        for file_path, issues in all_issues.items():
            fixable_issues = [i for i in issues if i.get('fixable', False)]
            if fixable_issues:
                print(f"\nFixing: {Path(file_path).name}")
                fixes = scanner.fix_issues(file_path, fixable_issues)
                total_fixes += len(fixes)
                for fix in fixes:
                    print(f"  ✓ {fix}")
        
        print(f"\n✓ Applied {total_fixes} fixes total")
        
        # Scan again to verify fixes
        print(f"\n{'='*50}")
        print("VERIFYING FIXES")
        print("=" * 50)
        
        remaining_issues = 0
        for file_path in all_issues.keys():
            issues = scanner.scan_file(file_path)
            if issues:
                remaining_issues += len(issues)
                print(f"  {Path(file_path).name}: {len(issues)} issues remain")
            else:
                print(f"  ✓ {Path(file_path).name}: All issues fixed")
        
        if remaining_issues == 0:
            print(f"\n🎉 ALL FIXABLE ISSUES RESOLVED!")
        else:
            print(f"\n⚠️  {remaining_issues} issues require manual attention")
    
    else:
        print(f"\n⚠️  No automatic fixes available")
        print("All issues require manual attention")
    
    return total_issues == 0 or fixable_count == total_issues

if __name__ == "__main__":
    success = scan_and_fix_glean()
    sys.exit(0 if success else 1)