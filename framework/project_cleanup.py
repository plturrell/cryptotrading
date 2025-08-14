#!/usr/bin/env python3
"""
Project Root Cleanup Tool

Organize and clean up scattered files in the project root directory.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
import json


class ProjectCleanup:
    """Clean up and organize project root files."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.cleanup_report = {
            "files_moved": [],
            "files_archived": [],
            "directories_created": [],
            "errors": [],
            "success": True
        }
        
        # Define file categories and their target locations
        self.file_categories = {
            "test_files": {
                "pattern": "test_*.py",
                "target_dir": "tests/legacy_root_tests",
                "description": "Legacy test files from project root"
            },
            "debug_files": {
                "pattern": ["debug_*.py", "trace_*.py", "verify_*.py", "final_*.py", "check_*.py"],
                "target_dir": "legacy_debugging_backup",
                "description": "Additional debugging and verification files"
            },
            "documentation": {
                "files": [
                    "A2A_ORCHESTRATION_ANSWERS.md",
                    "Day1_Progress_Report.md", 
                    "MASTER_TODO_30_DAYS.md",
                    "PRODUCTION_READY_FIXES.md"
                ],
                "target_dir": "docs/project_history",
                "description": "Project documentation and progress reports"
            },
            "build_configs": {
                "files": [
                    "build-deploy.sh",
                    "karma.conf.js",
                    "tsconfig.json",
                    "ui5.yaml",
                    "ui5-local.yaml",
                    "manifest.json",
                    "vercel.json"
                ],
                "target_dir": "config/build",
                "description": "Build and deployment configuration files"
            }
        }
    
    def analyze_root_files(self) -> Dict[str, Any]:
        """Analyze files in project root and categorize them."""
        print("🔍 Analyzing project root files...")
        
        root_files = [f for f in self.project_root.iterdir() if f.is_file()]
        
        analysis = {
            "total_files": len(root_files),
            "categorized": {},
            "uncategorized": [],
            "should_stay": [],
            "recommendations": []
        }
        
        # Files that should stay in root
        keep_in_root = {
            "README.md", ".gitignore", ".env", ".env.production", 
            "requirements.txt", "requirements-dev.txt", "requirements-vercel.txt",
            "pyproject.toml", "package.json", "package-lock.json",
            ".htaccess", ".vercelignore",
            "app.py", "app_vercel.py",
            "MIGRATION_GUIDE.md", "AGENT_TESTING_FRAMEWORK_SUMMARY.md"
        }
        
        for file_path in root_files:
            filename = file_path.name
            
            if filename in keep_in_root:
                analysis["should_stay"].append(filename)
            elif filename.startswith("test_"):
                analysis["categorized"].setdefault("test_files", []).append(filename)
            elif any(filename.startswith(prefix) for prefix in ["debug_", "trace_", "verify_", "final_", "check_"]):
                analysis["categorized"].setdefault("debug_files", []).append(filename)
            elif filename in self.file_categories["documentation"]["files"]:
                analysis["categorized"].setdefault("documentation", []).append(filename)
            elif filename in self.file_categories["build_configs"]["files"]:
                analysis["categorized"].setdefault("build_configs", []).append(filename)
            else:
                analysis["uncategorized"].append(filename)
        
        # Generate recommendations
        if analysis["categorized"].get("test_files"):
            analysis["recommendations"].append(f"Move {len(analysis['categorized']['test_files'])} test files to tests/legacy_root_tests/")
        
        if analysis["categorized"].get("debug_files"):
            analysis["recommendations"].append(f"Archive {len(analysis['categorized']['debug_files'])} debug files")
        
        if analysis["categorized"].get("documentation"):
            analysis["recommendations"].append(f"Organize {len(analysis['categorized']['documentation'])} documentation files")
        
        if analysis["categorized"].get("build_configs"):
            analysis["recommendations"].append(f"Move {len(analysis['categorized']['build_configs'])} config files")
        
        return analysis
    
    def cleanup_test_files(self):
        """Move test files to appropriate test directory."""
        print("📁 Organizing test files...")
        
        target_dir = self.project_root / self.file_categories["test_files"]["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_report["directories_created"].append(str(target_dir))
        
        test_files = list(self.project_root.glob("test_*.py"))
        
        for test_file in test_files:
            try:
                target_file = target_dir / test_file.name
                shutil.move(str(test_file), str(target_file))
                self.cleanup_report["files_moved"].append(f"{test_file.name} -> {target_dir.relative_to(self.project_root)}")
                print(f"  ✓ Moved: {test_file.name}")
            except Exception as e:
                self.cleanup_report["errors"].append(f"Failed to move {test_file.name}: {str(e)}")
        
        # Create README for the test directory
        readme_content = f"""# Legacy Root Test Files

This directory contains test files that were previously scattered in the project root.

## Files Moved
{chr(10).join(f"- {f.name}" for f in test_files)}

## Migration Note
These files have been moved here for organization. Consider:
1. Reviewing if they're still needed
2. Integrating useful tests into the main test suite
3. Updating any imports or references
4. Using the new Agent Testing Framework for new tests

## New Testing Approach
Use the Agent Testing Framework for new tests:
```bash
python3 -m framework.agent_testing.cli --full-suite
```

See `framework/agent_testing/README.md` for details.
"""
        
        readme_file = target_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
    
    def cleanup_debug_files(self):
        """Archive additional debug files."""
        print("📦 Archiving additional debug files...")
        
        target_dir = self.project_root / self.file_categories["debug_files"]["target_dir"]
        
        debug_patterns = ["debug_*.py", "trace_*.py", "verify_*.py", "final_*.py", "check_*.py"]
        debug_files = []
        
        for pattern in debug_patterns:
            debug_files.extend(self.project_root.glob(pattern))
        
        for debug_file in debug_files:
            try:
                target_file = target_dir / debug_file.name
                if not target_file.exists():  # Don't overwrite existing backups
                    shutil.move(str(debug_file), str(target_file))
                    self.cleanup_report["files_archived"].append(f"{debug_file.name} -> {target_dir.relative_to(self.project_root)}")
                    print(f"  ✓ Archived: {debug_file.name}")
                else:
                    # File already backed up, just remove the original
                    debug_file.unlink()
                    print(f"  ✓ Removed (already backed up): {debug_file.name}")
            except Exception as e:
                self.cleanup_report["errors"].append(f"Failed to archive {debug_file.name}: {str(e)}")
    
    def cleanup_documentation(self):
        """Organize documentation files."""
        print("📚 Organizing documentation files...")
        
        target_dir = self.project_root / self.file_categories["documentation"]["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_report["directories_created"].append(str(target_dir))
        
        doc_files = self.file_categories["documentation"]["files"]
        
        for filename in doc_files:
            source_file = self.project_root / filename
            if source_file.exists():
                try:
                    target_file = target_dir / filename
                    shutil.move(str(source_file), str(target_file))
                    self.cleanup_report["files_moved"].append(f"{filename} -> {target_dir.relative_to(self.project_root)}")
                    print(f"  ✓ Moved: {filename}")
                except Exception as e:
                    self.cleanup_report["errors"].append(f"Failed to move {filename}: {str(e)}")
        
        # Create index for documentation
        index_content = f"""# Project History Documentation

This directory contains historical project documentation and progress reports.

## Files
- `A2A_ORCHESTRATION_ANSWERS.md` - A2A orchestration implementation details
- `Day1_Progress_Report.md` - Initial development progress
- `MASTER_TODO_30_DAYS.md` - 30-day development roadmap
- `PRODUCTION_READY_FIXES.md` - Production readiness checklist

## Current Documentation
For current project documentation, see:
- `/README.md` - Main project README
- `/docs/` - Technical documentation
- `/framework/agent_testing/README.md` - Testing framework docs
"""
        
        index_file = target_dir / "README.md"
        with open(index_file, 'w') as f:
            f.write(index_content)
    
    def cleanup_build_configs(self):
        """Organize build and configuration files."""
        print("⚙️ Organizing build configuration files...")
        
        target_dir = self.project_root / self.file_categories["build_configs"]["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_report["directories_created"].append(str(target_dir))
        
        config_files = self.file_categories["build_configs"]["files"]
        
        for filename in config_files:
            source_file = self.project_root / filename
            if source_file.exists():
                try:
                    target_file = target_dir / filename
                    shutil.move(str(source_file), str(target_file))
                    self.cleanup_report["files_moved"].append(f"{filename} -> {target_dir.relative_to(self.project_root)}")
                    print(f"  ✓ Moved: {filename}")
                except Exception as e:
                    self.cleanup_report["errors"].append(f"Failed to move {filename}: {str(e)}")
        
        # Create configuration index
        config_index = f"""# Build Configuration Files

This directory contains build and deployment configuration files.

## Files
- `build-deploy.sh` - Build and deployment script
- `karma.conf.js` - Karma test configuration
- `tsconfig.json` - TypeScript configuration
- `ui5.yaml` - SAP UI5 configuration
- `ui5-local.yaml` - Local UI5 development configuration
- `manifest.json` - Application manifest
- `vercel.json` - Vercel deployment configuration

## Usage
These files are referenced by build tools and deployment systems.
Update paths in build scripts if needed after this reorganization.
"""
        
        config_readme = target_dir / "README.md"
        with open(config_readme, 'w') as f:
            f.write(config_index)
    
    def perform_cleanup(self) -> Dict[str, Any]:
        """Perform the complete cleanup operation."""
        print("🧹 Starting project root cleanup...")
        print("=" * 50)
        
        try:
            # Analyze current state
            analysis = self.analyze_root_files()
            print(f"📊 Found {analysis['total_files']} files in project root")
            
            # Perform cleanup operations
            self.cleanup_test_files()
            self.cleanup_debug_files()
            self.cleanup_documentation()
            self.cleanup_build_configs()
            
            print("\n✅ Cleanup completed successfully!")
            
        except Exception as e:
            self.cleanup_report["success"] = False
            self.cleanup_report["errors"].append(f"Cleanup failed: {str(e)}")
            print(f"❌ Cleanup failed: {e}")
        
        return self.cleanup_report
    
    def generate_cleanup_summary(self) -> str:
        """Generate a summary of the cleanup operation."""
        summary = f"""# Project Root Cleanup Summary

## Operations Completed
- Files moved: {len(self.cleanup_report['files_moved'])}
- Files archived: {len(self.cleanup_report['files_archived'])}
- Directories created: {len(self.cleanup_report['directories_created'])}

## Files Moved
{chr(10).join(f"- {f}" for f in self.cleanup_report['files_moved'])}

## Files Archived
{chr(10).join(f"- {f}" for f in self.cleanup_report['files_archived'])}

## New Directory Structure
```
project_root/
├── tests/legacy_root_tests/     # Former root test files
├── docs/project_history/        # Historical documentation
├── config/build/               # Build configurations
└── legacy_debugging_backup/    # All debug files
```

## Benefits
- Cleaner project root with only essential files
- Better organization of test files
- Centralized configuration management
- Preserved all historical files safely

## Next Steps
1. Update any scripts that reference moved files
2. Review legacy test files for integration opportunities
3. Update build scripts with new config paths
4. Consider removing unused legacy files after verification
"""
        return summary


def main():
    """Main cleanup entry point."""
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    
    cleanup = ProjectCleanup(project_root)
    
    # First, show analysis
    analysis = cleanup.analyze_root_files()
    print("📊 PROJECT ROOT ANALYSIS")
    print("=" * 30)
    print(f"Total files: {analysis['total_files']}")
    print(f"Files to keep in root: {len(analysis['should_stay'])}")
    print(f"Files to organize: {sum(len(files) for files in analysis['categorized'].values())}")
    print(f"Uncategorized files: {len(analysis['uncategorized'])}")
    
    if analysis['recommendations']:
        print("\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
    
    # Ask for confirmation
    print(f"\n🤔 Proceed with cleanup? (y/N): ", end="")
    if input().lower().strip() == 'y':
        # Perform cleanup
        report = cleanup.perform_cleanup()
        
        # Generate summary
        summary = cleanup.generate_cleanup_summary()
        summary_file = Path(project_root) / "PROJECT_CLEANUP_SUMMARY.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"\n📄 Cleanup summary saved to: {summary_file}")
        
        if report["success"]:
            print("🎉 Project root cleanup completed successfully!")
            return 0
        else:
            print("❌ Cleanup completed with errors:")
            for error in report["errors"]:
                print(f"  - {error}")
            return 1
    else:
        print("❌ Cleanup cancelled by user")
        return 0


if __name__ == "__main__":
    exit(main())
