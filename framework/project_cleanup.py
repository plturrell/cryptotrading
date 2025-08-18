#!/usr/bin/env python3
"""
Final Cleanup Script

Permanently remove legacy and backup files that are no longer needed.
"""

import shutil
from pathlib import Path
from typing import List, Dict, Any


class FinalCleanup:
    """Permanently remove legacy and backup files."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.cleanup_report = {
            "directories_removed": [],
            "files_removed": [],
            "space_freed": 0,
            "errors": [],
            "success": True
        }
        
        # Define what to remove
        self.items_to_remove = {
            "directories": [
                "legacy_debugging_backup",
                "tests/legacy_root_tests",
                "docs/project_history",
                "__pycache__"
            ],
            "files": [
                "deep_inspection.py",  # Last remaining debug file
                "start_blockchain_a2a.py",  # Appears to be a test/debug file
                "PROJECT_CLEANUP_SUMMARY.md",  # No longer needed
                "MIGRATION_GUIDE.md",  # No longer needed after cleanup
            ]
        }
    
    def calculate_size(self, path: Path) -> int:
        """Calculate total size of a file or directory."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total_size = 0
            try:
                for item in path.rglob('*'):
                    if item.is_file():
                        total_size += item.stat().st_size
            except (PermissionError, OSError):
                pass
            return total_size
        return 0
    
    def remove_directories(self):
        """Remove legacy directories."""
        print("🗂️ Removing legacy directories...")
        
        for dir_name in self.items_to_remove["directories"]:
            dir_path = self.project_root / dir_name
            
            if dir_path.exists():
                try:
                    size = self.calculate_size(dir_path)
                    shutil.rmtree(dir_path)
                    self.cleanup_report["directories_removed"].append(dir_name)
                    self.cleanup_report["space_freed"] += size
                    print(f"  ✓ Removed directory: {dir_name} ({self.format_size(size)})")
                except Exception as e:
                    self.cleanup_report["errors"].append(f"Failed to remove directory {dir_name}: {str(e)}")
                    print(f"  ❌ Failed to remove {dir_name}: {e}")
            else:
                print(f"  ⚠️ Directory not found: {dir_name}")
    
    def remove_files(self):
        """Remove legacy files."""
        print("📄 Removing legacy files...")
        
        for file_name in self.items_to_remove["files"]:
            file_path = self.project_root / file_name
            
            if file_path.exists():
                try:
                    size = self.calculate_size(file_path)
                    file_path.unlink()
                    self.cleanup_report["files_removed"].append(file_name)
                    self.cleanup_report["space_freed"] += size
                    print(f"  ✓ Removed file: {file_name} ({self.format_size(size)})")
                except Exception as e:
                    self.cleanup_report["errors"].append(f"Failed to remove file {file_name}: {str(e)}")
                    print(f"  ❌ Failed to remove {file_name}: {e}")
            else:
                print(f"  ⚠️ File not found: {file_name}")
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def show_before_after(self):
        """Show before and after comparison."""
        print("📊 BEFORE vs AFTER COMPARISON")
        print("=" * 40)
        
        # Count current files in root
        root_files = [f for f in self.project_root.iterdir() if f.is_file()]
        print(f"Files in project root: {len(root_files)}")
        
        # Show what will be removed
        total_items = len(self.items_to_remove["directories"]) + len(self.items_to_remove["files"])
        print(f"Items to remove: {total_items}")
        
        print("\n🗑️ Items to be permanently removed:")
        for dir_name in self.items_to_remove["directories"]:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                size = self.calculate_size(dir_path)
                print(f"  📁 {dir_name}/ ({self.format_size(size)})")
        
        for file_name in self.items_to_remove["files"]:
            file_path = self.project_root / file_name
            if file_path.exists():
                size = self.calculate_size(file_path)
                print(f"  📄 {file_name} ({self.format_size(size)})")
    
    def perform_cleanup(self) -> Dict[str, Any]:
        """Perform the final cleanup."""
        print("🧹 FINAL CLEANUP - Removing Legacy Files")
        print("=" * 50)
        
        try:
            # Show what will be removed
            self.show_before_after()
            
            # Perform removal
            self.remove_directories()
            self.remove_files()
            
            print(f"\n✅ Final cleanup completed!")
            print(f"📊 Summary:")
            print(f"  - Directories removed: {len(self.cleanup_report['directories_removed'])}")
            print(f"  - Files removed: {len(self.cleanup_report['files_removed'])}")
            print(f"  - Space freed: {self.format_size(self.cleanup_report['space_freed'])}")
            
            if self.cleanup_report["errors"]:
                print(f"  - Errors: {len(self.cleanup_report['errors'])}")
                for error in self.cleanup_report["errors"]:
                    print(f"    ❌ {error}")
            
        except Exception as e:
            self.cleanup_report["success"] = False
            self.cleanup_report["errors"].append(f"Final cleanup failed: {str(e)}")
            print(f"❌ Final cleanup failed: {e}")
        
        return self.cleanup_report
    
    def show_final_structure(self):
        """Show the final clean project structure."""
        print("\n🏗️ FINAL PROJECT STRUCTURE")
        print("=" * 30)
        
        # Show only the main directories that remain
        important_dirs = [
            "src", "framework", "tests", "docs", "config", 
            "api", "webapp", "scripts", "workflows"
        ]
        
        for dir_name in important_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                try:
                    child_count = len(list(dir_path.iterdir()))
                    print(f"📁 {dir_name}/ ({child_count} items)")
                except:
                    print(f"📁 {dir_name}/")
        
        # Show essential files
        essential_files = [
            "README.md", "app.py", "package.json", "requirements.txt",
            "pyproject.toml", "AGENT_TESTING_FRAMEWORK_SUMMARY.md"
        ]
        
        print("\n📄 Essential files:")
        for file_name in essential_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                size = self.calculate_size(file_path)
                print(f"  📄 {file_name} ({self.format_size(size)})")


def main():
    """Main cleanup entry point."""
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    
    cleanup = FinalCleanup(project_root)
    
    print("⚠️  FINAL CLEANUP WARNING")
    print("=" * 30)
    print("This will PERMANENTLY remove all legacy and backup files.")
    print("These files cannot be recovered after deletion.")
    print("\nFiles and directories to be removed:")
    
    # Show what will be removed
    for dir_name in cleanup.items_to_remove["directories"]:
        dir_path = Path(project_root) / dir_name
        if dir_path.exists():
            print(f"  🗂️ {dir_name}/")
    
    for file_name in cleanup.items_to_remove["files"]:
        file_path = Path(project_root) / file_name
        if file_path.exists():
            print(f"  📄 {file_name}")
    
    print(f"\n🤔 Are you sure you want to permanently delete these items? (y/N): ", end="")
    
    if input().lower().strip() == 'y':
        # Perform cleanup
        report = cleanup.perform_cleanup()
        cleanup.show_final_structure()
        
        if report["success"]:
            print("\n🎉 Project is now completely clean!")
            print("✨ Only essential files and the new framework remain.")
            return 0
        else:
            print("\n❌ Cleanup completed with some errors.")
            return 1
    else:
        print("❌ Final cleanup cancelled by user")
        return 0


if __name__ == "__main__":
    exit(main())
