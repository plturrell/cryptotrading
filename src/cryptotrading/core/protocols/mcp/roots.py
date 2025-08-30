"""
MCP Roots Implementation
Implements root directory management for file system access
"""

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class RootEntry:
    """Root directory entry"""

    uri: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP format"""
        result = {"uri": self.uri}
        if self.name:
            result["name"] = self.name
        return result


class RootManager:
    """Manager for root directories"""

    def __init__(self):
        self.roots: Dict[str, RootEntry] = {}
        self._change_handlers: List[Callable] = []
        self._allowed_schemes = {"file", "data", "config", "temp"}
        self._security_enabled = True

        # Default roots
        self._add_default_roots()

    def _add_default_roots(self):
        """Add default root directories"""
        # Current working directory
        cwd = os.getcwd()
        self.add_root(f"file://{cwd}", "Current Directory")

        # User home directory (if accessible)
        try:
            home = str(Path.home())
            self.add_root(f"file://{home}", "Home Directory")
        except Exception:
            pass

        # Temporary directory
        import tempfile

        temp_dir = tempfile.gettempdir()
        self.add_root(f"file://{temp_dir}", "Temporary Directory")

    def add_root(self, uri: str, name: Optional[str] = None):
        """Add root directory"""
        if not self._is_valid_root_uri(uri):
            raise ValueError(f"Invalid root URI: {uri}")

        if self._security_enabled and not self._is_safe_path(uri):
            raise ValueError(f"Root path not allowed for security reasons: {uri}")

        root = RootEntry(uri, name)
        self.roots[uri] = root

        logger.info(f"Added root: {uri} ({name})")
        self._notify_change()

    def remove_root(self, uri: str):
        """Remove root directory"""
        if uri in self.roots:
            del self.roots[uri]
            logger.info(f"Removed root: {uri}")
            self._notify_change()

    def list_roots(self) -> List[Dict[str, Any]]:
        """List all roots in MCP format"""
        return [root.to_dict() for root in self.roots.values()]

    def get_root(self, uri: str) -> Optional[RootEntry]:
        """Get root by URI"""
        return self.roots.get(uri)

    def is_path_allowed(self, path: str) -> bool:
        """Check if path is under any allowed root"""
        if not self._security_enabled:
            return True

        # Convert path to absolute
        abs_path = os.path.abspath(path)

        # Check against all roots
        for root_uri in self.roots:
            if root_uri.startswith("file://"):
                root_path = root_uri[7:]  # Remove file:// prefix
                root_abs = os.path.abspath(root_path)

                # Check if path is under this root
                try:
                    os.path.relpath(abs_path, root_abs)
                    if abs_path.startswith(root_abs):
                        return True
                except ValueError:
                    # Paths on different drives (Windows)
                    continue

        return False

    def resolve_relative_path(
        self, relative_path: str, base_root: Optional[str] = None
    ) -> Optional[str]:
        """Resolve relative path against roots"""
        if os.path.isabs(relative_path):
            return relative_path if self.is_path_allowed(relative_path) else None

        # Try base root first
        if base_root and base_root in self.roots:
            if base_root.startswith("file://"):
                base_path = base_root[7:]
                full_path = os.path.join(base_path, relative_path)
                if self.is_path_allowed(full_path):
                    return full_path

        # Try all roots
        for root_uri in self.roots:
            if root_uri.startswith("file://"):
                root_path = root_uri[7:]
                full_path = os.path.join(root_path, relative_path)
                if self.is_path_allowed(full_path) and os.path.exists(full_path):
                    return full_path

        return None

    def _is_valid_root_uri(self, uri: str) -> bool:
        """Check if URI is valid for root"""
        try:
            scheme = uri.split("://", 1)[0]
            return scheme in self._allowed_schemes
        except Exception:
            return False

    def _is_safe_path(self, uri: str) -> bool:
        """Check if path is safe (no path traversal, etc.)"""
        if not uri.startswith("file://"):
            return True  # Non-file URIs handled elsewhere

        path = uri[7:]  # Remove file:// prefix

        # Check for path traversal attempts
        if ".." in path:
            return False

        # Check for absolute path escapes
        abs_path = os.path.abspath(path)

        # Blocked paths
        blocked_patterns = [
            "/etc",
            "/root",
            "/usr/bin",
            "/bin",
            "/sbin",
            "C:\\Windows",
            "C:\\Program Files",
        ]

        for pattern in blocked_patterns:
            if abs_path.startswith(pattern):
                return False

        return True

    def on_change(self, handler: Callable):
        """Register change handler"""
        self._change_handlers.append(handler)

    def _notify_change(self):
        """Notify handlers of root list change"""
        for handler in self._change_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler())
                else:
                    handler()
            except Exception as e:
                logger.error(f"Error in root change handler: {e}")

    def set_security_enabled(self, enabled: bool):
        """Enable/disable security checks"""
        self._security_enabled = enabled
        logger.info(f"Root security {'enabled' if enabled else 'disabled'}")

    def add_allowed_scheme(self, scheme: str):
        """Add allowed URI scheme"""
        self._allowed_schemes.add(scheme)
        logger.info(f"Added allowed scheme: {scheme}")

    def scan_directory(self, root_uri: str, max_depth: int = 3) -> Dict[str, Any]:
        """Scan directory structure under root"""
        if root_uri not in self.roots:
            raise ValueError(f"Root not found: {root_uri}")

        if not root_uri.startswith("file://"):
            raise ValueError("Directory scanning only supported for file:// roots")

        root_path = root_uri[7:]

        def scan_recursive(path: str, depth: int) -> Dict[str, Any]:
            if depth > max_depth:
                return {"truncated": True}

            result = {
                "type": "directory" if os.path.isdir(path) else "file",
                "name": os.path.basename(path),
                "path": path,
                "size": 0,
                "modified": None,
            }

            try:
                stat = os.stat(path)
                result["size"] = stat.st_size
                result["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

                if os.path.isdir(path) and depth < max_depth:
                    children = []
                    try:
                        for item in os.listdir(path):
                            item_path = os.path.join(path, item)
                            if self.is_path_allowed(item_path):
                                child = scan_recursive(item_path, depth + 1)
                                children.append(child)
                    except PermissionError:
                        result["error"] = "Permission denied"

                    result["children"] = children
                    result["child_count"] = len(children)

            except Exception as e:
                result["error"] = str(e)

            return result

        return scan_recursive(root_path, 0)

    def search_files(
        self, pattern: str, root_uri: Optional[str] = None, max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for files matching pattern"""
        import fnmatch

        results = []
        roots_to_search = [root_uri] if root_uri else list(self.roots.keys())

        for root in roots_to_search:
            if not root.startswith("file://"):
                continue

            root_path = root[7:]

            try:
                for root_dir, dirs, files in os.walk(root_path):
                    if not self.is_path_allowed(root_dir):
                        dirs.clear()  # Don't descend into disallowed directories
                        continue

                    for file in files:
                        if fnmatch.fnmatch(file, pattern):
                            file_path = os.path.join(root_dir, file)

                            if self.is_path_allowed(file_path):
                                try:
                                    stat = os.stat(file_path)
                                    results.append(
                                        {
                                            "name": file,
                                            "path": file_path,
                                            "uri": f"file://{file_path}",
                                            "size": stat.st_size,
                                            "modified": datetime.fromtimestamp(
                                                stat.st_mtime
                                            ).isoformat(),
                                            "root": root,
                                        }
                                    )

                                    if len(results) >= max_results:
                                        return results

                                except Exception as e:
                                    logger.debug(f"Error accessing file {file_path}: {e}")

            except Exception as e:
                logger.error(f"Error searching root {root}: {e}")

        return results

    def get_directory_info(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get information about a directory"""
        if not uri.startswith("file://"):
            return None

        path = uri[7:]

        if not self.is_path_allowed(path) or not os.path.exists(path):
            return None

        try:
            stat = os.stat(path)

            info = {
                "uri": uri,
                "path": path,
                "type": "directory" if os.path.isdir(path) else "file",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:],
                "exists": True,
            }

            if os.path.isdir(path):
                try:
                    items = os.listdir(path)
                    info["item_count"] = len(items)
                    info["items"] = items[:50]  # Limit to first 50 items
                    if len(items) > 50:
                        info["truncated"] = True
                except PermissionError:
                    info["error"] = "Permission denied"

            return info

        except Exception as e:
            return {"uri": uri, "path": path, "exists": False, "error": str(e)}

    def export_config(self) -> Dict[str, Any]:
        """Export roots configuration"""
        return {
            "roots": [root.to_dict() for root in self.roots.values()],
            "security_enabled": self._security_enabled,
            "allowed_schemes": list(self._allowed_schemes),
        }

    def import_config(self, config: Dict[str, Any]):
        """Import roots configuration"""
        # Clear existing roots
        self.roots.clear()

        # Set security setting
        self._security_enabled = config.get("security_enabled", True)

        # Set allowed schemes
        if "allowed_schemes" in config:
            self._allowed_schemes = set(config["allowed_schemes"])

        # Add roots
        for root_data in config.get("roots", []):
            try:
                self.add_root(root_data["uri"], root_data.get("name"))
            except Exception as e:
                logger.error(f"Error importing root {root_data}: {e}")

        self._notify_change()


# Global root manager
root_manager = RootManager()


# Helper functions
def add_project_root(project_path: str, name: Optional[str] = None):
    """Add project directory as root"""
    abs_path = os.path.abspath(project_path)
    uri = f"file://{abs_path}"
    root_manager.add_root(uri, name or f"Project: {os.path.basename(abs_path)}")


def add_workspace_roots(workspace_config: Dict[str, Any]):
    """Add workspace roots from configuration"""
    for root_config in workspace_config.get("roots", []):
        try:
            root_manager.add_root(root_config["uri"], root_config.get("name"))
        except Exception as e:
            logger.error(f"Error adding workspace root: {e}")


def create_temp_root() -> str:
    """Create temporary root directory"""
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="mcp_root_")
    uri = f"file://{temp_dir}"
    root_manager.add_root(uri, "Temporary Root")

    return uri


def is_safe_path(path: str) -> bool:
    """Check if path is safe to access"""
    return root_manager.is_path_allowed(path)


def resolve_path(path: str, base_root: Optional[str] = None) -> Optional[str]:
    """Resolve path relative to roots"""
    return root_manager.resolve_relative_path(path, base_root)
