"""
Local file system storage implementation
"""

import os
import asyncio
from pathlib import Path
from typing import List, Optional, Union
import aiofiles
import shutil

from .base import StorageInterface, SyncStorageInterface


class LocalFileStorage(StorageInterface):
    """Async local file system storage implementation"""
    
    def __init__(self, base_path: str = "data/storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, key: str) -> Path:
        """Convert key to file path"""
        # Ensure key doesn't escape base directory
        safe_key = key.replace("..", "").lstrip("/")
        return self.base_path / safe_key
    
    async def read(self, key: str) -> Optional[bytes]:
        """Read data from file"""
        file_path = self._get_file_path(key)
        if not file_path.exists():
            return None
        
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
        except Exception:
            return None
    
    async def write(self, key: str, data: Union[bytes, str]) -> None:
        """Write data to file"""
        file_path = self._get_file_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)
    
    async def delete(self, key: str) -> None:
        """Delete file"""
        file_path = self._get_file_path(key)
        if file_path.exists():
            await asyncio.to_thread(file_path.unlink)
    
    async def exists(self, key: str) -> bool:
        """Check if file exists"""
        file_path = self._get_file_path(key)
        return await asyncio.to_thread(file_path.exists)
    
    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all keys with optional prefix filter"""
        keys = []
        search_path = self.base_path
        
        if prefix:
            search_path = self._get_file_path(prefix).parent
        
        def _walk_files():
            for root, _, files in os.walk(search_path):
                root_path = Path(root)
                for file in files:
                    file_path = root_path / file
                    relative_path = file_path.relative_to(self.base_path)
                    key = str(relative_path).replace(os.sep, '/')
                    
                    if prefix is None or key.startswith(prefix):
                        keys.append(key)
            return keys
        
        return await asyncio.to_thread(_walk_files)


class LocalFileStorageSync(SyncStorageInterface):
    """Synchronous local file system storage implementation"""
    
    def __init__(self, base_path: str = "data/storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, key: str) -> Path:
        """Convert key to file path"""
        # Ensure key doesn't escape base directory
        safe_key = key.replace("..", "").lstrip("/")
        return self.base_path / safe_key
    
    def read(self, key: str) -> Optional[bytes]:
        """Read data from file"""
        file_path = self._get_file_path(key)
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception:
            return None
    
    def write(self, key: str, data: Union[bytes, str]) -> None:
        """Write data to file"""
        file_path = self._get_file_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        with open(file_path, 'wb') as f:
            f.write(data)
    
    def delete(self, key: str) -> None:
        """Delete file"""
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
    
    def exists(self, key: str) -> bool:
        """Check if file exists"""
        file_path = self._get_file_path(key)
        return file_path.exists()
    
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all keys with optional prefix filter"""
        keys = []
        search_path = self.base_path
        
        if prefix:
            search_path = self._get_file_path(prefix).parent
            if not search_path.exists():
                return keys
        
        for root, _, files in os.walk(search_path):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                relative_path = file_path.relative_to(self.base_path)
                key = str(relative_path).replace(os.sep, '/')
                
                if prefix is None or key.startswith(prefix):
                    keys.append(key)
        
        return sorted(keys)