"""
Base storage interface for unified storage abstraction
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class StorageInterface(ABC):
    """Abstract base class for storage implementations"""

    @abstractmethod
    async def read(self, key: str) -> Optional[bytes]:
        """Read data from storage"""
        pass

    @abstractmethod
    async def write(self, key: str, data: Union[bytes, str]) -> None:
        """Write data to storage"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete data from storage"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in storage"""
        pass

    @abstractmethod
    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by prefix"""
        pass

    # Convenience methods with default implementations
    async def read_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Read and parse JSON data"""
        data = await self.read(key)
        if data:
            return json.loads(data.decode("utf-8"))
        return None

    async def write_json(self, key: str, data: Dict[str, Any]) -> None:
        """Write data as JSON"""
        json_str = json.dumps(data, indent=2)
        await self.write(key, json_str.encode("utf-8"))

    async def read_text(self, key: str) -> Optional[str]:
        """Read data as text"""
        data = await self.read(key)
        if data:
            return data.decode("utf-8")
        return None

    async def write_text(self, key: str, text: str) -> None:
        """Write text data"""
        await self.write(key, text.encode("utf-8"))


class SyncStorageInterface(ABC):
    """Synchronous version of storage interface for compatibility"""

    @abstractmethod
    def read(self, key: str) -> Optional[bytes]:
        """Read data from storage"""
        pass

    @abstractmethod
    def write(self, key: str, data: Union[bytes, str]) -> None:
        """Write data to storage"""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete data from storage"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in storage"""
        pass

    @abstractmethod
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by prefix"""
        pass

    # Convenience methods
    def read_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Read and parse JSON data"""
        data = self.read(key)
        if data:
            return json.loads(data.decode("utf-8"))
        return None

    def write_json(self, key: str, data: Dict[str, Any]) -> None:
        """Write data as JSON"""
        json_str = json.dumps(data, indent=2)
        self.write(key, json_str.encode("utf-8"))

    def read_text(self, key: str) -> Optional[str]:
        """Read data as text"""
        data = self.read(key)
        if data:
            return data.decode("utf-8")
        return None

    def write_text(self, key: str, text: str) -> None:
        """Write text data"""
        self.write(key, text.encode("utf-8"))
