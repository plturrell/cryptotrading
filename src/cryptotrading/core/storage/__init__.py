"""
Unified storage package for cryptotrading
Provides environment-aware storage implementations
"""

from .base import StorageInterface, SyncStorageInterface
from .factory import StorageFactory, get_storage, get_sync_storage
from .local import LocalFileStorage, LocalFileStorageSync
from .vercel import VercelBlobStorage, VercelBlobStorageSync, VercelKVStorage

# Legacy imports for backward compatibility
try:
    from .vercel_kv import VercelCache, VercelKVClient, get_cache_client, vercel_cache
except ImportError:
    VercelKVClient = VercelCache = vercel_cache = get_cache_client = None

__all__ = [
    # New unified interface
    "StorageInterface",
    "SyncStorageInterface",
    "StorageFactory",
    "get_storage",
    "get_sync_storage",
    "LocalFileStorage",
    "LocalFileStorageSync",
    "VercelBlobStorage",
    "VercelBlobStorageSync",
    "VercelKVStorage",
    # Legacy exports
    "VercelKVClient",
    "VercelCache",
    "vercel_cache",
    "get_cache_client",
]
