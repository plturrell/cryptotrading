"""
Storage factory for environment-aware storage selection
"""

import os
from typing import Optional, Union

from .base import StorageInterface, SyncStorageInterface
from .local import LocalFileStorage, LocalFileStorageSync
from .vercel import VercelBlobStorage, VercelBlobStorageSync, VercelKVStorage


class StorageFactory:
    """Factory for creating appropriate storage implementation based on environment"""

    @staticmethod
    def get_storage(
        storage_type: Optional[str] = None, sync: bool = False
    ) -> Union[StorageInterface, SyncStorageInterface]:
        """
        Get storage implementation based on environment

        Args:
            storage_type: Override storage type ('local', 'vercel_blob', 'vercel_kv')
            sync: Whether to return synchronous implementation

        Returns:
            Storage implementation instance
        """
        # Determine storage type from environment if not specified
        if storage_type is None:
            if os.environ.get("VERCEL"):
                # On Vercel, prefer KV for small data, Blob for large data
                if os.environ.get("KV_REST_API_URL"):
                    storage_type = "vercel_kv"
                elif os.environ.get("BLOB_READ_WRITE_TOKEN"):
                    storage_type = "vercel_blob"
                else:
                    # Fallback to local even on Vercel if no storage configured
                    storage_type = "local"
            else:
                storage_type = "local"

        # Create appropriate storage instance
        if storage_type == "local":
            base_path = os.environ.get("STORAGE_PATH", "data/storage")
            return LocalFileStorageSync(base_path) if sync else LocalFileStorage(base_path)

        elif storage_type == "vercel_blob":
            return VercelBlobStorageSync() if sync else VercelBlobStorage()

        elif storage_type == "vercel_kv":
            if sync:
                raise NotImplementedError("Synchronous Vercel KV not implemented yet")
            return VercelKVStorage()

        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

    @staticmethod
    def get_async_storage(storage_type: Optional[str] = None) -> StorageInterface:
        """Get async storage implementation"""
        return StorageFactory.get_storage(storage_type, sync=False)

    @staticmethod
    def get_sync_storage(storage_type: Optional[str] = None) -> SyncStorageInterface:
        """Get synchronous storage implementation"""
        return StorageFactory.get_storage(storage_type, sync=True)


# Convenience function for backward compatibility
def get_storage() -> StorageInterface:
    """Get default async storage for current environment"""
    return StorageFactory.get_async_storage()


def get_sync_storage() -> SyncStorageInterface:
    """Get default sync storage for current environment"""
    return StorageFactory.get_sync_storage()
