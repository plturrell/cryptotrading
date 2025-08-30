"""
Storage modules for cryptotrading.com
"""

from .vercel_blob import VercelBlobClient, put_blob, put_json_blob

__all__ = ["VercelBlobClient", "put_blob", "put_json_blob"]
