"""
Vercel KV storage adapter for caching and data persistence
Replaces Redis with Vercel's native KV store
"""

import base64
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Local cache directory for development
LOCAL_CACHE_DIR = Path(tempfile.gettempdir()) / "cryptotrading_cache"
LOCAL_CACHE_DIR.mkdir(exist_ok=True)


class VercelKVClient:
    """Vercel KV client that works locally and on Vercel"""

    def __init__(self):
        self.kv_url = os.environ.get("KV_REST_API_URL")
        self.kv_token = os.environ.get("KV_REST_API_TOKEN")
        self.fallback_cache = {}  # In-memory cache for local development
        self.is_local = os.environ.get("VERCEL_ENV") != "production"

        if self.is_local:
            logger.info("Running locally - using in-memory cache")
            self.use_fallback = True
        elif not self.kv_url or not self.kv_token:
            logger.warning("Vercel KV not configured, using in-memory fallback")
            self.use_fallback = True
        else:
            self.use_fallback = False
            logger.info("Vercel KV configured successfully")

    async def get(self, key: str) -> Optional[str]:
        """Get value from KV store"""
        if self.use_fallback:
            return self._get_fallback(key)

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.kv_url}/get/{key}", headers={"Authorization": f"Bearer {self.kv_token}"}
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("result")
                elif response.status_code == 404:
                    return None
                else:
                    logger.error(f"KV get error: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"KV get error: {e}")
            return self._get_fallback(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value in KV store with optional expiration"""
        if self.use_fallback:
            return self._set_fallback(key, value, ex)

        try:
            import httpx

            # Prepare data
            data = {"value": value}
            if ex:
                data["ex"] = ex

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.kv_url}/set/{key}",
                    headers={"Authorization": f"Bearer {self.kv_token}"},
                    json=data,
                )

                if response.status_code == 200:
                    return True
                else:
                    logger.error(f"KV set error: {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"KV set error: {e}")
            return self._set_fallback(key, value, ex)

    async def setex(self, key: str, seconds: int, value: str) -> bool:
        """Set value with expiration (Redis-compatible method)"""
        return await self.set(key, value, ex=seconds)

    async def delete(self, key: str) -> bool:
        """Delete key from KV store"""
        if self.use_fallback:
            return self._delete_fallback(key)

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.kv_url}/delete/{key}",
                    headers={"Authorization": f"Bearer {self.kv_token}"},
                )

                return response.status_code == 200

        except Exception as e:
            logger.error(f"KV delete error: {e}")
            return self._delete_fallback(key)

    def ping(self) -> bool:
        """Check if KV store is available"""
        if self.use_fallback:
            return True

        return bool(self.kv_url and self.kv_token)

    # Fallback methods for local development
    def _get_fallback(self, key: str) -> Optional[str]:
        """Get from local file cache (persistent) or memory cache"""
        # Try file cache first (persistent across restarts)
        try:
            cache_file = LOCAL_CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    entry = json.load(f)

                # Check expiration
                if entry.get("expires_at"):
                    expires = datetime.fromisoformat(entry["expires_at"])
                    if datetime.now() > expires:
                        cache_file.unlink()
                        return None

                return entry["value"]
        except Exception as e:
            logger.debug(f"File cache read error: {e}")

        # Fallback to memory cache
        entry = self.fallback_cache.get(key)
        if entry:
            if entry.get("expires_at") and datetime.now() > entry["expires_at"]:
                del self.fallback_cache[key]
                return None
            return entry["value"]

        return None

    def _set_fallback(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set in local file cache (persistent) and memory cache"""
        entry = {"value": value}
        if ex:
            expires_at = datetime.now() + timedelta(seconds=ex)
            entry["expires_at"] = expires_at.isoformat()

        # Save to file cache (persistent)
        try:
            cache_file = LOCAL_CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            with open(cache_file, "w") as f:
                json.dump(entry, f)
        except Exception as e:
            logger.debug(f"File cache write error: {e}")

        # Also save to memory cache
        if ex:
            entry["expires_at"] = datetime.now() + timedelta(seconds=ex)
        self.fallback_cache[key] = entry

        return True

    def _delete_fallback(self, key: str) -> bool:
        """Delete from both file and memory cache"""
        success = False

        # Delete from file cache
        try:
            cache_file = LOCAL_CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            if cache_file.exists():
                cache_file.unlink()
                success = True
        except Exception as e:
            logger.debug(f"File cache delete error: {e}")

        # Delete from memory cache
        if key in self.fallback_cache:
            del self.fallback_cache[key]
            success = True

        return success


class VercelCache:
    """High-level caching interface for ML predictions"""

    def __init__(self):
        self.kv = VercelKVClient()
        self.default_ttl = 300  # 5 minutes

    async def get_prediction(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction"""
        try:
            cached = await self.kv.get(f"prediction:{cache_key}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")

        return None

    async def set_prediction(
        self, cache_key: str, prediction: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Cache prediction with TTL"""
        try:
            ttl = ttl or self.default_ttl
            data = json.dumps(prediction)
            return await self.kv.setex(f"prediction:{cache_key}", ttl, data)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def get_features(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached features"""
        try:
            cached = await self.kv.get(f"features:{cache_key}")
            if cached:
                # Features are stored as base64 encoded pickle for efficiency
                import base64
                import pickle

                data = base64.b64decode(cached.encode())
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Feature cache get error: {e}")

        return None

    async def set_features(
        self, cache_key: str, features: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Cache features with TTL"""
        try:
            import base64
            import pickle

            ttl = ttl or self.default_ttl * 4  # Features cache longer

            # Serialize and encode features
            data = pickle.dumps(features)
            encoded = base64.b64encode(data).decode()

            return await self.kv.setex(f"features:{cache_key}", ttl, encoded)
        except Exception as e:
            logger.error(f"Feature cache set error: {e}")
            return False

    def generate_cache_key(self, *args) -> str:
        """Generate consistent cache key"""
        key_str = ":".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()


# Global cache instance
vercel_cache = VercelCache()


def get_cache_client() -> VercelCache:
    """Get the global cache client"""
    return vercel_cache
