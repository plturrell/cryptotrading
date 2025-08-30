"""
Feature Cache Persistence Layer
Stores computed ML features in database for reuse
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ...infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)


class FeatureCachePersistence:
    """
    Persists computed ML features to database
    Reduces computation time and tracks feature evolution
    """

    def __init__(self, db: Optional[UnifiedDatabase] = None):
        self.db = db or UnifiedDatabase()
        self._batch_buffer = []
        self._batch_size = 100

    def _generate_cache_key(self, symbol: str, features: List[str], timestamp: datetime) -> str:
        """Generate unique cache key for feature set"""
        key_str = f"{symbol}:{','.join(sorted(features))}:{timestamp.date()}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def store_features(
        self, symbol: str, features: Dict[str, float], timestamp: datetime, version: str = "1.0"
    ) -> bool:
        """Store computed features in cache"""
        try:
            feature_list = list(features.keys())
            cache_key = self._generate_cache_key(symbol, feature_list, timestamp)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO feature_cache
                    (cache_key, symbol, feature_names, feature_values, 
                     feature_count, version, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        cache_key,
                        symbol,
                        json.dumps(feature_list),
                        json.dumps(features),
                        len(features),
                        version,
                        timestamp,
                        timestamp + timedelta(hours=24),  # 24 hour expiry
                    ),
                )

                conn.commit()

            logger.debug(f"Stored {len(features)} features for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to store features: {e}")
            return False

    async def get_features(
        self, symbol: str, feature_names: List[str], timestamp: datetime, tolerance_hours: int = 1
    ) -> Optional[Dict[str, float]]:
        """Retrieve cached features if available"""
        try:
            # Look for features within tolerance window
            start_time = timestamp - timedelta(hours=tolerance_hours)
            end_time = timestamp + timedelta(hours=tolerance_hours)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT feature_values, created_at
                    FROM feature_cache
                    WHERE symbol = ?
                    AND created_at >= ?
                    AND created_at <= ?
                    AND expires_at > ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """,
                    (symbol, start_time, end_time, datetime.utcnow()),
                )

                row = cursor.fetchone()

                if row:
                    features = json.loads(row[0])
                    # Filter to requested features
                    return {k: v for k, v in features.items() if k in feature_names}

            return None

        except Exception as e:
            logger.error(f"Failed to get cached features: {e}")
            return None

    async def batch_store(self, feature_batches: List[Dict[str, Any]]) -> int:
        """Store multiple feature sets efficiently"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                data = []
                for batch in feature_batches:
                    feature_list = list(batch["features"].keys())
                    cache_key = self._generate_cache_key(
                        batch["symbol"], feature_list, batch["timestamp"]
                    )

                    data.append(
                        (
                            cache_key,
                            batch["symbol"],
                            json.dumps(feature_list),
                            json.dumps(batch["features"]),
                            len(batch["features"]),
                            batch.get("version", "1.0"),
                            batch["timestamp"],
                            batch["timestamp"] + timedelta(hours=24),
                        )
                    )

                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO feature_cache
                    (cache_key, symbol, feature_names, feature_values,
                     feature_count, version, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    data,
                )

                conn.commit()

                logger.info(f"Batch stored {len(data)} feature sets")
                return len(data)

        except Exception as e:
            logger.error(f"Failed to batch store features: {e}")
            return 0

    async def get_feature_history(
        self, symbol: str, feature_name: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Get historical values for a specific feature"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT created_at, feature_values
                    FROM feature_cache
                    WHERE symbol = ?
                    AND created_at >= ?
                    AND created_at <= ?
                    AND feature_names LIKE ?
                    ORDER BY created_at
                """,
                    (symbol, start_date, end_date, f'%"{feature_name}"%'),
                )

                data = []
                for row in cursor.fetchall():
                    timestamp = row[0]
                    features = json.loads(row[1])
                    if feature_name in features:
                        data.append({"timestamp": timestamp, "value": features[feature_name]})

                if data:
                    df = pd.DataFrame(data)
                    df.set_index("timestamp", inplace=True)
                    return df

                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get feature history: {e}")
            return pd.DataFrame()

    async def get_feature_statistics(
        self, symbol: str, feature_name: str, days: int = 30
    ) -> Dict[str, float]:
        """Get statistics for a feature over time"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT feature_values
                    FROM feature_cache
                    WHERE symbol = ?
                    AND created_at >= ?
                    AND feature_names LIKE ?
                """,
                    (symbol, start_date, f'%"{feature_name}"%'),
                )

                values = []
                for row in cursor.fetchall():
                    features = json.loads(row[0])
                    if feature_name in features:
                        values.append(features[feature_name])

                if values:
                    return {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values),
                        "count": len(values),
                    }

                return {}

        except Exception as e:
            logger.error(f"Failed to get feature statistics: {e}")
            return {}

    async def cleanup_expired(self, days_to_keep: int = 7) -> int:
        """Remove expired feature cache entries"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    DELETE FROM feature_cache
                    WHERE expires_at < ? OR created_at < ?
                """,
                    (datetime.utcnow(), cutoff_date),
                )

                deleted = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted} expired feature cache entries")
                return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup expired features: {e}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Total entries
                cursor.execute("SELECT COUNT(*) FROM feature_cache")
                total_entries = cursor.fetchone()[0]

                # Active entries
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM feature_cache
                    WHERE expires_at > ?
                """,
                    (datetime.utcnow(),),
                )
                active_entries = cursor.fetchone()[0]

                # Unique symbols
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM feature_cache")
                unique_symbols = cursor.fetchone()[0]

                # Average features per entry
                cursor.execute("SELECT AVG(feature_count) FROM feature_cache")
                avg_features = cursor.fetchone()[0] or 0

                # Cache size estimate (bytes)
                cursor.execute(
                    """
                    SELECT SUM(LENGTH(feature_values) + LENGTH(feature_names))
                    FROM feature_cache
                """
                )
                cache_size = cursor.fetchone()[0] or 0

                return {
                    "total_entries": total_entries,
                    "active_entries": active_entries,
                    "expired_entries": total_entries - active_entries,
                    "unique_symbols": unique_symbols,
                    "avg_features_per_entry": round(avg_features, 2),
                    "cache_size_bytes": cache_size,
                    "cache_size_mb": round(cache_size / 1024 / 1024, 2),
                }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


# Integration with existing FeatureStore
class FeatureStoreIntegration:
    """Integrates feature cache with feature store"""

    def __init__(self, feature_store, cache: Optional[FeatureCachePersistence] = None):
        self.feature_store = feature_store
        self.cache = cache or FeatureCachePersistence()

    async def compute_features_with_cache(
        self, symbol: str, features: List[str], timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Compute features with caching"""
        timestamp = timestamp or datetime.utcnow()

        # Check cache first
        cached = await self.cache.get_features(symbol, features, timestamp)
        if cached and len(cached) == len(features):
            logger.info(f"Using cached features for {symbol}")
            return cached

        # Compute missing features
        logger.info(f"Computing features for {symbol}")
        computed = await self.feature_store.get_feature_vector(symbol, timestamp)

        # Store in cache
        if computed:
            await self.cache.store_features(symbol, computed, timestamp)

        return computed


# Global cache instance
_feature_cache: Optional[FeatureCachePersistence] = None


async def get_feature_cache() -> FeatureCachePersistence:
    """Get global feature cache instance"""
    global _feature_cache
    if _feature_cache is None:
        _feature_cache = FeatureCachePersistence()
    return _feature_cache
