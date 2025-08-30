"""
Database-Backed Configuration Management
Stores configuration in database instead of files for better security and consistency
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)


@dataclass
class ConfigEntry:
    """Configuration entry with metadata"""

    key: str
    value: Any
    environment: str
    version: str
    created_at: datetime
    updated_at: datetime
    is_encrypted: bool = False
    metadata: Optional[Dict[str, Any]] = None


class DatabaseConfigManager:
    """
    Manages configuration in database instead of files
    Provides environment-specific configs with versioning
    """

    def __init__(self, db: Optional[UnifiedDatabase] = None):
        self.db = db or UnifiedDatabase()
        self._init_database()
        self._cache = {}
        self.environment = os.getenv("CRYPTOTRADING_ENV", "development")

    def _init_database(self):
        """Initialize configuration tables"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS app_configuration (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        config_key TEXT NOT NULL,
                        config_value TEXT NOT NULL,
                        environment TEXT NOT NULL,
                        version TEXT NOT NULL,
                        is_encrypted BOOLEAN DEFAULT FALSE,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(config_key, environment, version)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_config_env_key 
                    ON app_configuration(environment, config_key);
                    
                    CREATE TABLE IF NOT EXISTS config_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        config_key TEXT NOT NULL,
                        old_value TEXT,
                        new_value TEXT,
                        environment TEXT NOT NULL,
                        changed_by TEXT,
                        changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        change_reason TEXT
                    );
                """
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize config database: {e}")
            raise

    async def set_config(
        self,
        key: str,
        value: Any,
        environment: Optional[str] = None,
        version: str = "1.0.0",
        is_encrypted: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set configuration value"""
        env = environment or self.environment

        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value)
            else:
                serialized = str(value)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Check if exists
                cursor.execute(
                    """
                    SELECT config_value FROM app_configuration
                    WHERE config_key = ? AND environment = ? AND version = ?
                """,
                    (key, env, version),
                )

                old_value = cursor.fetchone()

                # Insert or update
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO app_configuration
                    (config_key, config_value, environment, version, 
                     is_encrypted, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                    (
                        key,
                        serialized,
                        env,
                        version,
                        is_encrypted,
                        json.dumps(metadata) if metadata else None,
                    ),
                )

                # Log history
                if old_value:
                    cursor.execute(
                        """
                        INSERT INTO config_history
                        (config_key, old_value, new_value, environment, changed_by)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (key, old_value[0], serialized, env, "system"),
                    )

                conn.commit()

                # Update cache
                cache_key = f"{env}:{key}:{version}"
                self._cache[cache_key] = value

                logger.info(f"Config set: {key} in {env} environment")
                return True

        except Exception as e:
            logger.error(f"Failed to set config: {e}")
            return False

    async def get_config(
        self,
        key: str,
        environment: Optional[str] = None,
        version: str = "1.0.0",
        default: Any = None,
    ) -> Any:
        """Get configuration value"""
        env = environment or self.environment
        cache_key = f"{env}:{key}:{version}"

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT config_value, is_encrypted FROM app_configuration
                    WHERE config_key = ? AND environment = ? AND version = ?
                """,
                    (key, env, version),
                )

                row = cursor.fetchone()

                if row:
                    value_str, is_encrypted = row

                    # Deserialize value
                    try:
                        value = json.loads(value_str)
                    except json.JSONDecodeError:
                        value = value_str

                    # Cache it
                    self._cache[cache_key] = value
                    return value

                # Try fallback to development environment
                if env != "development":
                    return await self.get_config(key, "development", version, default)

                return default

        except Exception as e:
            logger.error(f"Failed to get config: {e}")
            return default

    async def get_all_configs(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Get all configs for an environment"""
        env = environment or self.environment

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT config_key, config_value FROM app_configuration
                    WHERE environment = ?
                    ORDER BY config_key
                """,
                    (env,),
                )

                configs = {}
                for key, value_str in cursor.fetchall():
                    try:
                        configs[key] = json.loads(value_str)
                    except json.JSONDecodeError:
                        configs[key] = value_str

                return configs

        except Exception as e:
            logger.error(f"Failed to get all configs: {e}")
            return {}

    async def delete_config(
        self, key: str, environment: Optional[str] = None, version: str = "1.0.0"
    ) -> bool:
        """Delete configuration"""
        env = environment or self.environment

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    DELETE FROM app_configuration
                    WHERE config_key = ? AND environment = ? AND version = ?
                """,
                    (key, env, version),
                )

                conn.commit()

                # Remove from cache
                cache_key = f"{env}:{key}:{version}"
                self._cache.pop(cache_key, None)

                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to delete config: {e}")
            return False

    async def get_config_history(
        self, key: str, environment: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        env = environment or self.environment

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT * FROM config_history
                    WHERE config_key = ? AND environment = ?
                    ORDER BY changed_at DESC
                    LIMIT ?
                """,
                    (key, env, limit),
                )

                history = []
                for row in cursor.fetchall():
                    history.append(
                        {
                            "id": row[0],
                            "config_key": row[1],
                            "old_value": row[2],
                            "new_value": row[3],
                            "environment": row[4],
                            "changed_by": row[5],
                            "changed_at": row[6],
                            "change_reason": row[7],
                        }
                    )

                return history

        except Exception as e:
            logger.error(f"Failed to get config history: {e}")
            return []

    async def export_configs(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Export all configs for backup"""
        env = environment or self.environment

        configs = await self.get_all_configs(env)

        return {
            "environment": env,
            "exported_at": datetime.utcnow().isoformat(),
            "configs": configs,
        }

    async def import_configs(
        self, config_data: Dict[str, Any], environment: Optional[str] = None
    ) -> int:
        """Import configs from backup"""
        env = environment or config_data.get("environment", self.environment)
        imported = 0

        for key, value in config_data.get("configs", {}).items():
            if await self.set_config(key, value, env):
                imported += 1

        logger.info(f"Imported {imported} configs to {env} environment")
        return imported


# Singleton instance
_config_manager: Optional[DatabaseConfigManager] = None


async def get_config_manager() -> DatabaseConfigManager:
    """Get global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = DatabaseConfigManager()
    return _config_manager


# Helper functions for easy access
async def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    manager = await get_config_manager()
    return await manager.get_config(key, default=default)


async def set_config(key: str, value: Any) -> bool:
    """Set configuration value"""
    manager = await get_config_manager()
    return await manager.set_config(key, value)
