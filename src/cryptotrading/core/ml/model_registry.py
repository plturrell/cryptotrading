"""
ML Model Registry with Database Integration
Manages model lifecycle, versioning, and deployment tracking
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from ...infrastructure.database.unified_database import UnifiedDatabase
from .model_storage import VercelBlobModelStorage

logger = logging.getLogger(__name__)


class MLModelRegistry:
    """
    Centralized ML model registry with database tracking
    Handles model storage, versioning, and deployment
    """

    def __init__(self, db: Optional[UnifiedDatabase] = None):
        self.db = db or UnifiedDatabase()
        self.storage = VercelBlobModelStorage()
        self._cache = {}  # In-memory model cache
        self._cache_size = 10

    async def register_model(
        self,
        model_id: str,
        version: str,
        model_type: str,
        model_data: bytes,
        algorithm: str,
        parameters: Dict[str, Any],
        training_metrics: Dict[str, float],
        validation_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Register a new model in the registry

        Returns:
            Model registry ID
        """
        try:
            # Upload model to storage
            storage_result = await self.storage.upload_model(
                model_id,
                version,
                model_data,
                {
                    "model_type": model_type,
                    "algorithm": algorithm,
                    "parameters": parameters,
                    "training_metrics": training_metrics,
                    "validation_metrics": validation_metrics,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )

            # Store in database
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO ml_model_registry 
                    (model_id, version, model_type, algorithm, parameters,
                     training_metrics, validation_metrics, file_path, blob_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        model_id,
                        version,
                        model_type,
                        algorithm,
                        json.dumps(parameters),
                        json.dumps(training_metrics),
                        json.dumps(validation_metrics) if validation_metrics else None,
                        storage_result.get("local_path"),
                        storage_result.get("blob_url"),
                    ),
                )

                registry_id = cursor.lastrowid
                conn.commit()

            logger.info(f"Registered model {model_id} version {version} with ID {registry_id}")
            return str(registry_id)

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    async def deploy_model(self, model_id: str, version: str) -> bool:
        """Mark a model as deployed"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE ml_model_registry 
                    SET status = 'deployed', deployed_at = ?
                    WHERE model_id = ? AND version = ?
                """,
                    (datetime.utcnow(), model_id, version),
                )

                conn.commit()

            logger.info(f"Deployed model {model_id} version {version}")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False

    async def get_model(
        self, model_id: str, version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get model metadata and download URL
        If version not specified, gets latest deployed version
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                if version:
                    cursor.execute(
                        """
                        SELECT * FROM ml_model_registry 
                        WHERE model_id = ? AND version = ?
                    """,
                        (model_id, version),
                    )
                else:
                    # Get latest deployed version
                    cursor.execute(
                        """
                        SELECT * FROM ml_model_registry 
                        WHERE model_id = ? AND status = 'deployed'
                        ORDER BY deployed_at DESC
                        LIMIT 1
                    """,
                        (model_id,),
                    )

                row = cursor.fetchone()

                if row:
                    return {
                        "registry_id": row[0],
                        "model_id": row[1],
                        "version": row[2],
                        "model_type": row[3],
                        "algorithm": row[4],
                        "parameters": json.loads(row[5]) if row[5] else {},
                        "training_metrics": json.loads(row[6]) if row[6] else {},
                        "validation_metrics": json.loads(row[7]) if row[7] else {},
                        "file_path": row[8],
                        "blob_url": row[9],
                        "created_at": row[10],
                        "deployed_at": row[11],
                        "status": row[12],
                    }

                return None

        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            return None

    async def load_model(self, model_id: str, version: Optional[str] = None) -> Optional[Any]:
        """Load actual model object from storage"""
        cache_key = f"{model_id}:{version or 'latest'}"

        # Check cache
        if cache_key in self._cache:
            logger.info(f"Loading model {cache_key} from cache")
            return self._cache[cache_key]

        # Get model metadata
        model_info = await self.get_model(model_id, version)
        if not model_info:
            return None

        # Download model
        model_data = await self.storage.download_model(model_id, model_info["version"])

        if model_data:
            model = pickle.loads(model_data)

            # Cache it
            self._cache[cache_key] = model
            if len(self._cache) > self._cache_size:
                # Remove oldest
                oldest = next(iter(self._cache))
                del self._cache[oldest]

            return model

        return None

    async def list_models(
        self, model_type: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all models with optional filtering"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM ml_model_registry WHERE 1=1"
                params = []

                if model_type:
                    query += " AND model_type = ?"
                    params.append(model_type)

                if status:
                    query += " AND status = ?"
                    params.append(status)

                query += " ORDER BY created_at DESC"

                cursor.execute(query, params)

                models = []
                for row in cursor.fetchall():
                    models.append(
                        {
                            "registry_id": row[0],
                            "model_id": row[1],
                            "version": row[2],
                            "model_type": row[3],
                            "algorithm": row[4],
                            "status": row[12],
                            "created_at": row[10],
                            "deployed_at": row[11],
                        }
                    )

                return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def get_model_performance_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Get performance metrics history for a model"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT version, training_metrics, validation_metrics, created_at
                    FROM ml_model_registry 
                    WHERE model_id = ?
                    ORDER BY created_at DESC
                """,
                    (model_id,),
                )

                history = []
                for row in cursor.fetchall():
                    history.append(
                        {
                            "version": row[0],
                            "training_metrics": json.loads(row[1]) if row[1] else {},
                            "validation_metrics": json.loads(row[2]) if row[2] else {},
                            "created_at": row[3],
                        }
                    )

                return history

        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return []

    async def cleanup_old_models(self, days: int = 30) -> int:
        """Remove old non-deployed models"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Get models to delete
                cursor.execute(
                    """
                    SELECT model_id, version, file_path 
                    FROM ml_model_registry 
                    WHERE status != 'deployed' 
                    AND created_at < ?
                """,
                    (cutoff_date,),
                )

                models_to_delete = cursor.fetchall()

                # Delete from storage
                for model_id, version, file_path in models_to_delete:
                    try:
                        await self.storage.delete_model(model_id, version)
                    except Exception as e:
                        logger.error(f"Failed to delete model file: {e}")

                # Delete from database
                cursor.execute(
                    """
                    DELETE FROM ml_model_registry 
                    WHERE status != 'deployed' 
                    AND created_at < ?
                """,
                    (cutoff_date,),
                )

                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted_count} old models")
                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")
            return 0


# Global registry instance
_model_registry: Optional[MLModelRegistry] = None


async def get_model_registry() -> MLModelRegistry:
    """Get global model registry instance"""
    global _model_registry
    if _model_registry is None:
        _model_registry = MLModelRegistry()
    return _model_registry
