"""
Vercel Blob storage adapter for ML models
Handles full model persistence with versioning
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class VercelBlobModelStorage:
    """Storage adapter for ML models using Vercel Blob"""

    def __init__(self):
        self.blob_url = os.environ.get("VERCEL_BLOB_URL", "")
        self.blob_token = os.environ.get("VERCEL_BLOB_READ_WRITE_TOKEN", "")
        self.is_production = os.environ.get("VERCEL_ENV") == "production"
        self.local_storage_path = "/tmp/ml_models"

        # Create local storage directory
        if not self.is_production:
            os.makedirs(self.local_storage_path, exist_ok=True)

    async def upload_model(
        self, model_id: str, version: str, model_data: bytes, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Upload a model to Vercel Blob or local storage"""
        try:
            key = f"models/{model_id}/{version}/model.pkl"
            metadata_key = f"models/{model_id}/{version}/metadata.json"

            if self.is_production and self.blob_url:
                # Production: Upload to Vercel Blob
                async with aiohttp.ClientSession() as session:
                    # Upload model data
                    headers = {
                        "Authorization": f"Bearer {self.blob_token}",
                        "Content-Type": "application/octet-stream",
                    }

                    async with session.put(
                        f"{self.blob_url}/{key}", data=model_data, headers=headers
                    ) as response:
                        if response.status != 200:
                            raise ValueError(f"Failed to upload model: {response.status}")
                        model_result = await response.json()

                    # Upload metadata
                    headers["Content-Type"] = "application/json"
                    async with session.put(
                        f"{self.blob_url}/{metadata_key}",
                        data=json.dumps(metadata),
                        headers=headers,
                    ) as response:
                        if response.status != 200:
                            raise ValueError(f"Failed to upload metadata: {response.status}")
                        metadata_result = await response.json()

                return {
                    "model_url": model_result.get("url"),
                    "metadata_url": metadata_result.get("url"),
                    "size": len(model_data),
                    "uploaded_at": datetime.now().isoformat(),
                }

            else:
                # Local development: Save to disk
                model_path = os.path.join(
                    self.local_storage_path, f"{model_id}_{version}_model.pkl"
                )
                metadata_path = os.path.join(
                    self.local_storage_path, f"{model_id}_{version}_metadata.json"
                )

                # Save model
                with open(model_path, "wb") as f:
                    f.write(model_data)

                # Save metadata
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)

                return {
                    "model_path": model_path,
                    "metadata_path": metadata_path,
                    "size": len(model_data),
                    "uploaded_at": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            raise

    async def download_model(self, model_id: str, version: str) -> Optional[bytes]:
        """Download a model from Vercel Blob or local storage"""
        try:
            key = f"models/{model_id}/{version}/model.pkl"

            if self.is_production and self.blob_url:
                # Production: Download from Vercel Blob
                async with aiohttp.ClientSession() as session:
                    headers = {"Authorization": f"Bearer {self.blob_token}"}

                    async with session.get(f"{self.blob_url}/{key}", headers=headers) as response:
                        if response.status == 200:
                            return await response.read()
                        elif response.status == 404:
                            return None
                        else:
                            raise ValueError(f"Failed to download model: {response.status}")

            else:
                # Local development: Load from disk
                model_path = os.path.join(
                    self.local_storage_path, f"{model_id}_{version}_model.pkl"
                )

                if os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        return f.read()

                return None

        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return None

    async def download_metadata(self, model_id: str, version: str) -> Optional[Dict[str, Any]]:
        """Download model metadata"""
        try:
            key = f"models/{model_id}/{version}/metadata.json"

            if self.is_production and self.blob_url:
                # Production: Download from Vercel Blob
                async with aiohttp.ClientSession() as session:
                    headers = {"Authorization": f"Bearer {self.blob_token}"}

                    async with session.get(f"{self.blob_url}/{key}", headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 404:
                            return None
                        else:
                            raise ValueError(f"Failed to download metadata: {response.status}")

            else:
                # Local development: Load from disk
                metadata_path = os.path.join(
                    self.local_storage_path, f"{model_id}_{version}_metadata.json"
                )

                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        return json.load(f)

                return None

        except Exception as e:
            logger.error(f"Error downloading metadata: {e}")
            return None

    async def list_models(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            models = []

            if self.is_production and self.blob_url:
                # Production: List from Vercel Blob
                async with aiohttp.ClientSession() as session:
                    headers = {"Authorization": f"Bearer {self.blob_token}"}
                    prefix = f"models/{model_id}/" if model_id else "models/"

                    async with session.get(
                        f"{self.blob_url}?prefix={prefix}", headers=headers
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            blobs = data.get("blobs", [])

                            # Parse blob paths to extract model info
                            for blob in blobs:
                                if blob["pathname"].endswith("/metadata.json"):
                                    parts = blob["pathname"].split("/")
                                    if len(parts) >= 4:
                                        models.append(
                                            {
                                                "model_id": parts[1],
                                                "version": parts[2],
                                                "size": blob.get("size", 0),
                                                "uploaded_at": blob.get("uploadedAt"),
                                            }
                                        )

            else:
                # Local development: List from disk
                if os.path.exists(self.local_storage_path):
                    for filename in os.listdir(self.local_storage_path):
                        if filename.endswith("_metadata.json"):
                            parts = filename.replace("_metadata.json", "").split("_")
                            if len(parts) >= 2:
                                model_info = {
                                    "model_id": parts[0],
                                    "version": "_".join(parts[1:]),
                                    "uploaded_at": datetime.fromtimestamp(
                                        os.path.getmtime(
                                            os.path.join(self.local_storage_path, filename)
                                        )
                                    ).isoformat(),
                                }

                                # Get model size
                                model_filename = filename.replace("_metadata.json", "_model.pkl")
                                model_path = os.path.join(self.local_storage_path, model_filename)
                                if os.path.exists(model_path):
                                    model_info["size"] = os.path.getsize(model_path)

                                if not model_id or model_info["model_id"] == model_id:
                                    models.append(model_info)

            return sorted(models, key=lambda x: x.get("uploaded_at", ""), reverse=True)

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def delete_model(self, model_id: str, version: str) -> bool:
        """Delete a model from storage"""
        try:
            if self.is_production and self.blob_url:
                # Production: Delete from Vercel Blob
                async with aiohttp.ClientSession() as session:
                    headers = {"Authorization": f"Bearer {self.blob_token}"}

                    # Delete model and metadata
                    for suffix in ["model.pkl", "metadata.json"]:
                        key = f"models/{model_id}/{version}/{suffix}"

                        async with session.delete(
                            f"{self.blob_url}/{key}", headers=headers
                        ) as response:
                            if response.status not in [200, 404]:
                                logger.error(f"Failed to delete {key}: {response.status}")
                                return False

                return True

            else:
                # Local development: Delete from disk
                model_path = os.path.join(
                    self.local_storage_path, f"{model_id}_{version}_model.pkl"
                )
                metadata_path = os.path.join(
                    self.local_storage_path, f"{model_id}_{version}_metadata.json"
                )

                deleted = False
                if os.path.exists(model_path):
                    os.remove(model_path)
                    deleted = True

                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    deleted = True

                return deleted

        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False


class ModelVersionManager:
    """Manages model versions and lifecycle"""

    def __init__(self, storage: VercelBlobModelStorage):
        self.storage = storage
        self.version_cache = {}

    async def get_latest_version(self, model_id: str) -> Optional[str]:
        """Get the latest version of a model"""
        models = await self.storage.list_models(model_id)
        if models:
            return models[0]["version"]
        return None

    async def promote_version(self, model_id: str, version: str) -> bool:
        """Promote a model version to production"""
        try:
            # Download the model and metadata
            model_data = await self.storage.download_model(model_id, version)
            metadata = await self.storage.download_metadata(model_id, version)

            if not model_data or not metadata:
                return False

            # Update metadata with promotion info
            metadata["promoted_at"] = datetime.now().isoformat()
            metadata["status"] = "production"

            # Create a production version
            prod_version = f"prod_{int(datetime.now().timestamp())}"

            # Upload as production version
            await self.storage.upload_model(model_id, prod_version, model_data, metadata)

            logger.info(f"Promoted {model_id}:{version} to production as {prod_version}")
            return True

        except Exception as e:
            logger.error(f"Error promoting version: {e}")
            return False

    async def cleanup_old_versions(self, model_id: str, keep_count: int = 5) -> int:
        """Clean up old model versions, keeping the most recent ones"""
        try:
            models = await self.storage.list_models(model_id)

            if len(models) <= keep_count:
                return 0

            # Sort by upload time, newest first
            models.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)

            # Delete old versions
            deleted = 0
            for model in models[keep_count:]:
                if await self.storage.delete_model(model["model_id"], model["version"]):
                    deleted += 1
                    logger.info(f"Deleted old version {model['model_id']}:{model['version']}")

            return deleted

        except Exception as e:
            logger.error(f"Error cleaning up versions: {e}")
            return 0
