"""
Production-grade secret management with HashiCorp Vault integration
Supports both local development and production vault deployments
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import aiohttp
import base64
import secrets
from pathlib import Path

from ...core.security.credentials_manager import CredentialsManager

logger = logging.getLogger(__name__)

@dataclass
class Secret:
    """Secret with metadata"""
    key: str
    value: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    version: int = 1

class VaultConfig:
    """Vault configuration"""
    VAULT_URL = os.getenv("VAULT_URL", "https://localhost:8200")  # Force HTTPS for security
    VAULT_TOKEN = os.getenv("VAULT_TOKEN")
    
    @classmethod
    def validate_config(cls):
        """Validate vault configuration for security"""
        if not cls.VAULT_TOKEN:
            raise ValueError("VAULT_TOKEN is required")
        if cls.VAULT_URL.startswith("http://") and "localhost" not in cls.VAULT_URL:
            raise ValueError("HTTPS is required for non-localhost vault connections")
        return True
    VAULT_NAMESPACE = os.getenv("VAULT_NAMESPACE", "")
    KV_MOUNT_PATH = "secret"
    SECRET_PREFIX = "cryptotrading/a2a"
    LOCAL_VAULT_PATH = "/tmp/cryptotrading_vault"

class VaultError(Exception):
    """Vault operation failed"""
    pass

class SecretManager:
    """Production secret management with Vault backend"""
    
    def __init__(self, config: VaultConfig = None, use_local: bool = None):
        self.config = config or VaultConfig()
        
        # Determine if using local vault or remote
        if use_local is None:
            use_local = not bool(self.config.VAULT_TOKEN)
        
        self.use_local = use_local
        self.local_secrets: Dict[str, Secret] = {}
        
        # Initialize database credentials manager
        self.credentials_manager = CredentialsManager()
        
        if self.use_local:
            logger.warning("Using local secret storage - not suitable for production")
            self._init_local_vault()
        else:
            logger.info(f"Using Vault at {self.config.VAULT_URL}")
    
    def _init_local_vault(self):
        """Initialize local vault storage"""
        vault_path = Path(self.config.LOCAL_VAULT_PATH)
        vault_path.mkdir(exist_ok=True, parents=True)
        
        # Load existing secrets
        secrets_file = vault_path / "secrets.json"
        if secrets_file.exists():
            try:
                with open(secrets_file, 'r') as f:
                    data = json.load(f)
                    for key, secret_data in data.items():
                        secret_data['created_at'] = datetime.fromisoformat(secret_data['created_at'])
                        if secret_data.get('expires_at'):
                            secret_data['expires_at'] = datetime.fromisoformat(secret_data['expires_at'])
                        self.local_secrets[key] = Secret(**secret_data)
            except Exception as e:
                logger.error(f"Failed to load local secrets: {e}")
    
    def _save_local_vault(self):
        """Save local vault storage"""
        vault_path = Path(self.config.LOCAL_VAULT_PATH)
        secrets_file = vault_path / "secrets.json"
        
        try:
            data = {}
            for key, secret in self.local_secrets.items():
                secret_dict = asdict(secret)
                secret_dict['created_at'] = secret.created_at.isoformat()
                if secret.expires_at:
                    secret_dict['expires_at'] = secret.expires_at.isoformat()
                data[key] = secret_dict
            
            with open(secrets_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save local secrets: {e}")
    
    async def store_secret(
        self, 
        key: str, 
        value: str, 
        metadata: Dict[str, Any] = None,
        ttl_hours: int = None
    ):
        """Store secret with optional TTL"""
        expires_at = None
        if ttl_hours:
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
        
        secret = Secret(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        # Also store in database for persistence
        parts = key.split('/', 2)
        if len(parts) >= 2:
            service_name = parts[0]
            credential_type = parts[1]
            await self.credentials_manager.store_credential(
                service_name, credential_type, value, 
                metadata=metadata, expires_at=expires_at
            )
        
        if self.use_local:
            self.local_secrets[key] = secret
            self._save_local_vault()
            logger.info(f"Stored secret {key} locally")
        else:
            await self._store_vault_secret(secret)
            logger.info(f"Stored secret {key} in Vault")
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Retrieve secret value"""
        # Try vault first
        if self.use_local:
            secret = self.local_secrets.get(key)
            if secret:
                # Check expiration
                if secret.expires_at and secret.expires_at <= datetime.utcnow():
                    del self.local_secrets[key]
                    self._save_local_vault()
                    return None
                return secret.value
        else:
            value = await self._get_vault_secret(key)
            if value:
                return value
        
        # Fallback to database credentials
        parts = key.split('/', 2)
        if len(parts) >= 2:
            service_name = parts[0]
            credential_type = parts[1]
            return await self.credentials_manager.get_credential(service_name, credential_type)
            
        return None
    
    async def delete_secret(self, key: str):
        """Delete secret"""
        if self.use_local:
            if key in self.local_secrets:
                del self.local_secrets[key]
                self._save_local_vault()
                logger.info(f"Deleted secret {key} locally")
        else:
            await self._delete_vault_secret(key)
            logger.info(f"Deleted secret {key} from Vault")
    
    async def list_secrets(self, prefix: str = None) -> List[str]:
        """List secret keys"""
        if self.use_local:
            keys = list(self.local_secrets.keys())
            if prefix:
                keys = [k for k in keys if k.startswith(prefix)]
            return keys
        else:
            return await self._list_vault_secrets(prefix)
    
    async def rotate_secret(self, key: str, new_value: str = None) -> str:
        """Rotate secret with new value"""
        if not new_value:
            # Generate secure random value
            new_value = secrets.token_urlsafe(32)
        
        # Get existing metadata
        if self.use_local:
            existing = self.local_secrets.get(key)
            metadata = existing.metadata if existing else {}
        else:
            # Get metadata from Vault
            metadata = await self._get_vault_secret_metadata(key)
        
        # Store new version
        await self.store_secret(key, new_value, metadata)
        
        logger.info(f"Rotated secret {key}")
        return new_value
    
    async def _store_vault_secret(self, secret: Secret):
        """Store secret in HashiCorp Vault"""
        try:
            headers = {
                "X-Vault-Token": self.config.VAULT_TOKEN,
                "Content-Type": "application/json"
            }
            
            if self.config.VAULT_NAMESPACE:
                headers["X-Vault-Namespace"] = self.config.VAULT_NAMESPACE
            
            secret_path = f"{self.config.SECRET_PREFIX}/{secret.key}"
            url = f"{self.config.VAULT_URL}/v1/{self.config.KV_MOUNT_PATH}/data/{secret_path}"
            
            data = {
                "data": {
                    "value": secret.value,
                    "created_at": secret.created_at.isoformat(),
                    "metadata": secret.metadata
                }
            }
            
            if secret.expires_at:
                data["data"]["expires_at"] = secret.expires_at.isoformat()
            
            async with aiohttp.ClientSession() as session:
                async with session.put(url, headers=headers, json=data) as response:
                    if response.status not in [200, 204]:
                        raise VaultError(f"Failed to store secret: {await response.text()}")
                        
        except Exception as e:
            logger.error(f"Vault store error: {e}")
            raise VaultError(f"Failed to store secret in Vault: {e}")
    
    async def _get_vault_secret(self, key: str) -> Optional[str]:
        """Get secret from HashiCorp Vault"""
        try:
            headers = {
                "X-Vault-Token": self.config.VAULT_TOKEN
            }
            
            if self.config.VAULT_NAMESPACE:
                headers["X-Vault-Namespace"] = self.config.VAULT_NAMESPACE
            
            secret_path = f"{self.config.SECRET_PREFIX}/{key}"
            url = f"{self.config.VAULT_URL}/v1/{self.config.KV_MOUNT_PATH}/data/{secret_path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 404:
                        return None
                    elif response.status != 200:
                        raise VaultError(f"Failed to get secret: {await response.text()}")
                    
                    data = await response.json()
                    secret_data = data["data"]["data"]
                    
                    # Check expiration
                    if "expires_at" in secret_data:
                        expires_at = datetime.fromisoformat(secret_data["expires_at"])
                        if expires_at <= datetime.utcnow():
                            return None
                    
                    return secret_data["value"]
                    
        except Exception as e:
            logger.error(f"Vault get error: {e}")
            raise VaultError(f"Failed to get secret from Vault: {e}")
    
    async def _delete_vault_secret(self, key: str):
        """Delete secret from HashiCorp Vault"""
        try:
            headers = {
                "X-Vault-Token": self.config.VAULT_TOKEN
            }
            
            if self.config.VAULT_NAMESPACE:
                headers["X-Vault-Namespace"] = self.config.VAULT_NAMESPACE
            
            secret_path = f"{self.config.SECRET_PREFIX}/{key}"
            url = f"{self.config.VAULT_URL}/v1/{self.config.KV_MOUNT_PATH}/metadata/{secret_path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers) as response:
                    if response.status not in [200, 204, 404]:
                        raise VaultError(f"Failed to delete secret: {await response.text()}")
                        
        except Exception as e:
            logger.error(f"Vault delete error: {e}")
            raise VaultError(f"Failed to delete secret from Vault: {e}")
    
    async def _list_vault_secrets(self, prefix: str = None) -> List[str]:
        """List secrets from HashiCorp Vault"""
        try:
            headers = {
                "X-Vault-Token": self.config.VAULT_TOKEN
            }
            
            if self.config.VAULT_NAMESPACE:
                headers["X-Vault-Namespace"] = self.config.VAULT_NAMESPACE
            
            list_path = f"{self.config.SECRET_PREFIX}"
            if prefix:
                list_path = f"{list_path}/{prefix}"
            
            url = f"{self.config.VAULT_URL}/v1/{self.config.KV_MOUNT_PATH}/metadata/{list_path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.request("LIST", url, headers=headers) as response:
                    if response.status == 404:
                        return []
                    elif response.status != 200:
                        raise VaultError(f"Failed to list secrets: {await response.text()}")
                    
                    data = await response.json()
                    return data.get("data", {}).get("keys", [])
                    
        except Exception as e:
            logger.error(f"Vault list error: {e}")
            raise VaultError(f"Failed to list secrets from Vault: {e}")
    
    async def _get_vault_secret_metadata(self, key: str) -> Dict[str, Any]:
        """Get secret metadata from HashiCorp Vault"""
        try:
            headers = {
                "X-Vault-Token": self.config.VAULT_TOKEN
            }
            
            if self.config.VAULT_NAMESPACE:
                headers["X-Vault-Namespace"] = self.config.VAULT_NAMESPACE
            
            secret_path = f"{self.config.SECRET_PREFIX}/{key}"
            url = f"{self.config.VAULT_URL}/v1/{self.config.KV_MOUNT_PATH}/data/{secret_path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 404:
                        return {}
                    elif response.status != 200:
                        logger.warning(f"Failed to get secret metadata: {await response.text()}")
                        return {}
                    
                    data = await response.json()
                    secret_data = data["data"]["data"]
                    
                    # Return metadata or empty dict
                    return secret_data.get("metadata", {})
                    
        except Exception as e:
            logger.warning(f"Vault metadata retrieval error: {e}")
            return {}  # Return empty metadata on error, don't fail the operation

class DatabaseCredentials:
    """Database credential management"""
    
    def __init__(self, secret_manager: SecretManager):
        self.secret_manager = secret_manager
    
    async def get_database_url(self, db_name: str = "main") -> str:
        """Get database connection URL"""
        db_url = await self.secret_manager.get_secret(f"database/{db_name}/url")
        if not db_url:
            # Fallback to environment variable
            db_url = os.getenv(f"DATABASE_URL_{db_name.upper()}", 
                              os.getenv("DATABASE_URL", 
                                       "postgresql://postgres:password@localhost:5432/reks"))
        return db_url
    
    async def store_database_credentials(
        self, 
        db_name: str,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str
    ):
        """Store database credentials"""
        db_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        await self.secret_manager.store_secret(f"database/{db_name}/url", db_url)
        
        # Store components separately for flexibility
        await self.secret_manager.store_secret(f"database/{db_name}/host", host)
        await self.secret_manager.store_secret(f"database/{db_name}/port", str(port))
        await self.secret_manager.store_secret(f"database/{db_name}/database", database)
        await self.secret_manager.store_secret(f"database/{db_name}/username", username)
        await self.secret_manager.store_secret(f"database/{db_name}/password", password)

class APICredentials:
    """API key and credential management"""
    
    def __init__(self, secret_manager: SecretManager):
        self.secret_manager = secret_manager
    
    async def generate_service_account(self, service_name: str) -> Dict[str, str]:
        """Generate service account credentials"""
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        api_secret = secrets.token_urlsafe(48)
        
        await self.secret_manager.store_secret(f"service/{service_name}/api_key", api_key)
        await self.secret_manager.store_secret(f"service/{service_name}/api_secret", api_secret)
        
        return {
            "api_key": api_key,
            "api_secret": api_secret,
            "service_name": service_name
        }
    
    async def get_service_credentials(self, service_name: str) -> Optional[Dict[str, str]]:
        """Get service account credentials"""
        api_key = await self.secret_manager.get_secret(f"service/{service_name}/api_key")
        api_secret = await self.secret_manager.get_secret(f"service/{service_name}/api_secret")
        
        if api_key and api_secret:
            return {
                "api_key": api_key,
                "api_secret": api_secret,
                "service_name": service_name
            }
        return None
    
    async def rotate_service_credentials(self, service_name: str) -> Dict[str, str]:
        """Rotate service account credentials"""
        return await self.generate_service_account(service_name)

# Global instances
secret_manager = SecretManager()
db_credentials = DatabaseCredentials(secret_manager)
api_credentials = APICredentials(secret_manager)

async def initialize_secrets():
    """Initialize default secrets for development"""
    # Generate default encryption keys
    master_key = secrets.token_urlsafe(32)
    await secret_manager.store_secret("encryption/master_key", master_key)
    
    # Generate JWT secret
    jwt_secret = secrets.token_urlsafe(48)
    await secret_manager.store_secret("auth/jwt_secret", jwt_secret)
    
    # Generate database credentials
    await db_credentials.store_database_credentials(
        "main",
        "localhost",
        5432,
        "reks_production",
        "reks_user",
        secrets.token_urlsafe(16)
    )
    
    # Generate service accounts
    await api_credentials.generate_service_account("agent_coordinator")
    await api_credentials.generate_service_account("workflow_engine")
    await api_credentials.generate_service_account("data_loader")
    
    logger.info("Default secrets initialized")

async def setup_production_vault():
    """Setup production Vault configuration"""
    vault_policies = {
        "reks-a2a-policy": {
            "path": {
                "secret/data/reks/a2a/*": {
                    "capabilities": ["create", "read", "update", "delete", "list"]
                },
                "secret/metadata/reks/a2a/*": {
                    "capabilities": ["list", "delete"]
                }
            }
        }
    }
    
    logger.info("Vault policies defined for production setup")
    return vault_policies