"""
API Credentials Database Management
Secure storage and retrieval of API credentials
"""

import base64
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet

from ...infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)


class CredentialsManager:
    """
    Manages API credentials with database persistence
    Provides encryption, rotation, and access control
    """

    def __init__(self, db: Optional[UnifiedDatabase] = None, encryption_key: Optional[str] = None):
        self.db = db or UnifiedDatabase()

        # Initialize encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode())
        else:
            # Get encryption key from environment variable
            env_key = os.environ.get("CRYPTOTRADING_ENCRYPTION_KEY")
            if env_key:
                try:
                    # Validate key format
                    self.cipher = Fernet(env_key.encode())
                except Exception:
                    # If invalid, generate new key
                    logger.warning("Invalid encryption key format, generating new key")
                    key = Fernet.generate_key()
                    self.cipher = Fernet(key)
                    logger.info(
                        f"Generated new encryption key. Set CRYPTOTRADING_ENCRYPTION_KEY={key.decode()}"
                    )
            else:
                # Generate new key for development
                key = Fernet.generate_key()
                self.cipher = Fernet(key)
                logger.warning(f"No encryption key found. Generated new key for development.")
                logger.warning(f"For production, set CRYPTOTRADING_ENCRYPTION_KEY={key.decode()}")

    def _encrypt(self, value: str) -> str:
        """Encrypt sensitive value"""
        return base64.b64encode(self.cipher.encrypt(value.encode())).decode()

    def _decrypt(self, encrypted_value: str) -> str:
        """Decrypt sensitive value"""
        return self.cipher.decrypt(base64.b64decode(encrypted_value.encode())).decode()

    async def store_credential(
        self,
        service_name: str,
        credential_type: str,
        credential_value: str,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> bool:
        """Store an API credential securely"""
        try:
            # Encrypt the credential value
            encrypted_value = self._encrypt(credential_value)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO api_credentials
                    (service_name, credential_type, encrypted_value, 
                     metadata, is_active, expires_at, created_at, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        service_name,
                        credential_type,
                        encrypted_value,
                        json.dumps(metadata) if metadata else None,
                        True,
                        expires_at,
                        datetime.utcnow(),
                        None,
                    ),
                )

                conn.commit()

            logger.info(f"Stored credential for {service_name}:{credential_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to store credential: {e}")
            return False

    async def get_credential(self, service_name: str, credential_type: str) -> Optional[str]:
        """Retrieve and decrypt a credential"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT encrypted_value, expires_at FROM api_credentials
                    WHERE service_name = ? AND credential_type = ? AND is_active = 1
                """,
                    (service_name, credential_type),
                )

                row = cursor.fetchone()

                if row:
                    encrypted_value, expires_at = row

                    # Check expiration
                    if expires_at and datetime.fromisoformat(expires_at) < datetime.utcnow():
                        logger.warning(f"Credential expired for {service_name}:{credential_type}")
                        return None

                    # Update last used
                    cursor.execute(
                        """
                        UPDATE api_credentials
                        SET last_used = ?, usage_count = usage_count + 1
                        WHERE service_name = ? AND credential_type = ?
                    """,
                        (datetime.utcnow(), service_name, credential_type),
                    )
                    conn.commit()

                    # Decrypt and return
                    return self._decrypt(encrypted_value)

            return None

        except Exception as e:
            logger.error(f"Failed to get credential: {e}")
            return None

    async def rotate_credential(
        self, service_name: str, credential_type: str, new_value: str, keep_old_days: int = 7
    ) -> bool:
        """Rotate a credential, keeping old one temporarily"""
        try:
            # Mark old credential as rotated
            rotation_date = datetime.utcnow()
            expiry_date = rotation_date + timedelta(days=keep_old_days)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Update old credential
                cursor.execute(
                    """
                    UPDATE api_credentials
                    SET rotated_at = ?, expires_at = ?
                    WHERE service_name = ? AND credential_type = ? AND is_active = 1
                """,
                    (rotation_date, expiry_date, service_name, credential_type),
                )

                # Store new credential
                await self.store_credential(service_name, credential_type, new_value)

                conn.commit()

            logger.info(f"Rotated credential for {service_name}:{credential_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate credential: {e}")
            return False

    async def deactivate_credential(self, service_name: str, credential_type: str) -> bool:
        """Deactivate a credential"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE api_credentials
                    SET is_active = 0, deactivated_at = ?
                    WHERE service_name = ? AND credential_type = ?
                """,
                    (datetime.utcnow(), service_name, credential_type),
                )

                conn.commit()

                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to deactivate credential: {e}")
            return False

    async def list_credentials(self, service_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List credentials (without decrypted values)"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                if service_name:
                    cursor.execute(
                        """
                        SELECT service_name, credential_type, is_active, 
                               expires_at, created_at, last_used, usage_count
                        FROM api_credentials
                        WHERE service_name = ?
                        ORDER BY created_at DESC
                    """,
                        (service_name,),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT service_name, credential_type, is_active,
                               expires_at, created_at, last_used, usage_count
                        FROM api_credentials
                        ORDER BY service_name, credential_type
                    """
                    )

                credentials = []
                for row in cursor.fetchall():
                    credentials.append(
                        {
                            "service_name": row[0],
                            "credential_type": row[1],
                            "is_active": bool(row[2]),
                            "expires_at": row[3],
                            "created_at": row[4],
                            "last_used": row[5],
                            "usage_count": row[6],
                        }
                    )

                return credentials

        except Exception as e:
            logger.error(f"Failed to list credentials: {e}")
            return []

    async def get_credential_stats(self) -> Dict[str, Any]:
        """Get credential usage statistics"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Total credentials
                cursor.execute("SELECT COUNT(*) FROM api_credentials")
                total = cursor.fetchone()[0]

                # Active credentials
                cursor.execute("SELECT COUNT(*) FROM api_credentials WHERE is_active = 1")
                active = cursor.fetchone()[0]

                # Expiring soon (within 7 days)
                expiry_date = datetime.utcnow() + timedelta(days=7)
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM api_credentials
                    WHERE is_active = 1 AND expires_at IS NOT NULL AND expires_at < ?
                """,
                    (expiry_date,),
                )
                expiring_soon = cursor.fetchone()[0]

                # Most used credentials
                cursor.execute(
                    """
                    SELECT service_name, credential_type, usage_count
                    FROM api_credentials
                    WHERE is_active = 1
                    ORDER BY usage_count DESC
                    LIMIT 10
                """
                )

                most_used = []
                for row in cursor.fetchall():
                    most_used.append(
                        {"service_name": row[0], "credential_type": row[1], "usage_count": row[2]}
                    )

                return {
                    "total_credentials": total,
                    "active_credentials": active,
                    "inactive_credentials": total - active,
                    "expiring_soon": expiring_soon,
                    "most_used": most_used,
                }

        except Exception as e:
            logger.error(f"Failed to get credential stats: {e}")
            return {}

    async def cleanup_expired(self) -> int:
        """Remove expired credentials"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    DELETE FROM api_credentials
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                """,
                    (datetime.utcnow(),),
                )

                deleted = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted} expired credentials")
                return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup expired credentials: {e}")
            return 0


# Integration with existing vault
class VaultIntegration:
    """Integrates credentials manager with existing vault"""

    def __init__(self, vault, credentials_manager: Optional[CredentialsManager] = None):
        self.vault = vault
        self.credentials = credentials_manager or CredentialsManager()

    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret from vault with database fallback"""
        # Try vault first
        value = await self.vault.get_secret(key)
        if value:
            return value

        # Try credentials manager
        parts = key.split(":", 1)
        if len(parts) == 2:
            service_name, credential_type = parts
            return await self.credentials.get_credential(service_name, credential_type)

        return None

    async def set_secret(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set secret in both vault and database"""
        # Store in vault
        await self.vault.set_secret(key, value)

        # Store in credentials manager
        parts = key.split(":", 1)
        if len(parts) == 2:
            service_name, credential_type = parts
            expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl else None
            return await self.credentials.store_credential(
                service_name, credential_type, value, expires_at=expires_at
            )

        return True


# Global credentials manager instance
_credentials_manager: Optional[CredentialsManager] = None


async def get_credentials_manager() -> CredentialsManager:
    """Get global credentials manager instance"""
    global _credentials_manager
    if _credentials_manager is None:
        _credentials_manager = CredentialsManager()
    return _credentials_manager
