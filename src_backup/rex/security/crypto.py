"""
Production-grade encryption for message payloads and sensitive data
Implements AES-256-GCM encryption with key rotation and secure key derivation
"""

import os
import base64
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import json

logger = logging.getLogger(__name__)

@dataclass
class EncryptionKey:
    """Encryption key with metadata"""
    key_id: str
    key_data: bytes
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True

class CryptoConfig:
    """Encryption configuration"""
    AES_KEY_SIZE = 32  # 256 bits
    GCM_NONCE_SIZE = 12  # 96 bits
    RSA_KEY_SIZE = 2048
    PBKDF2_ITERATIONS = 100000
    KEY_ROTATION_DAYS = 90
    MASTER_KEY_ENV = "REKS_MASTER_KEY"

class EncryptionError(Exception):
    """Encryption operation failed"""
    pass

class DecryptionError(Exception):
    """Decryption operation failed"""
    pass

class KeyManager:
    """Production key management with rotation and secure storage"""
    
    def __init__(self, config: CryptoConfig = None):
        self.config = config or CryptoConfig()
        self.keys: Dict[str, EncryptionKey] = {}
        self.master_key = self._get_or_create_master_key()
        self.current_key_id = None
        
        # Initialize with current encryption key
        self._load_keys_from_storage()
        self._ensure_current_key()
    
    def _get_or_create_master_key(self) -> bytes:
        """Get master key from environment or create new one"""
        master_key_b64 = os.getenv(self.config.MASTER_KEY_ENV)
        
        if master_key_b64:
            try:
                return base64.b64decode(master_key_b64)
            except Exception as e:
                logger.error(f"Failed to decode master key: {e}")
                raise EncryptionError("Invalid master key format")
        else:
            # Generate new master key
            master_key = secrets.token_bytes(self.config.AES_KEY_SIZE)
            master_key_b64 = base64.b64encode(master_key).decode()
            
            logger.warning(f"Generated new master key. Set environment variable: "
                         f"{self.config.MASTER_KEY_ENV}={master_key_b64}")
            
            return master_key
    
    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key from master key using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.config.AES_KEY_SIZE,
            salt=salt,
            iterations=self.config.PBKDF2_ITERATIONS,
            backend=default_backend()
        )
        return kdf.derive(self.master_key)
    
    def _ensure_current_key(self):
        """Ensure we have a current valid encryption key"""
        # Check if current key exists and is valid
        if self.current_key_id:
            current_key = self.keys.get(self.current_key_id)
            if current_key and current_key.is_active:
                if not current_key.expires_at or current_key.expires_at > datetime.utcnow():
                    return
        
        # Create new key
        self.current_key_id = self.create_key()
    
    def create_key(self, expires_in_days: int = None) -> str:
        """Create new encryption key"""
        key_id = f"key-{secrets.token_urlsafe(8)}"
        salt = secrets.token_bytes(16)
        key_data = self._derive_key(salt)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        else:
            expires_at = datetime.utcnow() + timedelta(days=self.config.KEY_ROTATION_DAYS)
        
        encryption_key = EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            algorithm="AES-256-GCM",
            created_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
        self.keys[key_id] = encryption_key
        
        # Store salt for key derivation (in production, use secure key store)
        self._store_key_metadata(key_id, salt)
        
        logger.info(f"Created encryption key {key_id}")
        return key_id
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get encryption key by ID"""
        return self.keys.get(key_id)
    
    def get_current_key(self) -> EncryptionKey:
        """Get current active encryption key"""
        if not self.current_key_id:
            self._ensure_current_key()
        
        return self.keys[self.current_key_id]
    
    def rotate_keys(self):
        """Rotate encryption keys"""
        # Deactivate old keys
        for key in self.keys.values():
            if key.expires_at and key.expires_at <= datetime.utcnow():
                key.is_active = False
        
        # Create new current key
        self.current_key_id = self.create_key()
        
        logger.info("Encryption keys rotated")
    
    def _store_key_metadata(self, key_id: str, salt: bytes):
        """Store key metadata (salt) for derivation in database"""
        from ..database.models import EncryptionKeyMetadata
        from ..database.client import get_db
        
        try:
            db = get_db()
            with db.get_session() as session:
                # Check if key metadata already exists
                existing = session.query(EncryptionKeyMetadata).filter_by(key_id=key_id).first()
                
                if not existing:
                    # Store new key metadata
                    key_metadata = EncryptionKeyMetadata(
                        key_id=key_id,
                        salt=base64.b64encode(salt).decode(),
                        algorithm="AES-256-GCM",
                        created_at=datetime.utcnow()
                    )
                    session.add(key_metadata)
                    session.commit()
                    logger.info(f"Stored key metadata for {key_id}")
                
        except Exception as e:
            logger.error(f"Failed to store key metadata for {key_id}: {e}")
            raise EncryptionError(f"Failed to store key metadata: {e}")
    
    def _retrieve_key_metadata(self, key_id: str) -> Optional[bytes]:
        """Retrieve key metadata (salt) from database"""
        from ..database.models import EncryptionKeyMetadata
        from ..database.client import get_db
        
        try:
            db = get_db()
            with db.get_session() as session:
                key_metadata = session.query(EncryptionKeyMetadata).filter_by(key_id=key_id).first()
                
                if key_metadata:
                    return base64.b64decode(key_metadata.salt)
                else:
                    logger.warning(f"Key metadata not found for {key_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to retrieve key metadata for {key_id}: {e}")
            return None
    
    def _load_keys_from_storage(self):
        """Load existing keys from persistent storage"""
        from ..database.models import EncryptionKeyMetadata
        from ..database.client import get_db
        
        try:
            db = get_db()
            with db.get_session() as session:
                # Get all active key metadata
                key_metadata_list = session.query(EncryptionKeyMetadata).filter_by(is_active=True).all()
                
                for key_metadata in key_metadata_list:
                    try:
                        # Recreate the encryption key
                        salt = base64.b64decode(key_metadata.salt)
                        key_data = self._derive_key(salt)
                        
                        encryption_key = EncryptionKey(
                            key_id=key_metadata.key_id,
                            key_data=key_data,
                            algorithm=key_metadata.algorithm,
                            created_at=key_metadata.created_at,
                            expires_at=key_metadata.expires_at,
                            is_active=key_metadata.is_active
                        )
                        
                        self.keys[key_metadata.key_id] = encryption_key
                        
                        # Set as current if it's the most recent active key
                        if not self.current_key_id or key_metadata.created_at > self.keys[self.current_key_id].created_at:
                            self.current_key_id = key_metadata.key_id
                        
                        logger.debug(f"Loaded encryption key {key_metadata.key_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to load key {key_metadata.key_id}: {e}")
                        continue
                
                logger.info(f"Loaded {len(self.keys)} encryption keys from storage")
                
        except Exception as e:
            logger.warning(f"Failed to load keys from storage: {e}. Will create new keys as needed.")

class MessageEncryption:
    """Production message encryption with authenticated encryption"""
    
    def __init__(self, key_manager: KeyManager = None):
        self.key_manager = key_manager or KeyManager()
    
    def encrypt_payload(self, payload: Dict[str, Any], additional_data: bytes = None) -> Dict[str, str]:
        """Encrypt message payload with authenticated encryption"""
        try:
            # Serialize payload
            payload_json = json.dumps(payload, sort_keys=True)
            payload_bytes = payload_json.encode('utf-8')
            
            # Get current key
            key = self.key_manager.get_current_key()
            
            # Generate nonce
            nonce = secrets.token_bytes(self.key_manager.config.GCM_NONCE_SIZE)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key.key_data),
                modes.GCM(nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Add additional authenticated data if provided
            if additional_data:
                encryptor.authenticate_additional_data(additional_data)
            
            # Encrypt
            ciphertext = encryptor.update(payload_bytes) + encryptor.finalize()
            
            # Create encrypted message
            encrypted_payload = {
                "key_id": key.key_id,
                "algorithm": key.algorithm,
                "nonce": base64.b64encode(nonce).decode(),
                "ciphertext": base64.b64encode(ciphertext).decode(),
                "tag": base64.b64encode(encryptor.tag).decode(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if additional_data:
                encrypted_payload["aad_hash"] = hashlib.sha256(additional_data).hexdigest()
            
            logger.debug(f"Encrypted payload with key {key.key_id}")
            return encrypted_payload
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt payload: {e}")
    
    def decrypt_payload(self, encrypted_payload: Dict[str, str], additional_data: bytes = None) -> Dict[str, Any]:
        """Decrypt message payload with authentication verification"""
        try:
            # Extract components
            key_id = encrypted_payload["key_id"]
            nonce = base64.b64decode(encrypted_payload["nonce"])
            ciphertext = base64.b64decode(encrypted_payload["ciphertext"])
            tag = base64.b64decode(encrypted_payload["tag"])
            
            # Verify additional data hash if provided
            if additional_data:
                expected_hash = encrypted_payload.get("aad_hash")
                if expected_hash:
                    actual_hash = hashlib.sha256(additional_data).hexdigest()
                    if actual_hash != expected_hash:
                        raise DecryptionError("Additional data authentication failed")
            
            # Get decryption key
            key = self.key_manager.get_key(key_id)
            if not key:
                raise DecryptionError(f"Unknown key ID: {key_id}")
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key.key_data),
                modes.GCM(nonce, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Add additional authenticated data if provided
            if additional_data:
                decryptor.authenticate_additional_data(additional_data)
            
            # Decrypt
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Parse JSON
            payload_json = plaintext.decode('utf-8')
            payload = json.loads(payload_json)
            
            logger.debug(f"Decrypted payload with key {key_id}")
            return payload
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise DecryptionError(f"Failed to decrypt payload: {e}")
    
    def encrypt_field(self, value: str, field_name: str = None) -> str:
        """Encrypt individual field value"""
        try:
            value_bytes = value.encode('utf-8')
            key = self.key_manager.get_current_key()
            nonce = secrets.token_bytes(self.key_manager.config.GCM_NONCE_SIZE)
            
            # Use field name as additional data for binding
            additional_data = field_name.encode('utf-8') if field_name else None
            
            cipher = Cipher(
                algorithms.AES(key.key_data),
                modes.GCM(nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            if additional_data:
                encryptor.authenticate_additional_data(additional_data)
            
            ciphertext = encryptor.update(value_bytes) + encryptor.finalize()
            
            # Encode as base64 with metadata
            encrypted_data = {
                "k": key.key_id,
                "n": base64.b64encode(nonce).decode(),
                "c": base64.b64encode(ciphertext).decode(),
                "t": base64.b64encode(encryptor.tag).decode()
            }
            
            return base64.b64encode(json.dumps(encrypted_data).encode()).decode()
            
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt field: {e}")
    
    def decrypt_field(self, encrypted_value: str, field_name: str = None) -> str:
        """Decrypt individual field value"""
        try:
            # Decode and parse
            encrypted_data = json.loads(base64.b64decode(encrypted_value).decode())
            
            key_id = encrypted_data["k"]
            nonce = base64.b64decode(encrypted_data["n"])
            ciphertext = base64.b64decode(encrypted_data["c"])
            tag = base64.b64decode(encrypted_data["t"])
            
            key = self.key_manager.get_key(key_id)
            if not key:
                raise DecryptionError(f"Unknown key ID: {key_id}")
            
            additional_data = field_name.encode('utf-8') if field_name else None
            
            cipher = Cipher(
                algorithms.AES(key.key_data),
                modes.GCM(nonce, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            if additional_data:
                decryptor.authenticate_additional_data(additional_data)
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext.decode('utf-8')
            
        except Exception as e:
            raise DecryptionError(f"Failed to decrypt field: {e}")

# Global instances
key_manager = KeyManager()
message_encryption = MessageEncryption(key_manager)

def encrypt_sensitive_data(data: Dict[str, Any], sensitive_fields: list = None) -> Dict[str, Any]:
    """Encrypt sensitive fields in a data dictionary"""
    if not sensitive_fields:
        # Default sensitive fields
        sensitive_fields = ['password', 'api_key', 'private_key', 'secret', 'token']
    
    encrypted_data = data.copy()
    
    for field in sensitive_fields:
        if field in encrypted_data and isinstance(encrypted_data[field], str):
            encrypted_data[field] = message_encryption.encrypt_field(
                encrypted_data[field], 
                field
            )
    
    return encrypted_data

def decrypt_sensitive_data(encrypted_data: Dict[str, Any], sensitive_fields: list = None) -> Dict[str, Any]:
    """Decrypt sensitive fields in a data dictionary"""
    if not sensitive_fields:
        sensitive_fields = ['password', 'api_key', 'private_key', 'secret', 'token']
    
    decrypted_data = encrypted_data.copy()
    
    for field in sensitive_fields:
        if field in decrypted_data and isinstance(decrypted_data[field], str):
            try:
                decrypted_data[field] = message_encryption.decrypt_field(
                    decrypted_data[field],
                    field
                )
            except DecryptionError:
                # Field might not be encrypted
                pass
    
    return decrypted_data