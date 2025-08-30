"""
Data Encryption Service for On-Chain Storage
Provides encryption/decryption for sensitive data stored on blockchain
"""

import base64
import hashlib
import json
import logging
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from eth_keys import keys
from eth_utils import keccak

logger = logging.getLogger(__name__)


@dataclass
class EncryptionKey:
    """Encryption key metadata"""
    key_id: str
    key_type: str  # "symmetric" or "asymmetric"
    agent_id: str
    created_at: int
    expires_at: Optional[int] = None
    public_key: Optional[bytes] = None
    encrypted_private_key: Optional[bytes] = None


class DataEncryptionService:
    """Service for encrypting sensitive data before on-chain storage"""
    
    # Encryption modes
    MODE_SYMMETRIC = "symmetric"
    MODE_ASYMMETRIC = "asymmetric"
    MODE_HYBRID = "hybrid"  # Asymmetric for key exchange, symmetric for data
    
    def __init__(self):
        """Initialize encryption service"""
        self.backend = default_backend()
        self._key_store: Dict[str, EncryptionKey] = {}
        self._agent_keys: Dict[str, Dict[str, Any]] = {}
        
        logger.info("DataEncryptionService initialized")
    
    def generate_symmetric_key(self) -> bytes:
        """Generate a new symmetric encryption key"""
        return Fernet.generate_key()
    
    def generate_asymmetric_keypair(self) -> Tuple[bytes, bytes]:
        """Generate RSA keypair for asymmetric encryption"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=self.backend
        )
        
        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Serialize public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicKeyFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def derive_key_from_address(self, ethereum_address: str, salt: bytes = None) -> bytes:
        """Derive encryption key from Ethereum address"""
        if salt is None:
            salt = b"a2a_data_encryption"
        
        # Use PBKDF2 to derive key from address
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        
        key = kdf.derive(ethereum_address.encode())
        return base64.urlsafe_b64encode(key)
    
    def encrypt_symmetric(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using symmetric encryption (Fernet)"""
        try:
            f = Fernet(key)
            encrypted = f.encrypt(data)
            return encrypted
        except Exception as e:
            logger.error(f"Symmetric encryption failed: {e}")
            raise
    
    def decrypt_symmetric(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using symmetric encryption"""
        try:
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_data)
            return decrypted
        except Exception as e:
            logger.error(f"Symmetric decryption failed: {e}")
            raise
    
    def encrypt_asymmetric(self, data: bytes, public_key_pem: bytes) -> bytes:
        """Encrypt data using RSA public key"""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem,
                backend=self.backend
            )
            
            # RSA can only encrypt limited data, so we chunk if needed
            max_chunk_size = 190  # For 2048-bit RSA key
            
            if len(data) <= max_chunk_size:
                encrypted = public_key.encrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return encrypted
            else:
                # For larger data, use hybrid encryption
                return self._hybrid_encrypt(data, public_key_pem)
                
        except Exception as e:
            logger.error(f"Asymmetric encryption failed: {e}")
            raise
    
    def decrypt_asymmetric(self, encrypted_data: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt data using RSA private key"""
        try:
            private_key = serialization.load_pem_private_key(
                private_key_pem,
                password=None,
                backend=self.backend
            )
            
            # Check if this is hybrid encrypted
            if len(encrypted_data) > 256:  # Likely hybrid encrypted
                return self._hybrid_decrypt(encrypted_data, private_key_pem)
            
            decrypted = private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted
            
        except Exception as e:
            logger.error(f"Asymmetric decryption failed: {e}")
            raise
    
    def _hybrid_encrypt(self, data: bytes, public_key_pem: bytes) -> bytes:
        """Hybrid encryption: RSA for key, AES for data"""
        # Generate symmetric key
        sym_key = self.generate_symmetric_key()
        
        # Encrypt the symmetric key with RSA
        encrypted_key = self.encrypt_asymmetric(sym_key, public_key_pem)
        
        # Encrypt the data with symmetric key
        encrypted_data = self.encrypt_symmetric(data, sym_key)
        
        # Combine: [key_length(2 bytes)][encrypted_key][encrypted_data]
        key_length = len(encrypted_key).to_bytes(2, 'big')
        return key_length + encrypted_key + encrypted_data
    
    def _hybrid_decrypt(self, encrypted_blob: bytes, private_key_pem: bytes) -> bytes:
        """Hybrid decryption: Extract and decrypt key, then decrypt data"""
        # Extract key length
        key_length = int.from_bytes(encrypted_blob[:2], 'big')
        
        # Extract encrypted key and data
        encrypted_key = encrypted_blob[2:2+key_length]
        encrypted_data = encrypted_blob[2+key_length:]
        
        # Decrypt the symmetric key
        sym_key = self.decrypt_asymmetric(encrypted_key, private_key_pem)
        
        # Decrypt the data
        return self.decrypt_symmetric(encrypted_data, sym_key)
    
    def encrypt_for_agents(
        self,
        data: Any,
        sender_agent_id: str,
        receiver_agent_ids: List[str],
        mode: str = MODE_HYBRID
    ) -> Dict[str, bytes]:
        """
        Encrypt data for multiple recipient agents
        
        Args:
            data: Data to encrypt (will be serialized to JSON)
            sender_agent_id: ID of sending agent
            receiver_agent_ids: List of recipient agent IDs
            mode: Encryption mode
        
        Returns:
            Dictionary mapping agent_id to encrypted data
        """
        # Serialize data
        if isinstance(data, (dict, list)):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode('utf-8')
        
        encrypted_data = {}
        
        if mode == self.MODE_SYMMETRIC:
            # Use shared symmetric key (derived from sender)
            key = self.derive_key_from_address(sender_agent_id)
            encrypted = self.encrypt_symmetric(data_bytes, key)
            
            # Same encrypted data for all recipients
            for agent_id in receiver_agent_ids:
                encrypted_data[agent_id] = encrypted
                
        elif mode in [self.MODE_ASYMMETRIC, self.MODE_HYBRID]:
            # Encrypt separately for each recipient
            for agent_id in receiver_agent_ids:
                # Get or generate public key for agent
                public_key = self._get_agent_public_key(agent_id)
                
                if public_key:
                    encrypted = self.encrypt_asymmetric(data_bytes, public_key)
                    encrypted_data[agent_id] = encrypted
                else:
                    # Fallback to symmetric encryption
                    key = self.derive_key_from_address(agent_id)
                    encrypted = self.encrypt_symmetric(data_bytes, key)
                    encrypted_data[agent_id] = encrypted
        
        return encrypted_data
    
    def decrypt_agent_data(
        self,
        encrypted_data: bytes,
        agent_id: str,
        sender_agent_id: Optional[str] = None,
        mode: str = MODE_HYBRID
    ) -> Any:
        """
        Decrypt data for an agent
        
        Args:
            encrypted_data: Encrypted data bytes
            agent_id: ID of decrypting agent
            sender_agent_id: ID of sender (for symmetric mode)
            mode: Encryption mode used
        
        Returns:
            Decrypted data
        """
        try:
            if mode == self.MODE_SYMMETRIC:
                # Use shared key
                if sender_agent_id:
                    key = self.derive_key_from_address(sender_agent_id)
                else:
                    key = self.derive_key_from_address(agent_id)
                
                decrypted_bytes = self.decrypt_symmetric(encrypted_data, key)
                
            elif mode in [self.MODE_ASYMMETRIC, self.MODE_HYBRID]:
                # Get agent's private key
                private_key = self._get_agent_private_key(agent_id)
                
                if private_key:
                    decrypted_bytes = self.decrypt_asymmetric(encrypted_data, private_key)
                else:
                    # Fallback to symmetric
                    key = self.derive_key_from_address(agent_id)
                    decrypted_bytes = self.decrypt_symmetric(encrypted_data, key)
            
            else:
                raise ValueError(f"Unknown encryption mode: {mode}")
            
            # Try to deserialize as JSON
            try:
                return json.loads(decrypted_bytes.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Return as string or bytes
                try:
                    return decrypted_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    return decrypted_bytes
                    
        except Exception as e:
            logger.error(f"Failed to decrypt agent data: {e}")
            raise
    
    def register_agent_keys(
        self,
        agent_id: str,
        public_key: Optional[bytes] = None,
        private_key: Optional[bytes] = None
    ) -> bool:
        """Register encryption keys for an agent"""
        try:
            if not public_key:
                # Generate new keypair
                private_key, public_key = self.generate_asymmetric_keypair()
            
            self._agent_keys[agent_id] = {
                "public_key": public_key,
                "private_key": private_key,
                "symmetric_key": self.derive_key_from_address(agent_id)
            }
            
            logger.info(f"Registered encryption keys for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent keys: {e}")
            return False
    
    def _get_agent_public_key(self, agent_id: str) -> Optional[bytes]:
        """Get public key for an agent"""
        if agent_id in self._agent_keys:
            return self._agent_keys[agent_id].get("public_key")
        
        # Try to fetch from registry or generate
        # For now, return None to use symmetric fallback
        return None
    
    def _get_agent_private_key(self, agent_id: str) -> Optional[bytes]:
        """Get private key for an agent"""
        if agent_id in self._agent_keys:
            return self._agent_keys[agent_id].get("private_key")
        return None
    
    def create_data_hash(self, data: bytes) -> str:
        """Create hash of data for integrity verification"""
        return hashlib.sha256(data).hexdigest()
    
    def verify_data_integrity(self, data: bytes, expected_hash: str) -> bool:
        """Verify data integrity using hash"""
        actual_hash = self.create_data_hash(data)
        return actual_hash == expected_hash


# Singleton instance
_encryption_service: Optional[DataEncryptionService] = None


def get_encryption_service() -> DataEncryptionService:
    """Get or create the encryption service singleton"""
    global _encryption_service
    
    if _encryption_service is None:
        _encryption_service = DataEncryptionService()
    
    return _encryption_service


# Convenience functions
def encrypt_for_chain(
    data: Any,
    sender: str,
    receivers: List[str]
) -> Dict[str, bytes]:
    """Encrypt data for on-chain storage"""
    service = get_encryption_service()
    return service.encrypt_for_agents(data, sender, receivers)


def decrypt_from_chain(
    encrypted_data: bytes,
    agent_id: str,
    sender_id: Optional[str] = None
) -> Any:
    """Decrypt data from on-chain storage"""
    service = get_encryption_service()
    return service.decrypt_agent_data(encrypted_data, agent_id, sender_id)