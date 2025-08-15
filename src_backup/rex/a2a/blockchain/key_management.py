"""
Production Key Management for Blockchain A2A System
Secure handling of private keys and addresses
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from eth_account import Account
from cryptography.fernet import Fernet
import secrets

logger = logging.getLogger(__name__)

class SecureKeyManager:
    """Secure key management for blockchain agents"""
    
    def __init__(self, keystore_path: Optional[Path] = None):
        self.keystore_path = keystore_path or Path(__file__).parent / "keystore"
        self.keystore_path.mkdir(mode=0o700, exist_ok=True)  # Secure permissions
        
        # Initialize encryption key
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Agent key cache (in memory only)
        self._agent_keys: Dict[str, Dict[str, str]] = {}
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for keystore"""
        key_file = self.keystore_path / ".encryption_key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new encryption key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set secure permissions
            os.chmod(key_file, 0o600)
            logger.info("Generated new encryption key for keystore")
            return key
    
    def generate_agent_keys(self, agent_id: str, password: Optional[str] = None) -> Tuple[str, str]:
        """Generate new keys for an agent"""
        # Generate random private key
        account = Account.create()
        
        private_key = account.key.hex()
        address = account.address
        
        # Store securely
        self.store_agent_keys(agent_id, private_key, password)
        
        logger.info(f"Generated new keys for agent {agent_id}: {address}")
        return private_key, address
    
    def store_agent_keys(self, agent_id: str, private_key: str, password: Optional[str] = None):
        """Store agent keys securely"""
        # Validate private key
        try:
            account = Account.from_key(private_key)
            address = account.address
        except Exception as e:
            raise ValueError(f"Invalid private key: {e}")
        
        # Encrypt private key
        encrypted_key = self.cipher.encrypt(private_key.encode())
        
        # Create keystore entry
        keystore_entry = {
            "agent_id": agent_id,
            "address": address,
            "encrypted_private_key": encrypted_key.decode(),
            "created_at": str(datetime.now()),
            "key_derivation": "direct",  # Could implement HD wallets later
        }
        
        # Add password protection if provided
        if password:
            # Additional layer of password-based encryption
            password_cipher = Fernet(self._derive_key_from_password(password))
            keystore_entry["encrypted_private_key"] = password_cipher.encrypt(
                keystore_entry["encrypted_private_key"].encode()
            ).decode()
            keystore_entry["password_protected"] = True
        else:
            keystore_entry["password_protected"] = False
        
        # Save to file
        keystore_file = self.keystore_path / f"{agent_id}.json"
        with open(keystore_file, 'w') as f:
            json.dump(keystore_entry, f, indent=2)
        
        # Set secure permissions
        os.chmod(keystore_file, 0o600)
        
        # Cache in memory
        self._agent_keys[agent_id] = {
            "private_key": private_key,
            "address": address
        }
        
        logger.info(f"Stored keys for agent {agent_id} at {address}")
    
    def load_agent_keys(self, agent_id: str, password: Optional[str] = None) -> Tuple[str, str]:
        """Load agent keys from secure storage"""
        # Check cache first
        if agent_id in self._agent_keys:
            cached = self._agent_keys[agent_id]
            return cached["private_key"], cached["address"]
        
        # Load from keystore
        keystore_file = self.keystore_path / f"{agent_id}.json"
        
        if not keystore_file.exists():
            raise FileNotFoundError(f"No keystore found for agent {agent_id}")
        
        with open(keystore_file, 'r') as f:
            keystore_entry = json.load(f)
        
        # Decrypt private key
        encrypted_key = keystore_entry["encrypted_private_key"]
        
        if keystore_entry.get("password_protected", False):
            if not password:
                raise ValueError(f"Password required for agent {agent_id}")
            
            # First decrypt with password
            password_cipher = Fernet(self._derive_key_from_password(password))
            encrypted_key = password_cipher.decrypt(encrypted_key.encode()).decode()
        
        # Decrypt with master key
        private_key = self.cipher.decrypt(encrypted_key.encode()).decode()
        address = keystore_entry["address"]
        
        # Verify key integrity
        try:
            account = Account.from_key(private_key)
            if account.address != address:
                raise ValueError("Key integrity check failed")
        except Exception as e:
            raise ValueError(f"Failed to verify key integrity: {e}")
        
        # Cache in memory
        self._agent_keys[agent_id] = {
            "private_key": private_key,
            "address": address
        }
        
        logger.info(f"Loaded keys for agent {agent_id}")
        return private_key, address
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive encryption key from password"""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        import base64
        
        # Use a static salt for consistency (in production, use per-agent salts)
        salt = b"a2a_blockchain_system_salt_v1"
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def list_agents(self) -> Dict[str, str]:
        """List all agents and their addresses"""
        agents = {}
        
        for keystore_file in self.keystore_path.glob("*.json"):
            if keystore_file.name.startswith("."):
                continue
                
            try:
                with open(keystore_file, 'r') as f:
                    entry = json.load(f)
                
                agent_id = entry["agent_id"]
                address = entry["address"]
                agents[agent_id] = address
                
            except Exception as e:
                logger.warning(f"Failed to read keystore {keystore_file}: {e}")
        
        return agents
    
    def get_default_anvil_keys(self) -> Dict[str, Tuple[str, str]]:
        """Get default Anvil test keys for development"""
        anvil_accounts = {
            "deployer": (
                "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
            ),
            "historical-loader-001": (
                "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
                "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
            ),
            "database-001": (
                "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
                "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
            )
        }
        
        logger.warning("Using default Anvil keys - FOR DEVELOPMENT ONLY")
        return anvil_accounts
    
    def clear_cache(self):
        """Clear in-memory key cache"""
        self._agent_keys.clear()
        logger.info("Cleared key cache")


# Global key manager instance
key_manager = SecureKeyManager()

def get_agent_keys(agent_id: str, password: Optional[str] = None, use_anvil_defaults: bool = False) -> Tuple[str, str]:
    """Get agent keys with fallback to Anvil defaults for development"""
    try:
        return key_manager.load_agent_keys(agent_id, password)
    except FileNotFoundError:
        if use_anvil_defaults:
            anvil_keys = key_manager.get_default_anvil_keys()
            if agent_id in anvil_keys:
                private_key, address = anvil_keys[agent_id]
                logger.info(f"Using Anvil default keys for {agent_id} (development mode)")
                return private_key, address
        
        # Generate new keys if not found
        logger.info(f"Generating new keys for agent {agent_id}")
        return key_manager.generate_agent_keys(agent_id, password)

def create_agent_keys(agent_id: str, password: Optional[str] = None) -> Tuple[str, str]:
    """Create new keys for an agent"""
    return key_manager.generate_agent_keys(agent_id, password)