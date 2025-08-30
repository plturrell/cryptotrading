"""
Security Interface Definitions
Abstract interfaces for security components to prevent circular dependencies
"""
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union


class SecurityLevel(Enum):
    """Security levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IAuthenticator(ABC):
    """Authentication interface"""

    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials"""
        pass

    @abstractmethod
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate authentication token"""
        pass

    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> Optional[str]:
        """Refresh authentication token"""
        pass

    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke authentication token"""
        pass


class IPermissionChecker(ABC):
    """Permission checking interface"""

    @abstractmethod
    async def has_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has permission for action on resource"""
        pass

    @abstractmethod
    async def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for user"""
        pass

    @abstractmethod
    async def grant_permission(self, user_id: str, permission: str) -> bool:
        """Grant permission to user"""
        pass

    @abstractmethod
    async def revoke_permission(self, user_id: str, permission: str) -> bool:
        """Revoke permission from user"""
        pass


class ISecurityManager(ABC):
    """Security manager interface"""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize security manager"""
        pass

    @abstractmethod
    async def authenticate_request(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate incoming request"""
        pass

    @abstractmethod
    async def authorize_action(
        self, user_context: Dict[str, Any], resource: str, action: str
    ) -> bool:
        """Authorize user action"""
        pass

    @abstractmethod
    async def validate_input(self, input_data: Any, validation_rules: Dict[str, Any]) -> bool:
        """Validate input data"""
        pass

    @abstractmethod
    async def encrypt_data(self, data: str, key_id: str = None) -> str:
        """Encrypt sensitive data"""
        pass

    @abstractmethod
    async def decrypt_data(self, encrypted_data: str, key_id: str = None) -> str:
        """Decrypt sensitive data"""
        pass

    @abstractmethod
    async def audit_log(self, event: Dict[str, Any]):
        """Log security event for audit"""
        pass


class IRateLimiter(ABC):
    """Rate limiting interface"""

    @abstractmethod
    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        """Check if rate limit is exceeded"""
        pass

    @abstractmethod
    async def get_remaining_quota(self, key: str, limit: int, window_seconds: int) -> int:
        """Get remaining quota for key"""
        pass

    @abstractmethod
    async def reset_rate_limit(self, key: str) -> bool:
        """Reset rate limit for key"""
        pass


class IInputValidator(ABC):
    """Input validation interface"""

    @abstractmethod
    def validate_string(
        self, value: str, max_length: int = None, allowed_chars: str = None
    ) -> bool:
        """Validate string input"""
        pass

    @abstractmethod
    def validate_number(
        self, value: Union[int, float], min_value: float = None, max_value: float = None
    ) -> bool:
        """Validate numeric input"""
        pass

    @abstractmethod
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pass

    @abstractmethod
    def validate_json(self, json_str: str, schema: Dict[str, Any] = None) -> bool:
        """Validate JSON input"""
        pass

    @abstractmethod
    def sanitize_input(self, value: Any) -> Any:
        """Sanitize input to prevent injection attacks"""
        pass


class ICryptoProvider(ABC):
    """Cryptographic provider interface"""

    @abstractmethod
    async def generate_key(self, key_type: str, key_size: int = None) -> str:
        """Generate cryptographic key"""
        pass

    @abstractmethod
    async def encrypt(self, data: bytes, key: str, algorithm: str = None) -> bytes:
        """Encrypt data"""
        pass

    @abstractmethod
    async def decrypt(self, encrypted_data: bytes, key: str, algorithm: str = None) -> bytes:
        """Decrypt data"""
        pass

    @abstractmethod
    async def sign(self, data: bytes, private_key: str) -> bytes:
        """Sign data"""
        pass

    @abstractmethod
    async def verify_signature(self, data: bytes, signature: bytes, public_key: str) -> bool:
        """Verify digital signature"""
        pass

    @abstractmethod
    async def hash_data(self, data: bytes, algorithm: str = "sha256") -> str:
        """Hash data"""
        pass


class ISecurityAuditor(ABC):
    """Security audit interface"""

    @abstractmethod
    async def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event"""
        pass

    @abstractmethod
    async def get_security_events(
        self, filters: Dict[str, Any] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get security events"""
        pass

    @abstractmethod
    async def analyze_threats(self, time_window: int = 3600) -> Dict[str, Any]:
        """Analyze security threats"""
        pass

    @abstractmethod
    async def generate_security_report(self, report_type: str) -> Dict[str, Any]:
        """Generate security report"""
        pass
