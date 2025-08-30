"""
User Management System with Authentication
Supports Craig, Irina, Dasha, and Dany as initial users
"""

import hashlib
import json
import logging
import secrets
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import jwt

logger = logging.getLogger(__name__)


class UserRole(Enum):
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"


class UserStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


@dataclass
class User:
    id: int
    username: str
    email: str
    first_name: str
    last_name: str
    role: UserRole
    status: UserStatus
    avatar_url: Optional[str] = None
    phone: Optional[str] = None
    language: str = "en"
    timezone: str = "UTC"
    two_factor_enabled: bool = False
    last_login: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class UserPreferences:
    user_id: int
    theme: str = "dark"
    language: str = "en"
    currency: str = "USD"
    notifications_enabled: bool = True
    email_notifications: bool = True
    sms_notifications: bool = False
    trading_view: str = "advanced"
    default_exchange: str = "binance"
    chart_indicators: List[str] = None
    favorite_pairs: List[str] = None


class UserManagementService:
    """
    Complete user management with authentication and authorization
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use the rex_trading.db in project root
            import os

            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                )
            )
            self.db_path = os.path.join(project_root, "rex_trading.db")
        else:
            self.db_path = db_path
        self.jwt_secret = secrets.token_urlsafe(32)
        self._init_database()
        self._create_initial_users()

    def _init_database(self):
        """Initialize user management tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Enhanced users table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    first_name VARCHAR(50) NOT NULL,
                    last_name VARCHAR(50) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'active',
                    avatar_url VARCHAR(255),
                    phone VARCHAR(20),
                    language VARCHAR(5) DEFAULT 'en',
                    timezone VARCHAR(50) DEFAULT 'UTC',
                    two_factor_enabled BOOLEAN DEFAULT 0,
                    two_factor_secret VARCHAR(100),
                    api_key VARCHAR(100) UNIQUE,
                    last_login TIMESTAMP,
                    login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # User preferences table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE NOT NULL,
                    theme VARCHAR(20) DEFAULT 'dark',
                    language VARCHAR(5) DEFAULT 'en',
                    currency VARCHAR(10) DEFAULT 'USD',
                    notifications_enabled BOOLEAN DEFAULT 1,
                    email_notifications BOOLEAN DEFAULT 1,
                    sms_notifications BOOLEAN DEFAULT 0,
                    trading_view VARCHAR(20) DEFAULT 'advanced',
                    default_exchange VARCHAR(50) DEFAULT 'binance',
                    chart_indicators TEXT,
                    favorite_pairs TEXT,
                    custom_settings TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """
            )

            # User sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token VARCHAR(255) UNIQUE NOT NULL,
                    ip_address VARCHAR(45),
                    user_agent TEXT,
                    device_info TEXT,
                    location VARCHAR(100),
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """
            )

            # Audit log table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    details TEXT,
                    ip_address VARCHAR(45),
                    user_agent TEXT,
                    status VARCHAR(20),
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """
            )

            conn.commit()

    def _create_initial_users(self):
        """Create the four initial users"""
        initial_users = [
            {
                "username": "craig",
                "email": "craig@rex.com",
                "password": "Craig2024!",
                "first_name": "Craig",
                "last_name": "Wright",
                "role": UserRole.ADMIN,
                "avatar_url": "/images/avatars/craig.jpg",
                "language": "en",
                "phone": "+1-555-0101",
            },
            {
                "username": "irina",
                "email": "irina@rex.com",
                "password": "Irina2024!",
                "first_name": "Irina",
                "last_name": "Petrova",
                "role": UserRole.TRADER,
                "avatar_url": "/images/avatars/irina.jpg",
                "language": "ru",
                "phone": "+7-495-0102",
            },
            {
                "username": "dasha",
                "email": "dasha@rex.com",
                "password": "Dasha2024!",
                "first_name": "Dasha",
                "last_name": "Ivanova",
                "role": UserRole.ANALYST,
                "avatar_url": "/images/avatars/dasha.jpg",
                "language": "ru",
                "phone": "+7-495-0103",
            },
            {
                "username": "dany",
                "email": "dany@rex.com",
                "password": "Dany2024!",
                "first_name": "Dany",
                "last_name": "Chen",
                "role": UserRole.TRADER,
                "avatar_url": "/images/avatars/dany.jpg",
                "language": "en",
                "phone": "+1-555-0104",
            },
        ]

        for user_data in initial_users:
            try:
                # Check if user already exists
                if not self.get_user_by_username(user_data["username"]):
                    self.create_user(**user_data)
                    logger.info(f"Created initial user: {user_data['username']}")

                    # Set up default preferences
                    self._create_default_preferences(user_data["username"])
            except Exception as e:
                logger.error(f"Error creating user {user_data['username']}: {e}")

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        first_name: str,
        last_name: str,
        role: UserRole,
        **kwargs,
    ) -> Optional[int]:
        """Create a new user"""
        password_hash = self._hash_password(password)
        api_key = secrets.token_urlsafe(32)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO users (
                        username, email, password_hash, first_name, last_name,
                        role, status, avatar_url, phone, language, api_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        username,
                        email,
                        password_hash,
                        first_name,
                        last_name,
                        role.value if isinstance(role, UserRole) else role,
                        kwargs.get("status", "active"),
                        kwargs.get("avatar_url"),
                        kwargs.get("phone"),
                        kwargs.get("language", "en"),
                        api_key,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError as e:
                logger.error(f"User creation failed: {e}")
                return None

    def _create_default_preferences(self, username: str):
        """Create default preferences for a user"""
        user = self.get_user_by_username(username)
        if not user:
            return

        # Customize preferences based on user
        preferences = {
            "craig": {
                "theme": "light",
                "trading_view": "professional",
                "favorite_pairs": ["BTC/USD", "ETH/USD", "SOL/USD"],
                "chart_indicators": ["RSI", "MACD", "Bollinger"],
            },
            "irina": {
                "theme": "dark",
                "language": "ru",
                "trading_view": "advanced",
                "favorite_pairs": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                "chart_indicators": ["Volume", "RSI", "SMA"],
            },
            "dasha": {
                "theme": "dark",
                "language": "ru",
                "trading_view": "analyst",
                "favorite_pairs": ["BTC/USD", "ETH/USD"],
                "chart_indicators": ["MACD", "RSI", "Fibonacci"],
            },
            "dany": {
                "theme": "dark",
                "trading_view": "advanced",
                "favorite_pairs": ["BTC/USDT", "SOL/USDT", "MATIC/USDT"],
                "chart_indicators": ["EMA", "RSI", "Volume"],
            },
        }

        user_prefs = preferences.get(username, {})

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO user_preferences (
                    user_id, theme, language, trading_view, 
                    favorite_pairs, chart_indicators
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    user["id"],
                    user_prefs.get("theme", "dark"),
                    user_prefs.get("language", user.get("language", "en")),
                    user_prefs.get("trading_view", "advanced"),
                    json.dumps(user_prefs.get("favorite_pairs", [])),
                    json.dumps(user_prefs.get("chart_indicators", [])),
                ),
            )
            conn.commit()

    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return JWT token"""
        user = self.get_user_by_username(username)

        if not user or user["status"] != "active":
            self._log_audit(username, "login_failed", "Invalid user or inactive")
            return None

        # Check if account is locked
        if user.get("locked_until"):
            locked_until = datetime.fromisoformat(user["locked_until"])
            if locked_until > datetime.now():
                self._log_audit(user["id"], "login_failed", "Account locked")
                return None

        # Verify password
        if not self._verify_password(password, user["password_hash"]):
            self._handle_failed_login(user["id"])
            return None

        # Create session
        session_token = self._create_session(user["id"])

        # Generate JWT token
        jwt_token = self._generate_jwt(user)

        # Update last login
        self._update_last_login(user["id"])

        # Log successful login
        self._log_audit(user["id"], "login_success", "User logged in successfully")

        return {
            "token": jwt_token,
            "session": session_token,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "first_name": user["first_name"],
                "last_name": user["last_name"],
                "role": user["role"],
                "avatar_url": user["avatar_url"],
                "language": user["language"],
            },
        }

    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
        )
        return salt + pwd_hash.hex()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        salt = password_hash[:32]
        stored_hash = password_hash[32:]
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
        )
        return pwd_hash.hex() == stored_hash

    def _generate_jwt(self, user: Dict) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "exp": datetime.utcnow() + timedelta(hours=24),
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def _create_session(self, user_id: int) -> str:
        """Create user session"""
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_sessions (
                    user_id, session_token, expires_at
                ) VALUES (?, ?, ?)
            """,
                (user_id, session_token, expires_at),
            )
            conn.commit()

        return session_token

    def _update_last_login(self, user_id: int):
        """Update user's last login time"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP,
                    login_attempts = 0,
                    locked_until = NULL
                WHERE id = ?
            """,
                (user_id,),
            )
            conn.commit()

    def _handle_failed_login(self, user_id: int):
        """Handle failed login attempt"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users 
                SET login_attempts = login_attempts + 1
                WHERE id = ?
            """,
                (user_id,),
            )

            # Lock account after 5 failed attempts
            cursor.execute("SELECT login_attempts FROM users WHERE id = ?", (user_id,))
            attempts = cursor.fetchone()[0]

            if attempts >= 5:
                locked_until = datetime.now() + timedelta(minutes=30)
                cursor.execute(
                    """
                    UPDATE users 
                    SET locked_until = ?
                    WHERE id = ?
                """,
                    (locked_until, user_id),
                )

            conn.commit()

    def _log_audit(self, user_id: Any, action: str, details: str):
        """Log user action for audit"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_audit_log (
                    user_id, action, details
                ) VALUES (?, ?, ?)
            """,
                (user_id if isinstance(user_id, int) else 0, action, details),
            )
            conn.commit()

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_preferences(self, user_id: int) -> Optional[Dict]:
        """Get user preferences"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                prefs = dict(row)
                # Parse JSON fields
                if prefs.get("favorite_pairs"):
                    prefs["favorite_pairs"] = json.loads(prefs["favorite_pairs"])
                if prefs.get("chart_indicators"):
                    prefs["chart_indicators"] = json.loads(prefs["chart_indicators"])
                return prefs
            return None

    def update_user_preferences(self, user_id: int, preferences: Dict) -> bool:
        """Update user preferences"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Convert lists to JSON
            if "favorite_pairs" in preferences:
                preferences["favorite_pairs"] = json.dumps(preferences["favorite_pairs"])
            if "chart_indicators" in preferences:
                preferences["chart_indicators"] = json.dumps(preferences["chart_indicators"])

            # Build update query
            fields = []
            values = []
            for key, value in preferences.items():
                fields.append(f"{key} = ?")
                values.append(value)

            values.append(user_id)

            cursor.execute(
                f"""
                UPDATE user_preferences 
                SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """,
                values,
            )

            conn.commit()
            return cursor.rowcount > 0

    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate user session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.*, u.username, u.role 
                FROM user_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ? 
                AND s.expires_at > CURRENT_TIMESTAMP
                AND s.is_active = 1
            """,
                (session_token,),
            )
            row = cursor.fetchone()

            if row:
                # Update last activity
                cursor.execute(
                    """
                    UPDATE user_sessions 
                    SET last_activity = CURRENT_TIMESTAMP
                    WHERE session_token = ?
                """,
                    (session_token,),
                )
                conn.commit()
                return dict(row)

            return None


# Initialize service
user_service = UserManagementService()
