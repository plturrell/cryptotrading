#!/usr/bin/env python3
"""
Create all database tables for the cryptocurrency trading platform
"""

import os
import sqlite3
import sys

# Database path
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rex_trading.db"
)


def create_tables():
    """Create all necessary database tables"""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Users table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            role TEXT DEFAULT 'trader',
            status TEXT DEFAULT 'active',
            avatar_url TEXT,
            phone TEXT,
            language TEXT DEFAULT 'en',
            timezone TEXT DEFAULT 'UTC',
            two_factor_enabled INTEGER DEFAULT 0,
            two_factor_secret TEXT,
            api_key TEXT UNIQUE,
            last_login TIMESTAMP,
            login_attempts INTEGER DEFAULT 0,
            locked_until TIMESTAMP,
            password_reset_token TEXT,
            password_reset_expires TIMESTAMP,
            email_verified INTEGER DEFAULT 0,
            email_verification_token TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
        )

        # User preferences table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            theme TEXT DEFAULT 'dark',
            trading_view TEXT DEFAULT 'advanced',
            default_exchange TEXT DEFAULT 'binance',
            currency TEXT DEFAULT 'USD',
            notifications_enabled INTEGER DEFAULT 1,
            email_notifications INTEGER DEFAULT 1,
            sms_notifications INTEGER DEFAULT 0,
            price_alerts INTEGER DEFAULT 1,
            news_alerts INTEGER DEFAULT 1,
            chart_indicators TEXT,
            favorite_pairs TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )"""
        )

        # User sessions table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            logout_time TIMESTAMP,
            expires_at TIMESTAMP,
            is_active INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )"""
        )

        # User audit log table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS user_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            user_agent TEXT,
            status TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )"""
        )

        # API credentials table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS api_credentials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            api_secret_hash TEXT NOT NULL,
            name TEXT,
            permissions TEXT,
            rate_limit INTEGER DEFAULT 1000,
            calls_made INTEGER DEFAULT 0,
            last_used TIMESTAMP,
            expires_at TIMESTAMP,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )"""
        )

        # Trading pairs table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS trading_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            base_currency TEXT NOT NULL,
            quote_currency TEXT NOT NULL,
            exchange TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            min_trade_size REAL,
            max_trade_size REAL,
            tick_size REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
        )

        # Market data table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            trades_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
        )

        # Orders table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            order_type TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL,
            quantity REAL NOT NULL,
            filled_quantity REAL DEFAULT 0,
            status TEXT DEFAULT 'pending',
            exchange TEXT,
            exchange_order_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )"""
        )

        # Portfolio table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            asset TEXT NOT NULL,
            quantity REAL NOT NULL,
            average_price REAL,
            current_price REAL,
            pnl REAL,
            pnl_percentage REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )"""
        )

        # Watchlist table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            notes TEXT,
            alert_price REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )"""
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio(user_id)")

        conn.commit()

        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        print("✅ Database tables created successfully!")
        print("\n📋 Created tables:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"   - {table[0]} ({count} records)")

    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    create_tables()
