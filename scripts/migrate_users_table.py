#!/usr/bin/env python3
"""
Migration script to add missing columns to users table
"""

import sqlite3
import sys
import os

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rex_trading.db')

def migrate_users_table():
    """Add missing columns to users table"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check current schema
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        existing_columns = {col[1] for col in columns}
        
        print(f"Existing columns: {existing_columns}")
        
        # Add missing columns if they don't exist
        migrations = [
            ("first_name", "TEXT"),
            ("last_name", "TEXT"),
            ("password_hash", "TEXT"),
            ("role", "TEXT DEFAULT 'trader'"),
            ("status", "TEXT DEFAULT 'active'"),
            ("language", "TEXT DEFAULT 'en'"),
            ("timezone", "TEXT DEFAULT 'UTC'"),
            ("phone", "TEXT"),
            ("two_factor_enabled", "INTEGER DEFAULT 0"),
            ("two_factor_secret", "TEXT"),
            ("last_login", "TIMESTAMP"),
            ("login_attempts", "INTEGER DEFAULT 0"),
            ("locked_until", "TIMESTAMP"),
            ("password_reset_token", "TEXT"),
            ("password_reset_expires", "TIMESTAMP"),
            ("email_verified", "INTEGER DEFAULT 0"),
            ("email_verification_token", "TEXT"),
            ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        ]
        
        for column_name, column_type in migrations:
            if column_name not in existing_columns:
                print(f"Adding column: {column_name} {column_type}")
                try:
                    cursor.execute(f"ALTER TABLE users ADD COLUMN {column_name} {column_type}")
                    conn.commit()
                    print(f"✅ Added column: {column_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        print(f"⚠️ Warning for {column_name}: {e}")
        
        # Verify final schema
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        print("\n📋 Final users table schema:")
        for col in columns:
            print(f"   {col[1]} ({col[2]})")
        
        conn.commit()
        print("\n✅ Users table migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_users_table()