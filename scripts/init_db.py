#!/usr/bin/env python3
"""
Initialize rex.com SQLite database
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rex.database import DatabaseClient, User
from datetime import datetime
import hashlib

def init_database():
    """Initialize database with default data"""
    print("🚀 Initializing rex.com database...")
    
    # Create database client
    db = DatabaseClient(db_path='data/rex.db')
    
    # Create admin user
    with db.get_session() as session:
        # Check if admin exists
        admin = session.query(User).filter(User.username == 'admin').first()
        
        if not admin:
            # Create admin user
            admin = User(
                username='admin',
                email='admin@rex.com',
                password_hash=hashlib.sha256('admin123'.encode()).hexdigest(),
                api_key='rex_admin_api_key_2024',
                is_active=True
            )
            session.add(admin)
            print("✅ Admin user created")
        else:
            print("ℹ️  Admin user already exists")
    
    # Create sample market data
    from rex.database.models import MarketData
    
    with db.get_session() as session:
        # Add sample BTC data
        btc_data = MarketData(
            symbol='BTC',
            price=45000.0,
            volume_24h=25000000000.0,
            high_24h=46000.0,
            low_24h=44000.0,
            change_24h=1000.0,
            change_percent_24h=2.27,
            market_cap=880000000000.0
        )
        session.add(btc_data)
        
        # Add sample ETH data
        eth_data = MarketData(
            symbol='ETH',
            price=2500.0,
            volume_24h=15000000000.0,
            high_24h=2600.0,
            low_24h=2400.0,
            change_24h=50.0,
            change_percent_24h=2.04,
            market_cap=300000000000.0
        )
        session.add(eth_data)
        
        print("✅ Sample market data added")
    
    print("✅ Database initialization completed!")
    print(f"📁 Database location: {os.path.abspath('data/rex.db')}")
    
    db.close()

if __name__ == '__main__':
    init_database()