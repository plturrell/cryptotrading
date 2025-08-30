#!/usr/bin/env python3
"""
Initialize enhanced database tables for 58 factors
"""

import sys

sys.path.append("src")

from sqlalchemy import create_engine

from cryptotrading.data.database.models import Base
from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase


def initialize_enhanced_tables():
    """Create all enhanced tables in the database"""
    print("🚀 Initializing enhanced database tables...")

    # Get database
    db = UnifiedDatabase()

    # Initialize the database first
    db.initialize()

    # Create all tables defined in models.py
    print("Creating tables from models...")

    # Get the database URL
    db_url = f"sqlite:///data/cryptotrading.db"

    # Create engine
    engine = create_engine(db_url)

    # Create all tables
    Base.metadata.create_all(engine)

    # List all tables created
    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print(f"\n✅ Created {len(tables)} tables:")
    for table in sorted(tables):
        print(f"   - {table}")

    # Check for enhanced tables
    enhanced_tables = [
        "time_series",
        "factor_data",
        "onchain_data",
        "sentiment_data",
        "macro_data",
        "data_quality_metrics",
        "data_ingestion_jobs",
    ]

    missing_tables = [t for t in enhanced_tables if t not in tables]
    if missing_tables:
        print(f"\n⚠️  Warning: Missing tables: {missing_tables}")
    else:
        print(f"\n✅ All enhanced tables created successfully!")

    return len(missing_tables) == 0


if __name__ == "__main__":
    success = initialize_enhanced_tables()
    if success:
        print("\n✅ Database initialization complete!")
    else:
        print("\n❌ Some tables failed to create")
        sys.exit(1)
