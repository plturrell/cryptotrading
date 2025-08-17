"""
Database client module - compatibility layer for unified database access
Provides get_db() function that returns the unified database instance
"""

import logging
from typing import Optional
from ...infrastructure.database.unified_database import UnifiedDatabase, DatabaseConfig

logger = logging.getLogger(__name__)

# Global database instance
_db_instance: Optional[UnifiedDatabase] = None

def get_db() -> UnifiedDatabase:
    """
    Get global database instance with lazy initialization
    
    Returns:
        UnifiedDatabase: The global database instance
    """
    global _db_instance
    
    if _db_instance is None:
        try:
            _db_instance = UnifiedDatabase()
            # Initialize the database asynchronously
            import asyncio
            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If event loop is already running, schedule the initialization
                    asyncio.create_task(_db_instance.initialize())
                else:
                    # If no event loop is running, run initialization directly
                    loop.run_until_complete(_db_instance.initialize())
            except RuntimeError:
                # No event loop exists, create one and run initialization
                asyncio.run(_db_instance.initialize())
                
            logger.info("Database client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database client: {e}")
            # Return a non-initialized instance for now
            _db_instance = UnifiedDatabase()
    
    return _db_instance

def close_db():
    """Close the global database instance"""
    global _db_instance
    
    if _db_instance:
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(_db_instance.close())
                else:
                    loop.run_until_complete(_db_instance.close())
            except RuntimeError:
                asyncio.run(_db_instance.close())
        except Exception as e:
            logger.error(f"Error closing database: {e}")
        finally:
            _db_instance = None

def reset_db():
    """Reset the global database instance (useful for testing)"""
    global _db_instance
    close_db()
    _db_instance = None

# For backwards compatibility, also provide these functions
def get_database():
    """Alias for get_db()"""
    return get_db()

def initialize_database(config: DatabaseConfig = None) -> UnifiedDatabase:
    """Initialize database with specific config"""
    global _db_instance
    
    if _db_instance is not None:
        close_db()
    
    _db_instance = UnifiedDatabase(config)
    
    try:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_db_instance.initialize())
            else:
                loop.run_until_complete(_db_instance.initialize())
        except RuntimeError:
            asyncio.run(_db_instance.initialize())
    except Exception as e:
        logger.error(f"Failed to initialize database with config: {e}")
    
    return _db_instance