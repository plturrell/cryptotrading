"""
Database test utilities and mock implementations
"""

import logging

logger = logging.getLogger(__name__)

class MockPool:
    """Mock connection pool for testing without database dependencies"""
    
    def __init__(self):
        pass
    
    async def acquire(self):
        return MockConnection()
    
    async def close(self):
        pass

class MockConnection:
    """Mock database connection"""
    
    def __init__(self):
        pass
    
    async def execute(self, query, *args):
        logger.debug(f"Mock execution: {query[:50]}...")
        return "MOCK_RESULT"
    
    async def fetch(self, query, *args):
        logger.debug(f"Mock fetch: {query[:50]}...")
        return []
    
    async def fetchrow(self, query, *args):
        logger.debug(f"Mock fetchrow: {query[:50]}...")
        return None