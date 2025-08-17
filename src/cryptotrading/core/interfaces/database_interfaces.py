"""
Database Interface Definitions
Abstract interfaces for database components to prevent circular dependencies
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from datetime import datetime


class IConnectionPool(ABC):
    """Database connection pool interface"""
    
    @abstractmethod
    async def get_connection(self) -> Any:
        """Get a database connection"""
        pass
    
    @abstractmethod
    async def return_connection(self, connection: Any):
        """Return connection to pool"""
        pass
    
    @abstractmethod
    async def close_all(self):
        """Close all connections"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        pass


class IDatabaseClient(ABC):
    """Database client interface"""
    
    @abstractmethod
    async def execute(self, query: str, parameters: List[Any] = None) -> Any:
        """Execute a database query"""
        pass
    
    @abstractmethod
    async def fetch_one(self, query: str, parameters: List[Any] = None) -> Optional[Dict[str, Any]]:
        """Fetch single result"""
        pass
    
    @abstractmethod
    async def fetch_all(self, query: str, parameters: List[Any] = None) -> List[Dict[str, Any]]:
        """Fetch all results"""
        pass
    
    @abstractmethod
    async def begin_transaction(self) -> Any:
        """Begin a transaction"""
        pass
    
    @abstractmethod
    async def commit_transaction(self, transaction: Any):
        """Commit a transaction"""
        pass
    
    @abstractmethod
    async def rollback_transaction(self, transaction: Any):
        """Rollback a transaction"""
        pass


class IDatabaseManager(ABC):
    """Database manager interface"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize database manager"""
        pass
    
    @abstractmethod
    async def get_client(self, database_name: str = None) -> IDatabaseClient:
        """Get database client"""
        pass
    
    @abstractmethod
    async def create_tables(self, schema: Dict[str, Any]) -> bool:
        """Create database tables"""
        pass
    
    @abstractmethod
    async def migrate(self, migration_scripts: List[str]) -> bool:
        """Run database migrations"""
        pass
    
    @abstractmethod
    async def backup(self, backup_path: str) -> bool:
        """Create database backup"""
        pass
    
    @abstractmethod
    async def restore(self, backup_path: str) -> bool:
        """Restore from backup"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        pass


class IRepository(ABC):
    """Repository pattern interface"""
    
    @abstractmethod
    async def create(self, entity: Dict[str, Any]) -> str:
        """Create new entity"""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def update(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity"""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete entity"""
        pass
    
    @abstractmethod
    async def list_entities(self, filters: Dict[str, Any] = None,
                          limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List entities with filtering"""
        pass
    
    @abstractmethod
    async def count(self, filters: Dict[str, Any] = None) -> int:
        """Count entities"""
        pass


class IQueryBuilder(ABC):
    """Query builder interface"""
    
    @abstractmethod
    def select(self, fields: List[str] = None) -> 'IQueryBuilder':
        """Add SELECT clause"""
        pass
    
    @abstractmethod
    def from_table(self, table: str) -> 'IQueryBuilder':
        """Add FROM clause"""
        pass
    
    @abstractmethod
    def where(self, condition: str, parameters: List[Any] = None) -> 'IQueryBuilder':
        """Add WHERE clause"""
        pass
    
    @abstractmethod
    def join(self, table: str, condition: str, join_type: str = "INNER") -> 'IQueryBuilder':
        """Add JOIN clause"""
        pass
    
    @abstractmethod
    def order_by(self, field: str, direction: str = "ASC") -> 'IQueryBuilder':
        """Add ORDER BY clause"""
        pass
    
    @abstractmethod
    def limit(self, count: int, offset: int = 0) -> 'IQueryBuilder':
        """Add LIMIT clause"""
        pass
    
    @abstractmethod
    def build(self) -> tuple[str, List[Any]]:
        """Build the final query and parameters"""
        pass