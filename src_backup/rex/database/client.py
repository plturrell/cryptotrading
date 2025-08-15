"""
Production database client with connection pooling for rex.com
Supports both SQLite and PostgreSQL with proper connection management
"""

import os
import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool, StaticPool
from contextlib import contextmanager
from .models import Base

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration with production defaults"""
    # Connection pooling settings
    POOL_SIZE = 20  # Number of connections to maintain in pool
    MAX_OVERFLOW = 30  # Additional connections beyond pool_size
    POOL_TIMEOUT = 30  # Seconds to wait for connection
    POOL_RECYCLE = 3600  # Seconds before recreating connections (1 hour)
    POOL_PRE_PING = True  # Test connections before use
    
    # SQLite specific settings
    SQLITE_POOL_SIZE = 5  # Smaller pool for SQLite
    SQLITE_TIMEOUT = 20  # SQLite lock timeout
    
    # PostgreSQL specific settings  
    POSTGRES_CONNECT_TIMEOUT = 10
    POSTGRES_COMMAND_TIMEOUT = 60

class DatabaseClient:
    def __init__(self, db_url: str = None, config: DatabaseConfig = None):
        """Initialize database client with connection pooling"""
        self.config = config or DatabaseConfig()
        
        if db_url:
            self.db_url = db_url
        else:
            self.db_url = self._get_database_url()
        
        self.is_sqlite = 'sqlite' in self.db_url
        self.is_postgres = 'postgresql' in self.db_url
        
        # Create engine with appropriate pooling
        self.engine = self._create_engine()
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False  # Keep objects accessible after commit
        )
        
        # Create scoped session for thread safety
        self.Session = scoped_session(self.SessionLocal)
        
        # Initialize database
        self.init_db()
        
        # Setup connection event listeners
        self._setup_event_listeners()
        
        logger.info(f"Database client initialized with {self.db_url}")
    
    def _get_database_url(self) -> str:
        """Get database URL from environment with fallbacks"""
        # Try production database URL first
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url
        
        # Try specific database configuration
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'reks_production')
        db_user = os.getenv('DB_USER', 'reks_user')
        db_password = os.getenv('DB_PASSWORD')
        
        if db_password:
            return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Fall back to SQLite
        if os.getenv('VERCEL'):
            db_path = '/tmp/rex.db'
        else:
            db_path = os.getenv('DATABASE_PATH', 'data/rex.db')
        
        # Ensure directory exists for SQLite
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        return f'sqlite:///{db_path}'
    
    def _create_engine(self):
        """Create database engine with appropriate pooling configuration"""
        if self.is_sqlite:
            return self._create_sqlite_engine()
        elif self.is_postgres:
            return self._create_postgres_engine()
        else:
            # Generic engine
            return create_engine(
                self.db_url,
                poolclass=QueuePool,
                pool_size=self.config.POOL_SIZE,
                max_overflow=self.config.MAX_OVERFLOW,
                pool_timeout=self.config.POOL_TIMEOUT,
                pool_recycle=self.config.POOL_RECYCLE,
                pool_pre_ping=self.config.POOL_PRE_PING,
                echo=False
            )
    
    def _create_sqlite_engine(self):
        """Create SQLite engine with optimized settings"""
        connect_args = {
            'check_same_thread': False,
            'timeout': self.config.SQLITE_TIMEOUT,
        }
        
        # For SQLite, use StaticPool with minimal configuration
        return create_engine(
            self.db_url,
            poolclass=StaticPool,
            connect_args=connect_args,
            echo=False
        )
    
    def _create_postgres_engine(self):
        """Create PostgreSQL engine with production settings"""
        connect_args = {
            'connect_timeout': self.config.POSTGRES_CONNECT_TIMEOUT,
            'application_name': 'reks_a2a_system',
            'options': '-c statement_timeout={}s'.format(self.config.POSTGRES_COMMAND_TIMEOUT)
        }
        
        return create_engine(
            self.db_url,
            poolclass=QueuePool,
            pool_size=self.config.POOL_SIZE,
            max_overflow=self.config.MAX_OVERFLOW,
            pool_timeout=self.config.POOL_TIMEOUT,
            pool_recycle=self.config.POOL_RECYCLE,
            pool_pre_ping=self.config.POOL_PRE_PING,
            connect_args=connect_args,
            echo=False
        )
    
    def _setup_event_listeners(self):
        """Setup database event listeners for monitoring and optimization"""
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for performance"""
            if self.is_sqlite:
                cursor = dbapi_connection.cursor()
                # Performance optimizations
                cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
                cursor.execute("PRAGMA synchronous=NORMAL")  # Faster than FULL
                cursor.execute("PRAGMA cache_size=10000")  # 10MB cache
                cursor.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
                cursor.close()
        
        @event.listens_for(self.engine, "checkout")
        def ping_connection(dbapi_connection, connection_record, connection_proxy):
            """Ensure connection is alive on checkout"""
            if self.config.POOL_PRE_PING:
                try:
                    # Test the connection
                    dbapi_connection.execute("SELECT 1").fetchone()
                except Exception as e:
                    logger.warning(f"Connection ping failed: {e}")
                    # Invalidate the connection
                    connection_record.info['invalidated'] = True
                    raise
        
        @event.listens_for(self.engine, "invalidate")
        def handle_invalidated(dbapi_connection, connection_record, exception):
            """Handle invalidated connections"""
            logger.warning(f"Connection invalidated: {exception}")
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status for monitoring"""
        pool = self.engine.pool
        
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalidated(),
            "total_connections": pool.size() + pool.overflow(),
            "pool_class": pool.__class__.__name__
        }
    
    def init_db(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session context manager"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query, params=None):
        """Execute raw SQL query"""
        with self.engine.connect() as conn:
            result = conn.execute(query, params or {})
            return result.fetchall()
    
    def add_trade(self, user_id: int, symbol: str, side: str, quantity: float, price: float, **kwargs):
        """Add new trade to database"""
        from .models import Trade
        
        with self.get_session() as session:
            trade = Trade(
                user_id=user_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                total=quantity * price,
                **kwargs
            )
            session.add(trade)
            return trade.id
    
    def get_user_trades(self, user_id: int, limit: int = 100):
        """Get user's trade history"""
        from .models import Trade
        
        with self.get_session() as session:
            trades = session.query(Trade)\
                .filter(Trade.user_id == user_id)\
                .order_by(Trade.executed_at.desc())\
                .limit(limit)\
                .all()
            return trades
    
    def save_ai_analysis(self, symbol: str, model: str, analysis_type: str, analysis: str, **kwargs):
        """Save AI analysis to database"""
        from .models import AIAnalysis
        
        with self.get_session() as session:
            ai_analysis = AIAnalysis(
                symbol=symbol,
                model=model,
                analysis_type=analysis_type,
                analysis=analysis,
                **kwargs
            )
            session.add(ai_analysis)
            session.flush()  # Flush to get the ID
            analysis_id = ai_analysis.id
            return analysis_id
    
    def get_latest_analysis(self, symbol: str, model: str = None):
        """Get latest AI analysis for symbol"""
        from .models import AIAnalysis
        
        with self.get_session() as session:
            query = session.query(AIAnalysis).filter(AIAnalysis.symbol == symbol)
            
            if model:
                query = query.filter(AIAnalysis.model == model)
            
            return query.order_by(AIAnalysis.created_at.desc()).first()
    
    def update_portfolio(self, user_id: int, symbol: str, quantity: float, average_price: float):
        """Update user portfolio"""
        from .models import Portfolio
        
        with self.get_session() as session:
            portfolio = session.query(Portfolio)\
                .filter(Portfolio.user_id == user_id, Portfolio.symbol == symbol)\
                .first()
            
            if portfolio:
                # Update existing position
                total_quantity = portfolio.quantity + quantity
                if total_quantity > 0:
                    portfolio.average_price = (
                        (portfolio.quantity * portfolio.average_price + quantity * average_price) 
                        / total_quantity
                    )
                    portfolio.quantity = total_quantity
                else:
                    # Position closed
                    session.delete(portfolio)
            else:
                # Create new position
                portfolio = Portfolio(
                    user_id=user_id,
                    symbol=symbol,
                    quantity=quantity,
                    average_price=average_price
                )
                session.add(portfolio)
    
    def close(self):
        """Close database connection"""
        self.Session.remove()
        self.engine.dispose()

# Global database instance
db_client = None

def get_db():
    """Get database client instance"""
    global db_client
    if db_client is None:
        db_client = DatabaseClient()
    return db_client