"""
SQLite database client for рекс.com
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from .models import Base
import logging

logger = logging.getLogger(__name__)

class DatabaseClient:
    def __init__(self, db_path: str = None):
        """Initialize SQLite database client"""
        if db_path is None:
            # Use /tmp for Vercel serverless functions
            if os.getenv('VERCEL'):
                db_path = '/tmp/рекс.db'
            else:
                db_path = os.getenv('DATABASE_PATH', 'data/рекс.db')
        
        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        self.db_url = f'sqlite:///{db_path}'
        self.engine = create_engine(
            self.db_url,
            connect_args={'check_same_thread': False},
            echo=False
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Create scoped session for thread safety
        self.Session = scoped_session(self.SessionLocal)
        
        # Initialize database
        self.init_db()
    
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