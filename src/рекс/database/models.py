"""
Database models for рекс.com trading platform
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    api_key = Column(String(100), unique=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trades = relationship("Trade", back_populates="user")
    portfolios = relationship("Portfolio", back_populates="user")
    
class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # buy/sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    status = Column(String(20), default='pending')  # pending/completed/cancelled
    order_id = Column(String(100))
    exchange = Column(String(50), default='binance')
    executed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="trades")
    
class Portfolio(Base):
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    average_price = Column(Float, nullable=False)
    current_price = Column(Float)
    pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    
class AIAnalysis(Base):
    __tablename__ = 'ai_analyses'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    model = Column(String(50), nullable=False)  # deepseek-r1, perplexity
    analysis_type = Column(String(50), nullable=False)  # signal, news, market
    signal = Column(String(20))  # BUY/SELL/HOLD
    confidence = Column(Float)
    analysis = Column(Text, nullable=False)
    raw_response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    price = Column(Float, nullable=False)
    volume_24h = Column(Float)
    high_24h = Column(Float)
    low_24h = Column(Float)
    change_24h = Column(Float)
    change_percent_24h = Column(Float)
    market_cap = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
class TradingSignal(Base):
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    signal_type = Column(String(20), nullable=False)  # BUY/SELL/HOLD
    strength = Column(Float)  # 0.0 to 1.0
    rsi = Column(Float)
    macd = Column(Float)
    ma_50 = Column(Float)
    ma_200 = Column(Float)
    volume_signal = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)