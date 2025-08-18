"""
Create factor_definitions table to store factor metadata
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Enum
from datetime import datetime
import enum
from .models import Base


class FactorCategory(enum.Enum):
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    VOLATILITY = "volatility"


class FactorDefinition(Base):
    """Store factor definitions and metadata"""
    __tablename__ = 'factor_definitions'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    category = Column(Enum(FactorCategory), nullable=False)
    description = Column(Text)
    
    # Calculation parameters
    lookback_hours = Column(Float)  # How much historical data needed
    update_frequency_minutes = Column(Integer, default=15)  # How often to recalculate
    
    # Validation rules stored as JSON
    min_value = Column(Float)
    max_value = Column(Float)
    
    # Technical details
    formula = Column(Text)  # Mathematical formula or description
    dependencies = Column(Text)  # Comma-separated list of required inputs
    
    # Metadata
    is_active = Column(Integer, default=1)  # Boolean flag
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Script to populate factor definitions
FACTOR_DEFINITIONS = [
    {
        'name': 'spot_price',
        'category': FactorCategory.PRICE,
        'description': 'Current market price',
        'lookback_hours': 0.0,
        'update_frequency_minutes': 1,
        'min_value': 0.0,
        'dependencies': 'close'
    },
    {
        'name': 'sma_20',
        'category': FactorCategory.PRICE,
        'description': '20-period Simple Moving Average',
        'lookback_hours': 20.0,
        'update_frequency_minutes': 15,
        'min_value': 0.0,
        'formula': 'SUM(close, 20) / 20',
        'dependencies': 'close'
    },
    {
        'name': 'ema_50',
        'category': FactorCategory.PRICE,
        'description': '50-period Exponential Moving Average',
        'lookback_hours': 50.0,
        'update_frequency_minutes': 15,
        'min_value': 0.0,
        'formula': 'EMA calculation with alpha=2/(50+1)',
        'dependencies': 'close'
    },
    {
        'name': 'volume',
        'category': FactorCategory.VOLUME,
        'description': 'Trading volume',
        'lookback_hours': 0.0,
        'update_frequency_minutes': 1,
        'min_value': 0.0,
        'dependencies': 'volume'
    },
    {
        'name': 'obv',
        'category': FactorCategory.VOLUME,
        'description': 'On-Balance Volume indicator',
        'lookback_hours': 720.0,  # 30 days
        'update_frequency_minutes': 15,
        'formula': 'Cumulative sum of signed volume',
        'dependencies': 'close,volume'
    },
    {
        'name': 'rsi_14',
        'category': FactorCategory.TECHNICAL,
        'description': '14-period Relative Strength Index',
        'lookback_hours': 14.0,
        'update_frequency_minutes': 15,
        'min_value': 0.0,
        'max_value': 100.0,
        'formula': '100 - (100 / (1 + RS))',
        'dependencies': 'close'
    },
    {
        'name': 'macd',
        'category': FactorCategory.TECHNICAL,
        'description': 'MACD line (12-EMA - 26-EMA)',
        'lookback_hours': 26.0,
        'update_frequency_minutes': 15,
        'formula': 'EMA(close, 12) - EMA(close, 26)',
        'dependencies': 'close'
    },
    {
        'name': 'macd_signal',
        'category': FactorCategory.TECHNICAL,
        'description': 'MACD signal line (9-EMA of MACD)',
        'lookback_hours': 35.0,  # 26 + 9
        'update_frequency_minutes': 15,
        'formula': 'EMA(MACD, 9)',
        'dependencies': 'close'
    },
    {
        'name': 'bb_position',
        'category': FactorCategory.TECHNICAL,
        'description': 'Position within Bollinger Bands (0-1)',
        'lookback_hours': 20.0,
        'update_frequency_minutes': 15,
        'min_value': -0.5,
        'max_value': 1.5,
        'formula': '(close - BB_lower) / (BB_upper - BB_lower)',
        'dependencies': 'close'
    },
    {
        'name': 'stoch_k',
        'category': FactorCategory.TECHNICAL,
        'description': 'Stochastic %K',
        'lookback_hours': 14.0,
        'update_frequency_minutes': 15,
        'min_value': 0.0,
        'max_value': 100.0,
        'formula': '100 * (close - min(low, 14)) / (max(high, 14) - min(low, 14))',
        'dependencies': 'high,low,close'
    },
    {
        'name': 'atr',
        'category': FactorCategory.VOLATILITY,
        'description': 'Average True Range',
        'lookback_hours': 14.0,
        'update_frequency_minutes': 15,
        'min_value': 0.0,
        'formula': 'SMA of True Range over 14 periods',
        'dependencies': 'high,low,close'
    },
    {
        'name': 'volatility',
        'category': FactorCategory.VOLATILITY,
        'description': '24-hour realized volatility',
        'lookback_hours': 24.0,
        'update_frequency_minutes': 15,
        'min_value': 0.0,
        'max_value': 5.0,
        'formula': 'Standard deviation of hourly returns * sqrt(24)',
        'dependencies': 'close'
    }
]


def populate_factor_definitions(db_session):
    """Populate the factor_definitions table"""
    for factor_def in FACTOR_DEFINITIONS:
        factor = FactorDefinition(**factor_def)
        db_session.add(factor)
    
    db_session.commit()
    print(f"Added {len(FACTOR_DEFINITIONS)} factor definitions to database")