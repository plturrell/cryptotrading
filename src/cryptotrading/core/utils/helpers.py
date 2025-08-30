"""
Helper utilities for crypto trading operations
"""

import hashlib
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union


def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique identifier"""
    unique_id = str(uuid.uuid4())
    return f"{prefix}_{unique_id}" if prefix else unique_id


def hash_data(data: Union[str, Dict, List]) -> str:
    """Generate a hash for data"""
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.sha256(data_str.encode()).hexdigest()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0:
        return default
    return numerator / denominator


def format_percentage(value: float, precision: int = 2) -> str:
    """Format a decimal value as percentage"""
    return f"{value * 100:.{precision}f}%"


def format_currency(amount: float, currency: str = "USD", precision: int = 2) -> str:
    """Format a number as currency"""
    return f"{currency} {amount:,.{precision}f}"


def calculate_change(current: float, previous: float) -> Dict[str, float]:
    """Calculate absolute and percentage change"""
    absolute_change = current - previous
    percentage_change = safe_divide(absolute_change, previous) if previous else 0.0
    
    return {
        "absolute": absolute_change,
        "percentage": percentage_change,
        "formatted_percentage": format_percentage(percentage_change)
    }


def parse_timeframe(timeframe: str) -> timedelta:
    """Parse timeframe string into timedelta"""
    timeframe_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1)
    }
    
    return timeframe_map.get(timeframe, timedelta(minutes=1))


def round_to_precision(value: float, precision: int) -> float:
    """Round a value to specified precision"""
    return round(value, precision)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max"""
    return max(min_val, min(max_val, value))


def normalize_symbol(symbol: str) -> str:
    """Normalize a trading symbol"""
    return symbol.upper().replace("/", "").replace("-", "")


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def get_timestamp(dt: Optional[datetime] = None) -> int:
    """Get Unix timestamp"""
    if dt is None:
        dt = datetime.now()
    return int(dt.timestamp())


def from_timestamp(timestamp: int) -> datetime:
    """Convert Unix timestamp to datetime"""
    return datetime.fromtimestamp(timestamp)


def is_trading_hours(dt: Optional[datetime] = None) -> bool:
    """Check if current time is within trading hours (crypto trades 24/7)"""
    # Crypto markets are always open, but can be used for traditional markets
    return True