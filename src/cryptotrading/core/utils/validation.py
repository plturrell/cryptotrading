"""
Validation utilities for crypto trading platform
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union


def validate_symbol(symbol: str) -> Tuple[bool, Optional[str]]:
    """Validate a trading symbol"""
    if not symbol or not isinstance(symbol, str):
        return False, "Symbol must be a non-empty string"
    
    # Basic symbol validation - alphanumeric with some special chars
    if not re.match(r'^[A-Za-z0-9/_-]+$', symbol):
        return False, "Symbol contains invalid characters"
    
    if len(symbol) < 2 or len(symbol) > 20:
        return False, "Symbol length must be between 2 and 20 characters"
    
    return True, None


def validate_price(price: Union[int, float]) -> Tuple[bool, Optional[str]]:
    """Validate a price value"""
    if price is None:
        return False, "Price cannot be None"
    
    try:
        price = float(price)
    except (ValueError, TypeError):
        return False, "Price must be a valid number"
    
    if price <= 0:
        return False, "Price must be positive"
    
    if price > 1e12:  # Arbitrary large number check
        return False, "Price is unrealistically high"
    
    return True, None


def validate_quantity(quantity: Union[int, float]) -> Tuple[bool, Optional[str]]:
    """Validate a quantity value"""
    if quantity is None:
        return False, "Quantity cannot be None"
    
    try:
        quantity = float(quantity)
    except (ValueError, TypeError):
        return False, "Quantity must be a valid number"
    
    if quantity <= 0:
        return False, "Quantity must be positive"
    
    return True, None


def validate_percentage(percentage: Union[int, float], min_val: float = 0, max_val: float = 100) -> Tuple[bool, Optional[str]]:
    """Validate a percentage value"""
    if percentage is None:
        return False, "Percentage cannot be None"
    
    try:
        percentage = float(percentage)
    except (ValueError, TypeError):
        return False, "Percentage must be a valid number"
    
    if percentage < min_val or percentage > max_val:
        return False, f"Percentage must be between {min_val} and {max_val}"
    
    return True, None


def validate_timeframe(timeframe: str) -> Tuple[bool, Optional[str]]:
    """Validate a timeframe string"""
    if not timeframe or not isinstance(timeframe, str):
        return False, "Timeframe must be a non-empty string"
    
    valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
    
    if timeframe not in valid_timeframes:
        return False, f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
    
    return True, None


def validate_order_side(side: str) -> Tuple[bool, Optional[str]]:
    """Validate an order side"""
    if not side or not isinstance(side, str):
        return False, "Side must be a non-empty string"
    
    side = side.upper()
    valid_sides = ["BUY", "SELL"]
    
    if side not in valid_sides:
        return False, f"Side must be one of: {', '.join(valid_sides)}"
    
    return True, None


def validate_order_type(order_type: str) -> Tuple[bool, Optional[str]]:
    """Validate an order type"""
    if not order_type or not isinstance(order_type, str):
        return False, "Order type must be a non-empty string"
    
    order_type = order_type.upper()
    valid_types = ["MARKET", "LIMIT", "STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT"]
    
    if order_type not in valid_types:
        return False, f"Order type must be one of: {', '.join(valid_types)}"
    
    return True, None


def validate_risk_parameters(risk_params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate risk management parameters"""
    required_fields = ["max_position_size", "max_daily_loss", "stop_loss_percentage"]
    
    for field in required_fields:
        if field not in risk_params:
            return False, f"Missing required risk parameter: {field}"
    
    # Validate max position size
    valid, error = validate_percentage(risk_params["max_position_size"], 0.1, 50.0)
    if not valid:
        return False, f"Invalid max_position_size: {error}"
    
    # Validate max daily loss
    valid, error = validate_percentage(risk_params["max_daily_loss"], 0.1, 20.0)
    if not valid:
        return False, f"Invalid max_daily_loss: {error}"
    
    # Validate stop loss percentage
    valid, error = validate_percentage(risk_params["stop_loss_percentage"], 0.1, 10.0)
    if not valid:
        return False, f"Invalid stop_loss_percentage: {error}"
    
    return True, None


def validate_portfolio_allocation(allocations: Dict[str, float]) -> Tuple[bool, Optional[str]]:
    """Validate portfolio allocation percentages"""
    if not allocations or not isinstance(allocations, dict):
        return False, "Allocations must be a non-empty dictionary"
    
    total_allocation = sum(allocations.values())
    
    if abs(total_allocation - 100.0) > 0.01:  # Allow small rounding errors
        return False, f"Total allocation must sum to 100%, got {total_allocation}%"
    
    for asset, allocation in allocations.items():
        valid, error = validate_percentage(allocation, 0.0, 100.0)
        if not valid:
            return False, f"Invalid allocation for {asset}: {error}"
    
    return True, None


def validate_api_credentials(credentials: Dict[str, str]) -> Tuple[bool, Optional[str]]:
    """Validate API credentials"""
    required_fields = ["api_key", "api_secret"]
    
    for field in required_fields:
        if field not in credentials or not credentials[field]:
            return False, f"Missing or empty required credential: {field}"
    
    # Basic validation - ensure they're not obviously invalid
    api_key = credentials["api_key"]
    api_secret = credentials["api_secret"]
    
    if len(api_key) < 10:
        return False, "API key appears to be too short"
    
    if len(api_secret) < 10:
        return False, "API secret appears to be too short"
    
    return True, None


def sanitize_input(value: str, max_length: int = 1000) -> str:
    """Sanitize string input"""
    if not isinstance(value, str):
        value = str(value)
    
    # Remove potentially harmful characters
    value = re.sub(r'[<>"\']', '', value)
    
    # Truncate if too long
    if len(value) > max_length:
        value = value[:max_length]
    
    return value.strip()