"""
MCP Rate Limiting for Serverless Environments
Lightweight rate limiting using in-memory tracking
"""
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10


@dataclass
class RequestRecord:
    """Record of a request"""
    timestamp: float
    endpoint: str
    user_id: str


class TokenBucket:
    """Simple token bucket for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens"""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


class MCPRateLimiter:
    """Rate limiter for MCP requests"""
    
    def __init__(self):
        self.request_history: Dict[str, List[RequestRecord]] = {}
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.default_limits = RateLimit()
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def check_rate_limit(self, user_id: str, endpoint: str = "default", 
                        custom_limits: Optional[RateLimit] = None) -> Dict[str, Any]:
        """Check if request is within rate limits"""
        self._cleanup_old_records()
        
        limits = custom_limits or self.default_limits
        now = time.time()
        
        # Get or create user's request history
        user_key = f"{user_id}:{endpoint}"
        if user_key not in self.request_history:
            self.request_history[user_key] = []
        
        history = self.request_history[user_key]
        
        # Check minute-based limit
        minute_ago = now - 60
        recent_requests = [r for r in history if r.timestamp > minute_ago]
        
        if len(recent_requests) >= limits.requests_per_minute:
            return {
                "allowed": False,
                "reason": "Rate limit exceeded: too many requests per minute",
                "retry_after": 60 - (now - recent_requests[0].timestamp),
                "limit_type": "per_minute"
            }
        
        # Check hour-based limit
        hour_ago = now - 3600
        hourly_requests = [r for r in history if r.timestamp > hour_ago]
        
        if len(hourly_requests) >= limits.requests_per_hour:
            return {
                "allowed": False,
                "reason": "Rate limit exceeded: too many requests per hour",
                "retry_after": 3600 - (now - hourly_requests[0].timestamp),
                "limit_type": "per_hour"
            }
        
        # Check burst limit using token bucket
        bucket_key = f"burst:{user_key}"
        if bucket_key not in self.token_buckets:
            # Create token bucket: burst_limit capacity, refill at 1 token per 6 seconds
            self.token_buckets[bucket_key] = TokenBucket(
                capacity=limits.burst_limit,
                refill_rate=1.0 / 6.0  # 10 requests per minute
            )
        
        bucket = self.token_buckets[bucket_key]
        if not bucket.consume():
            return {
                "allowed": False,
                "reason": "Rate limit exceeded: burst limit reached",
                "retry_after": 6,  # Wait 6 seconds for next token
                "limit_type": "burst"
            }
        
        # Record the request
        history.append(RequestRecord(
            timestamp=now,
            endpoint=endpoint,
            user_id=user_id
        ))
        
        return {
            "allowed": True,
            "remaining_minute": limits.requests_per_minute - len(recent_requests) - 1,
            "remaining_hour": limits.requests_per_hour - len(hourly_requests) - 1,
            "remaining_burst": int(bucket.tokens)
        }
    
    def _cleanup_old_records(self):
        """Clean up old request records"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        # Remove records older than 1 hour
        cutoff = now - 3600
        
        for user_key in list(self.request_history.keys()):
            history = self.request_history[user_key]
            self.request_history[user_key] = [
                r for r in history if r.timestamp > cutoff
            ]
            
            # Remove empty histories
            if not self.request_history[user_key]:
                del self.request_history[user_key]
        
        # Clean up old token buckets
        active_users = set(self.request_history.keys())
        bucket_keys_to_remove = [
            key for key in self.token_buckets.keys()
            if key.replace("burst:", "") not in active_users
        ]
        
        for key in bucket_keys_to_remove:
            del self.token_buckets[key]
        
        self.last_cleanup = now
        logger.debug("Cleaned up old rate limit records")
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get rate limit statistics for a user"""
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600
        
        total_minute = 0
        total_hour = 0
        endpoints = set()
        
        for user_key, history in self.request_history.items():
            if user_key.startswith(f"{user_id}:"):
                endpoint = user_key.split(":", 1)[1]
                endpoints.add(endpoint)
                
                minute_requests = sum(1 for r in history if r.timestamp > minute_ago)
                hour_requests = sum(1 for r in history if r.timestamp > hour_ago)
                
                total_minute += minute_requests
                total_hour += hour_requests
        
        return {
            "user_id": user_id,
            "requests_last_minute": total_minute,
            "requests_last_hour": total_hour,
            "active_endpoints": list(endpoints),
            "limits": {
                "per_minute": self.default_limits.requests_per_minute,
                "per_hour": self.default_limits.requests_per_hour,
                "burst": self.default_limits.burst_limit
            }
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics"""
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600
        
        total_users = len(set(
            key.split(":", 1)[0] for key in self.request_history.keys()
        ))
        
        total_requests_minute = 0
        total_requests_hour = 0
        
        for history in self.request_history.values():
            total_requests_minute += sum(1 for r in history if r.timestamp > minute_ago)
            total_requests_hour += sum(1 for r in history if r.timestamp > hour_ago)
        
        return {
            "active_users": total_users,
            "total_requests_last_minute": total_requests_minute,
            "total_requests_last_hour": total_requests_hour,
            "tracked_endpoints": len(self.request_history),
            "token_buckets": len(self.token_buckets)
        }


class RateLimitMiddleware:
    """Middleware for applying rate limits to MCP requests"""
    
    def __init__(self, rate_limiter: MCPRateLimiter = None):
        self.rate_limiter = rate_limiter or MCPRateLimiter()
        
        # Different limits for different tool types
        self.tool_limits = {
            "get_portfolio": RateLimit(requests_per_minute=30, requests_per_hour=500),
            "get_market_data": RateLimit(requests_per_minute=60, requests_per_hour=1000),
            "execute_trade": RateLimit(requests_per_minute=10, requests_per_hour=100, burst_limit=3),
            "analyze_sentiment": RateLimit(requests_per_minute=20, requests_per_hour=300),
            "get_risk_metrics": RateLimit(requests_per_minute=15, requests_per_hour=200)
        }
    
    def check_request(self, user_id: str, method: str, 
                     tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Check if request should be allowed"""
        # Determine endpoint and limits
        if tool_name and tool_name in self.tool_limits:
            endpoint = f"tool:{tool_name}"
            limits = self.tool_limits[tool_name]
        else:
            endpoint = f"method:{method}"
            limits = None
        
        return self.rate_limiter.check_rate_limit(user_id, endpoint, limits)
    
    def apply_rate_limit(self, user_id: str, method: str, 
                        tool_name: Optional[str] = None):
        """Decorator to apply rate limiting"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Check rate limit
                result = self.check_request(user_id, method, tool_name)
                
                if not result["allowed"]:
                    raise RateLimitExceeded(
                        message=result["reason"],
                        retry_after=result["retry_after"],
                        limit_type=result["limit_type"]
                    )
                
                # Execute function
                return func(*args, **kwargs)
            
            return wrapper
        return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, message: str, retry_after: float, limit_type: str):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_type = limit_type
        self.message = message


# Global rate limiter instance
global_rate_limiter = MCPRateLimiter()
rate_limit_middleware = RateLimitMiddleware(global_rate_limiter)


def check_rate_limit(user_id: str, endpoint: str = "default") -> Dict[str, Any]:
    """Check rate limit for user and endpoint"""
    return global_rate_limiter.check_rate_limit(user_id, endpoint)


def get_rate_limit_stats(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Get rate limiting statistics"""
    if user_id:
        return global_rate_limiter.get_user_stats(user_id)
    else:
        return global_rate_limiter.get_global_stats()


def rate_limit(user_id: str, endpoint: str = "default"):
    """Decorator for rate limiting functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = check_rate_limit(user_id, endpoint)
            
            if not result["allowed"]:
                raise RateLimitExceeded(
                    message=result["reason"],
                    retry_after=result["retry_after"],
                    limit_type=result["limit_type"]
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
