"""
Production-grade rate limiting with multiple algorithms and distributed support
Protects against DoS attacks and ensures fair resource usage
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aioredis
import redis

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""

    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_LOG = "sliding_log"


class RateLimitScope(Enum):
    """Rate limit scopes"""

    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    SERVICE = "service"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""

    name: str
    scope: RateLimitScope
    algorithm: RateLimitAlgorithm
    max_requests: int
    window_seconds: int
    burst_allowance: int = 0
    penalty_seconds: int = 0


@dataclass
class RateLimitResult:
    """Rate limit check result"""

    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    reason: Optional[str] = None


class RateLimitError(Exception):
    """Rate limit exceeded"""

    pass


class TokenBucket:
    """Token bucket algorithm implementation"""

    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens"""
        now = time.time()

        # Refill tokens based on elapsed time
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def get_remaining(self) -> int:
        """Get remaining tokens"""
        now = time.time()
        elapsed = now - self.last_refill
        return min(self.max_tokens, self.tokens + elapsed * self.refill_rate)


class SlidingWindow:
    """Sliding window algorithm implementation"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()

    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        now = time.time()

        # Remove old requests outside the window
        while self.requests and self.requests[0] <= now - self.window_seconds:
            self.requests.popleft()

        # Check if we're under the limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True

        return False

    def get_remaining(self) -> int:
        """Get remaining requests in window"""
        now = time.time()

        # Clean old requests
        while self.requests and self.requests[0] <= now - self.window_seconds:
            self.requests.popleft()

        return max(0, self.max_requests - len(self.requests))

    def reset_time(self) -> datetime:
        """Get time when oldest request expires"""
        if self.requests:
            oldest_request = self.requests[0]
            reset_timestamp = oldest_request + self.window_seconds
            return datetime.fromtimestamp(reset_timestamp)
        return datetime.now()


class FixedWindow:
    """Fixed window algorithm implementation"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.current_window = 0
        self.request_count = 0

    def _get_current_window(self) -> int:
        """Get current window timestamp"""
        return int(time.time() // self.window_seconds)

    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        current_window = self._get_current_window()

        # Reset counter if we're in a new window
        if current_window != self.current_window:
            self.current_window = current_window
            self.request_count = 0

        # Check if we're under the limit
        if self.request_count < self.max_requests:
            self.request_count += 1
            return True

        return False

    def get_remaining(self) -> int:
        """Get remaining requests in current window"""
        current_window = self._get_current_window()

        if current_window != self.current_window:
            return self.max_requests

        return max(0, self.max_requests - self.request_count)

    def reset_time(self) -> datetime:
        """Get time when current window resets"""
        next_window = (self.current_window + 1) * self.window_seconds
        return datetime.fromtimestamp(next_window)


class DistributedRateLimiter:
    """Distributed rate limiter using Redis"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.local_fallback = {}

    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis for distributed rate limiting")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using local fallback.")
            self.redis_client = None

    async def check_rate_limit(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Check rate limit for key"""
        if self.redis_client:
            return await self._check_redis_rate_limit(key, rule)
        else:
            return await self._check_local_rate_limit(key, rule)

    async def _check_redis_rate_limit(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Check rate limit using Redis"""
        if rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self._redis_sliding_window(key, rule)
        elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return await self._redis_fixed_window(key, rule)
        elif rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._redis_token_bucket(key, rule)
        else:
            # Fallback to local
            return await self._check_local_rate_limit(key, rule)

    async def _redis_sliding_window(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Sliding window using Redis sorted sets"""
        now = time.time()
        window_start = now - rule.window_seconds

        pipe = self.redis_client.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current requests
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(now): now})

        # Set expiration
        pipe.expire(key, rule.window_seconds)

        results = await pipe.execute()
        current_count = results[1]

        allowed = current_count < rule.max_requests
        remaining = max(0, rule.max_requests - current_count - 1)

        # Calculate reset time (when oldest request expires)
        oldest_scores = await self.redis_client.zrange(key, 0, 0, withscores=True)
        if oldest_scores:
            oldest_time = oldest_scores[0][1]
            reset_time = datetime.fromtimestamp(oldest_time + rule.window_seconds)
        else:
            reset_time = datetime.now() + timedelta(seconds=rule.window_seconds)

        if not allowed:
            # Remove the request we just added since it's not allowed
            await self.redis_client.zrem(key, str(now))

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=rule.window_seconds if not allowed else None,
        )

    async def _redis_fixed_window(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Fixed window using Redis"""
        current_window = int(time.time() // rule.window_seconds)
        window_key = f"{key}:{current_window}"

        pipe = self.redis_client.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, rule.window_seconds)

        results = await pipe.execute()
        current_count = results[0]

        allowed = current_count <= rule.max_requests
        remaining = max(0, rule.max_requests - current_count)

        next_window = (current_window + 1) * rule.window_seconds
        reset_time = datetime.fromtimestamp(next_window)

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=int(next_window - time.time()) if not allowed else None,
        )

    async def _redis_token_bucket(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Token bucket using Redis hash"""
        now = time.time()
        refill_rate = rule.max_requests / rule.window_seconds

        # Lua script for atomic token bucket operation
        lua_script = """
        local key = KEYS[1]
        local max_tokens = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local tokens_requested = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or max_tokens
        local last_refill = tonumber(bucket[2]) or now
        
        -- Refill tokens
        local elapsed = now - last_refill
        tokens = math.min(max_tokens, tokens + elapsed * refill_rate)
        
        -- Check if we can consume tokens
        local allowed = tokens >= tokens_requested
        if allowed then
            tokens = tokens - tokens_requested
        end
        
        -- Update bucket
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
        
        return {allowed and 1 or 0, math.floor(tokens)}
        """

        result = await self.redis_client.eval(
            lua_script, 1, key, rule.max_requests, refill_rate, now, 1
        )

        allowed = bool(result[0])
        remaining = result[1]

        # Calculate reset time (approximate)
        if remaining == 0:
            seconds_to_refill = 1 / refill_rate
            reset_time = datetime.now() + timedelta(seconds=seconds_to_refill)
        else:
            reset_time = datetime.now()

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=int(1 / refill_rate) if not allowed else None,
        )

    async def _check_local_rate_limit(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Local rate limiting fallback"""
        if key not in self.local_fallback:
            if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                refill_rate = rule.max_requests / rule.window_seconds
                self.local_fallback[key] = TokenBucket(rule.max_requests, refill_rate)
            elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                self.local_fallback[key] = SlidingWindow(rule.max_requests, rule.window_seconds)
            elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                self.local_fallback[key] = FixedWindow(rule.max_requests, rule.window_seconds)

        limiter = self.local_fallback[key]

        if isinstance(limiter, TokenBucket):
            allowed = limiter.consume()
            remaining = int(limiter.get_remaining())
            reset_time = datetime.now()  # Token bucket doesn't have fixed reset
        else:
            allowed = limiter.is_allowed()
            remaining = limiter.get_remaining()
            reset_time = limiter.reset_time()

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=rule.window_seconds if not allowed else None,
        )


class RateLimitManager:
    """Main rate limit manager"""

    def __init__(self, redis_url: str = None):
        self.rules: Dict[str, RateLimitRule] = {}
        self.penalties: Dict[str, datetime] = {}

        # Initialize distributed limiter
        if redis_url:
            self.limiter = DistributedRateLimiter(redis_url)
        else:
            self.limiter = DistributedRateLimiter()

    async def initialize(self):
        """Initialize rate limiter"""
        await self.limiter.initialize()
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default rate limiting rules"""
        # Global API limits
        self.add_rule(
            RateLimitRule(
                name="global_api",
                scope=RateLimitScope.GLOBAL,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                max_requests=10000,
                window_seconds=3600,  # 10k requests per hour globally
            )
        )

        # Per-user limits
        self.add_rule(
            RateLimitRule(
                name="user_api",
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                max_requests=100,
                window_seconds=60,  # 100 requests per minute per user
                burst_allowance=10,
            )
        )

        # Per-IP limits
        self.add_rule(
            RateLimitRule(
                name="ip_api",
                scope=RateLimitScope.IP,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                max_requests=1000,
                window_seconds=3600,  # 1k requests per hour per IP
                penalty_seconds=300,  # 5 minute penalty for violations
            )
        )

        # API key limits
        self.add_rule(
            RateLimitRule(
                name="api_key",
                scope=RateLimitScope.API_KEY,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                max_requests=500,
                window_seconds=60,  # 500 requests per minute per API key
            )
        )

        # Message sending limits
        self.add_rule(
            RateLimitRule(
                name="message_send",
                scope=RateLimitScope.SERVICE,
                algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
                max_requests=50,
                window_seconds=60,  # 50 messages per minute per service
            )
        )

    def add_rule(self, rule: RateLimitRule):
        """Add rate limiting rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added rate limit rule: {rule.name}")

    def get_rule(self, rule_name: str) -> Optional[RateLimitRule]:
        """Get rate limiting rule by name"""
        return self.rules.get(rule_name)

    async def check_rate_limit(
        self, rule_name: str, identifier: str, request_weight: int = 1
    ) -> RateLimitResult:
        """Check rate limit for identifier"""
        rule = self.rules.get(rule_name)
        if not rule:
            # No rule means no limit
            return RateLimitResult(allowed=True, remaining=float("inf"), reset_time=datetime.now())

        # Check if identifier is under penalty
        penalty_key = f"{rule_name}:{identifier}"
        if penalty_key in self.penalties:
            if datetime.now() < self.penalties[penalty_key]:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=self.penalties[penalty_key],
                    retry_after=int((self.penalties[penalty_key] - datetime.now()).total_seconds()),
                    reason="Under penalty",
                )
            else:
                # Penalty expired
                del self.penalties[penalty_key]

        # Create rate limit key
        rate_limit_key = f"rate_limit:{rule.scope.value}:{rule_name}:{identifier}"

        # Check rate limit
        result = await self.limiter.check_rate_limit(rate_limit_key, rule)

        # Apply penalty if violated and rule has penalty
        if not result.allowed and rule.penalty_seconds > 0:
            penalty_until = datetime.now() + timedelta(seconds=rule.penalty_seconds)
            self.penalties[penalty_key] = penalty_until
            result.retry_after = rule.penalty_seconds
            result.reason = f"Rate limit exceeded, penalty applied"

        return result

    async def is_allowed(self, rule_name: str, identifier: str) -> bool:
        """Simple check if request is allowed"""
        result = await self.check_rate_limit(rule_name, identifier)
        return result.allowed

    def create_identifier(
        self,
        scope: RateLimitScope,
        user_id: str = None,
        ip_address: str = None,
        api_key: str = None,
        service_name: str = None,
    ) -> str:
        """Create identifier for rate limiting"""
        if scope == RateLimitScope.GLOBAL:
            return "global"
        elif scope == RateLimitScope.USER and user_id:
            return f"user:{user_id}"
        elif scope == RateLimitScope.IP and ip_address:
            return f"ip:{ip_address}"
        elif scope == RateLimitScope.API_KEY and api_key:
            # Hash API key for privacy
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return f"api_key:{api_key_hash}"
        elif scope == RateLimitScope.SERVICE and service_name:
            return f"service:{service_name}"
        else:
            return "unknown"


# Global rate limiter instance
rate_limiter = RateLimitManager()


async def check_rate_limit(rule_name: str, identifier: str):
    """Decorator for rate limiting"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await rate_limiter.check_rate_limit(rule_name, identifier)

            if not result.allowed:
                raise RateLimitError(f"Rate limit exceeded: {result.reason}")

            # Add rate limit info to response headers
            kwargs["rate_limit_info"] = {
                "remaining": result.remaining,
                "reset_time": result.reset_time.isoformat(),
                "retry_after": result.retry_after,
            }

            return await func(*args, **kwargs)

        return wrapper

    return decorator
