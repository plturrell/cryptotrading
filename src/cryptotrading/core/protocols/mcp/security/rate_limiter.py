"""
Rate Limiting System for MCP using Vercel Edge Config

This module provides comprehensive rate limiting for MCP servers deployed on Vercel.
Uses Vercel Edge Config for fast, globally distributed rate limit storage.

Features:
- Multiple rate limiting algorithms (sliding window, token bucket)
- Vercel Edge Config integration for global state
- Per-user, per-API-key, and global rate limits
- Configurable time windows and burst handling
- Automatic cleanup of expired entries
- Integration with MCP authentication system
"""

import asyncio
import time
import json
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"


class RateLimitExceeded(Exception):
    """Rate limit exceeded exception"""
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    name: str
    requests_per_window: int
    window_seconds: int
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    burst_limit: Optional[int] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.burst_limit is None:
            self.burst_limit = self.requests_per_window * 2


@dataclass
class RateLimitState:
    """Rate limit state for tracking"""
    requests: List[float]  # Timestamps of requests
    tokens: float  # For token bucket algorithm
    last_refill: float  # Last token refill time
    window_start: float  # For fixed window algorithm
    
    def cleanup_old_requests(self, window_seconds: int):
        """Remove requests older than window"""
        cutoff = time.time() - window_seconds
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]


class VercelEdgeConfigClient:
    """Client for Vercel Edge Config API"""
    
    def __init__(self, edge_config_id: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize Edge Config client
        
        Args:
            edge_config_id: Edge Config ID (from EDGE_CONFIG env var)
            token: Vercel token (from VERCEL_TOKEN env var)
        """
        self.edge_config_id = edge_config_id or os.getenv("EDGE_CONFIG")
        self.token = token or os.getenv("VERCEL_TOKEN")
        
        if not self.edge_config_id:
            logger.warning("EDGE_CONFIG not set, falling back to local storage")
            self.use_local = True
            self.local_storage: Dict[str, Any] = {}
        else:
            self.use_local = False
            
        self.base_url = f"https://edge-config.vercel.com/{self.edge_config_id}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Edge Config"""
        if self.use_local:
            return self.local_storage.get(key)
        
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/item/{key}", headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        return None
                    else:
                        logger.error(f"Edge Config get failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Edge Config get error: {e}")
            return None
    
    async def set(self, key: str, value: Any) -> bool:
        """Set value in Edge Config"""
        if self.use_local:
            self.local_storage[key] = value
            return True
        
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            # Edge Config uses PATCH for updates
            data = {
                "items": [
                    {
                        "operation": "upsert",
                        "key": key,
                        "value": value
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.patch(f"{self.base_url}/items", 
                                       headers=headers, 
                                       json=data) as response:
                    if response.status in [200, 201]:
                        return True
                    else:
                        logger.error(f"Edge Config set failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Edge Config set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Edge Config"""
        if self.use_local:
            self.local_storage.pop(key, None)
            return True
        
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "items": [
                    {
                        "operation": "delete",
                        "key": key
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.patch(f"{self.base_url}/items", 
                                       headers=headers, 
                                       json=data) as response:
                    return response.status in [200, 204]
                    
        except Exception as e:
            logger.error(f"Edge Config delete error: {e}")
            return False


class RateLimiter:
    """Advanced rate limiter with multiple algorithms"""
    
    def __init__(self, edge_config: VercelEdgeConfigClient):
        """
        Initialize rate limiter
        
        Args:
            edge_config: Vercel Edge Config client
        """
        self.edge_config = edge_config
        self.configs: Dict[str, RateLimitConfig] = {}
        self.local_cache: Dict[str, RateLimitState] = {}
        self.cache_ttl = 60  # seconds
        self.last_cleanup = time.time()
        
        # Load default configurations
        self._load_default_configs()
    
    def _load_default_configs(self):
        """Load default rate limit configurations"""
        # Global rate limits
        self.add_config(RateLimitConfig(
            name="global",
            requests_per_window=1000,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW
        ))
        
        # Per-user rate limits
        self.add_config(RateLimitConfig(
            name="per_user",
            requests_per_window=100,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW
        ))
        
        # Per-API-key rate limits
        self.add_config(RateLimitConfig(
            name="per_api_key",
            requests_per_window=200,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_limit=300
        ))
        
        # Expensive operations (tool calls)
        self.add_config(RateLimitConfig(
            name="tool_calls",
            requests_per_window=50,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW
        ))
        
        # Load custom configs from environment
        self._load_env_configs()
    
    def _load_env_configs(self):
        """Load rate limit configs from environment variables"""
        config_json = os.getenv("MCP_RATE_LIMITS")
        if config_json:
            try:
                configs = json.loads(config_json)
                for config_data in configs:
                    config = RateLimitConfig(
                        name=config_data["name"],
                        requests_per_window=config_data["requests_per_window"],
                        window_seconds=config_data["window_seconds"],
                        algorithm=RateLimitAlgorithm(config_data.get("algorithm", "sliding_window")),
                        burst_limit=config_data.get("burst_limit"),
                        enabled=config_data.get("enabled", True)
                    )
                    self.add_config(config)
            except Exception as e:
                logger.error(f"Failed to load rate limit configs from env: {e}")
    
    def add_config(self, config: RateLimitConfig):
        """Add rate limit configuration"""
        self.configs[config.name] = config
        logger.info(f"Added rate limit config: {config.name} - {config.requests_per_window}/{config.window_seconds}s")
    
    def _get_cache_key(self, identifier: str, config_name: str) -> str:
        """Generate cache key for rate limit state"""
        # Hash long identifiers for consistent key length
        if len(identifier) > 50:
            identifier = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        return f"rate_limit:{config_name}:{identifier}"
    
    async def _get_state(self, cache_key: str) -> RateLimitState:
        """Get rate limit state from cache or Edge Config"""
        # Check local cache first
        if cache_key in self.local_cache:
            state = self.local_cache[cache_key]
            return state
        
        # Try Edge Config
        try:
            data = await self.edge_config.get(cache_key)
            if data:
                state = RateLimitState(
                    requests=data.get("requests", []),
                    tokens=data.get("tokens", 0),
                    last_refill=data.get("last_refill", time.time()),
                    window_start=data.get("window_start", time.time())
                )
            else:
                # Create new state
                state = RateLimitState(
                    requests=[],
                    tokens=0,
                    last_refill=time.time(),
                    window_start=time.time()
                )
        except Exception as e:
            logger.error(f"Failed to get rate limit state: {e}")
            # Fallback to new state
            state = RateLimitState(
                requests=[],
                tokens=0,
                last_refill=time.time(),
                window_start=time.time()
            )
        
        # Cache locally
        self.local_cache[cache_key] = state
        return state
    
    async def _save_state(self, cache_key: str, state: RateLimitState):
        """Save rate limit state to cache and Edge Config"""
        # Update local cache
        self.local_cache[cache_key] = state
        
        # Save to Edge Config
        try:
            data = {
                "requests": state.requests,
                "tokens": state.tokens,
                "last_refill": state.last_refill,
                "window_start": state.window_start,
                "updated_at": time.time()
            }
            await self.edge_config.set(cache_key, data)
        except Exception as e:
            logger.error(f"Failed to save rate limit state: {e}")
    
    async def check_rate_limit(self, identifier: str, config_name: str = "global") -> Tuple[bool, int]:
        """
        Check if request is within rate limit
        
        Args:
            identifier: Unique identifier (user_id, api_key, IP, etc.)
            config_name: Rate limit configuration to use
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        if config_name not in self.configs:
            logger.warning(f"Rate limit config '{config_name}' not found, allowing request")
            return True, 0
        
        config = self.configs[config_name]
        if not config.enabled:
            return True, 0
        
        cache_key = self._get_cache_key(identifier, config_name)
        state = await self._get_state(cache_key)
        
        now = time.time()
        
        if config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            allowed, retry_after = self._check_sliding_window(state, config, now)
        elif config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            allowed, retry_after = self._check_token_bucket(state, config, now)
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            allowed, retry_after = self._check_fixed_window(state, config, now)
        else:
            logger.error(f"Unknown algorithm: {config.algorithm}")
            return True, 0
        
        if allowed:
            # Record this request
            if config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                state.requests.append(now)
            elif config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                state.tokens -= 1
            elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                state.requests.append(now)
        
        # Save updated state
        await self._save_state(cache_key, state)
        
        return allowed, retry_after
    
    def _check_sliding_window(self, state: RateLimitState, config: RateLimitConfig, now: float) -> Tuple[bool, int]:
        """Check sliding window rate limit"""
        # Clean old requests
        state.cleanup_old_requests(config.window_seconds)
        
        # Check if under limit
        if len(state.requests) < config.requests_per_window:
            return True, 0
        
        # Calculate retry after
        oldest_request = min(state.requests) if state.requests else now
        retry_after = int(oldest_request + config.window_seconds - now) + 1
        return False, max(retry_after, 1)
    
    def _check_token_bucket(self, state: RateLimitState, config: RateLimitConfig, now: float) -> Tuple[bool, int]:
        """Check token bucket rate limit"""
        # Refill tokens
        time_passed = now - state.last_refill
        tokens_to_add = time_passed * (config.requests_per_window / config.window_seconds)
        state.tokens = min(config.burst_limit, state.tokens + tokens_to_add)
        state.last_refill = now
        
        # Check if token available
        if state.tokens >= 1:
            return True, 0
        
        # Calculate retry after
        tokens_needed = 1 - state.tokens
        retry_after = int(tokens_needed / (config.requests_per_window / config.window_seconds)) + 1
        return False, retry_after
    
    def _check_fixed_window(self, state: RateLimitState, config: RateLimitConfig, now: float) -> Tuple[bool, int]:
        """Check fixed window rate limit"""
        # Check if we need a new window
        if now - state.window_start >= config.window_seconds:
            state.window_start = now
            state.requests = []
        
        # Check if under limit
        if len(state.requests) < config.requests_per_window:
            return True, 0
        
        # Calculate retry after (time until next window)
        retry_after = int(state.window_start + config.window_seconds - now) + 1
        return False, max(retry_after, 1)
    
    async def cleanup_expired_entries(self):
        """Clean up expired rate limit entries"""
        now = time.time()
        
        # Only cleanup every 5 minutes
        if now - self.last_cleanup < 300:
            return
        
        self.last_cleanup = now
        
        # Clean local cache
        expired_keys = []
        for key, state in self.local_cache.items():
            # Remove entries older than 1 hour
            if state.last_refill < now - 3600:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.local_cache[key]
            # Also try to delete from Edge Config
            try:
                await self.edge_config.delete(key)
            except Exception as e:
                logger.error(f"Failed to delete expired key {key}: {e}")
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired rate limit entries")
    
    async def get_rate_limit_status(self, identifier: str, config_name: str = "global") -> Dict[str, Any]:
        """
        Get current rate limit status
        
        Args:
            identifier: Unique identifier
            config_name: Rate limit configuration name
            
        Returns:
            Status information including remaining requests
        """
        if config_name not in self.configs:
            return {"error": "Config not found"}
        
        config = self.configs[config_name]
        cache_key = self._get_cache_key(identifier, config_name)
        state = await self._get_state(cache_key)
        
        now = time.time()
        
        if config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            state.cleanup_old_requests(config.window_seconds)
            remaining = max(0, config.requests_per_window - len(state.requests))
            reset_time = min(state.requests) + config.window_seconds if state.requests else now
            
        elif config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            # Refill tokens for accurate count
            time_passed = now - state.last_refill
            tokens_to_add = time_passed * (config.requests_per_window / config.window_seconds)
            current_tokens = min(config.burst_limit, state.tokens + tokens_to_add)
            remaining = int(current_tokens)
            reset_time = now + (config.burst_limit - current_tokens) / (config.requests_per_window / config.window_seconds)
            
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            if now - state.window_start >= config.window_seconds:
                remaining = config.requests_per_window
                reset_time = now + config.window_seconds
            else:
                remaining = max(0, config.requests_per_window - len(state.requests))
                reset_time = state.window_start + config.window_seconds
        
        return {
            "limit": config.requests_per_window,
            "remaining": remaining,
            "reset_time": reset_time,
            "window_seconds": config.window_seconds,
            "algorithm": config.algorithm.value
        }


class RateLimitMiddleware:
    """Rate limiting middleware for MCP requests"""
    
    def __init__(self, rate_limiter: RateLimiter):
        """
        Initialize rate limit middleware
        
        Args:
            rate_limiter: RateLimiter instance
        """
        self.rate_limiter = rate_limiter
    
    async def check_request_limits(self, identifier: str, method: str, 
                                  api_key_id: Optional[str] = None) -> None:
        """
        Check all applicable rate limits for a request
        
        Args:
            identifier: User/session identifier
            method: MCP method being called
            api_key_id: API key ID if using API key auth
            
        Raises:
            RateLimitExceeded: If any rate limit is exceeded
        """
        # Check global rate limit
        allowed, retry_after = await self.rate_limiter.check_rate_limit(
            "global", "global"
        )
        if not allowed:
            raise RateLimitExceeded("Global rate limit exceeded", retry_after)
        
        # Check per-user rate limit
        allowed, retry_after = await self.rate_limiter.check_rate_limit(
            identifier, "per_user"
        )
        if not allowed:
            raise RateLimitExceeded("User rate limit exceeded", retry_after)
        
        # Check API key specific limits
        if api_key_id:
            allowed, retry_after = await self.rate_limiter.check_rate_limit(
                api_key_id, "per_api_key"
            )
            if not allowed:
                raise RateLimitExceeded("API key rate limit exceeded", retry_after)
        
        # Check method-specific limits
        if method.startswith("tools/call"):
            allowed, retry_after = await self.rate_limiter.check_rate_limit(
                identifier, "tool_calls"
            )
            if not allowed:
                raise RateLimitExceeded("Tool call rate limit exceeded", retry_after)
        
        # Cleanup expired entries periodically
        await self.rate_limiter.cleanup_expired_entries()


# Convenience functions for Vercel environment
def create_edge_config_client() -> VercelEdgeConfigClient:
    """Create Edge Config client with environment variables"""
    return VercelEdgeConfigClient()


def create_rate_limiter() -> RateLimiter:
    """Create rate limiter with Edge Config"""
    edge_config = create_edge_config_client()
    return RateLimiter(edge_config)


def create_rate_limit_middleware() -> RateLimitMiddleware:
    """Create rate limit middleware"""
    rate_limiter = create_rate_limiter()
    return RateLimitMiddleware(rate_limiter)