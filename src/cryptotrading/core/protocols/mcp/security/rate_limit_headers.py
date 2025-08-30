"""
Rate Limit Response Headers for MCP Security

This module provides standard rate limit headers in HTTP responses to help
clients understand and respect rate limits, following RFC 6585 and industry best practices.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class RateLimitInfo:
    """Rate limit information for response headers"""

    limit: int  # Maximum requests allowed in window
    remaining: int  # Requests remaining in current window
    reset: int  # Unix timestamp when window resets
    window: int  # Window duration in seconds
    policy: str = "sliding"  # Rate limit policy (sliding, fixed, token-bucket)
    retry_after: Optional[int] = None  # Seconds to wait before retry (when exceeded)

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers following RFC standards"""
        headers = {
            # Standard rate limit headers (RFC draft)
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(self.reset),
            "X-RateLimit-Window": str(self.window),
            "X-RateLimit-Policy": self.policy,
            # GitHub-style headers (widely adopted)
            "X-Rate-Limit-Limit": str(self.limit),
            "X-Rate-Limit-Remaining": str(max(0, self.remaining)),
            "X-Rate-Limit-Reset": str(self.reset),
            # Additional headers for client guidance
            "X-Rate-Limit-Used": str(max(0, self.limit - self.remaining)),
        }

        # Add retry-after header when rate limited
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
            headers["X-Rate-Limit-Retry-After"] = str(self.retry_after)

        return headers


class RateLimitHeaderManager:
    """Manages rate limit headers for MCP responses"""

    def __init__(self):
        """Initialize rate limit header manager"""
        self.default_window = 60  # 1 minute default window

    def create_rate_limit_info(
        self,
        limit: int,
        used: int,
        window_start: float,
        window_duration: int = 60,
        policy: str = "sliding",
    ) -> RateLimitInfo:
        """
        Create rate limit info from usage data

        Args:
            limit: Maximum requests allowed in window
            used: Number of requests used in current window
            window_start: Unix timestamp when current window started
            window_duration: Window duration in seconds
            policy: Rate limiting policy

        Returns:
            RateLimitInfo with calculated values
        """
        now = time.time()
        remaining = max(0, limit - used)

        # Calculate when the window resets
        if policy == "sliding":
            # For sliding window, reset is when oldest request expires
            reset = int(window_start + window_duration)
        else:
            # For fixed window, reset is at next window boundary
            reset = int(((now // window_duration) + 1) * window_duration)

        # Calculate retry-after if rate limited
        retry_after = None
        if remaining == 0:
            retry_after = max(1, reset - int(now))

        return RateLimitInfo(
            limit=limit,
            remaining=remaining,
            reset=reset,
            window=window_duration,
            policy=policy,
            retry_after=retry_after,
        )

    def create_from_rate_limiter_status(self, status: Dict[str, Any]) -> RateLimitInfo:
        """
        Create rate limit info from rate limiter status

        Args:
            status: Status dictionary from rate limiter

        Returns:
            RateLimitInfo object
        """
        # Extract values with sensible defaults
        limit = status.get("limit", 100)
        used = status.get("used", 0)
        remaining = status.get("remaining", limit)
        reset_time = status.get("reset_time", time.time() + 60)
        window = status.get("window", self.default_window)
        policy = status.get("policy", "sliding")

        # Calculate retry-after if exceeded
        retry_after = None
        if remaining == 0:
            retry_after = max(1, int(reset_time - time.time()))

        return RateLimitInfo(
            limit=limit,
            remaining=remaining,
            reset=int(reset_time),
            window=window,
            policy=policy,
            retry_after=retry_after,
        )

    def add_headers_to_response(
        self, response_dict: Dict[str, Any], rate_limit_info: RateLimitInfo
    ) -> Dict[str, Any]:
        """
        Add rate limit headers to response dictionary

        Args:
            response_dict: Response dictionary to modify
            rate_limit_info: Rate limit information

        Returns:
            Modified response dictionary with headers
        """
        # Add headers section if it doesn't exist
        if "headers" not in response_dict:
            response_dict["headers"] = {}

        # Add rate limit headers
        response_dict["headers"].update(rate_limit_info.to_headers())

        return response_dict

    def create_rate_limit_exceeded_response(
        self, rate_limit_info: RateLimitInfo, error_message: str = "Rate limit exceeded"
    ) -> Dict[str, Any]:
        """
        Create a complete rate limit exceeded response

        Args:
            rate_limit_info: Rate limit information
            error_message: Error message to include

        Returns:
            Complete error response with headers
        """
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32003,  # Custom MCP error code for rate limiting
                "message": error_message,
                "data": {
                    "type": "rate_limit_exceeded",
                    "limit": rate_limit_info.limit,
                    "window": rate_limit_info.window,
                    "retry_after": rate_limit_info.retry_after,
                    "policy": rate_limit_info.policy,
                },
            },
            "headers": rate_limit_info.to_headers(),
        }

        return response

    def get_rate_limit_guidance(self, rate_limit_info: RateLimitInfo) -> Dict[str, Any]:
        """
        Get client guidance for rate limit handling

        Args:
            rate_limit_info: Rate limit information

        Returns:
            Dictionary with guidance information
        """
        guidance = {
            "current_usage": {
                "limit": rate_limit_info.limit,
                "used": rate_limit_info.limit - rate_limit_info.remaining,
                "remaining": rate_limit_info.remaining,
                "percentage_used": (
                    (rate_limit_info.limit - rate_limit_info.remaining) / rate_limit_info.limit
                )
                * 100,
            },
            "timing": {
                "window_duration": rate_limit_info.window,
                "reset_time": rate_limit_info.reset,
                "reset_in_seconds": max(0, rate_limit_info.reset - int(time.time())),
                "policy": rate_limit_info.policy,
            },
            "recommendations": [],
        }

        # Add recommendations based on usage
        usage_percentage = guidance["current_usage"]["percentage_used"]

        if usage_percentage >= 90:
            guidance["recommendations"].append(
                "Critical: Very close to rate limit. Slow down requests immediately."
            )
        elif usage_percentage >= 75:
            guidance["recommendations"].append(
                "Warning: Approaching rate limit. Consider reducing request frequency."
            )
        elif usage_percentage >= 50:
            guidance["recommendations"].append(
                "Info: Half of rate limit used. Monitor request frequency."
            )

        if rate_limit_info.retry_after:
            guidance["recommendations"].append(
                f"Rate limited: Wait {rate_limit_info.retry_after} seconds before retry."
            )

        # Add general best practices
        guidance["best_practices"] = [
            "Implement exponential backoff for retries",
            "Cache responses when possible to reduce API calls",
            "Monitor rate limit headers in responses",
            "Distribute requests evenly across the time window",
            "Use batch operations when available",
        ]

        return guidance


class MultiTierRateLimitManager:
    """Manages multiple rate limit tiers with appropriate headers"""

    def __init__(self):
        """Initialize multi-tier rate limit manager"""
        self.header_manager = RateLimitHeaderManager()
        self.tier_configs = {
            "global": {"limit": 1000, "window": 60, "priority": 1},
            "user": {"limit": 100, "window": 60, "priority": 2},
            "method": {"limit": 50, "window": 60, "priority": 3},
            "ip": {"limit": 200, "window": 60, "priority": 4},
        }

    def get_most_restrictive_limit(
        self, rate_limit_statuses: Dict[str, Dict[str, Any]]
    ) -> RateLimitInfo:
        """
        Get the most restrictive rate limit from multiple tiers

        Args:
            rate_limit_statuses: Dictionary of tier_name -> status

        Returns:
            RateLimitInfo for the most restrictive limit
        """
        most_restrictive = None
        lowest_remaining_ratio = float("inf")

        for tier_name, status in rate_limit_statuses.items():
            if not status:
                continue

            limit = status.get("limit", 100)
            remaining = status.get("remaining", limit)

            # Calculate remaining ratio (lower is more restrictive)
            remaining_ratio = remaining / limit if limit > 0 else 0

            if remaining_ratio < lowest_remaining_ratio:
                lowest_remaining_ratio = remaining_ratio
                most_restrictive = self.header_manager.create_from_rate_limiter_status(status)
                # Add tier information
                most_restrictive.policy = f"{tier_name}-{most_restrictive.policy}"

        # Default fallback
        if most_restrictive is None:
            most_restrictive = RateLimitInfo(
                limit=100, remaining=100, reset=int(time.time() + 60), window=60, policy="default"
            )

        return most_restrictive

    def create_composite_headers(
        self, rate_limit_statuses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Create composite rate limit headers from multiple tiers

        Args:
            rate_limit_statuses: Dictionary of tier_name -> status

        Returns:
            Dictionary of headers
        """
        # Get most restrictive limit for primary headers
        most_restrictive = self.get_most_restrictive_limit(rate_limit_statuses)
        headers = most_restrictive.to_headers()

        # Add tier-specific headers
        for tier_name, status in rate_limit_statuses.items():
            if not status:
                continue

            tier_info = self.header_manager.create_from_rate_limiter_status(status)
            tier_prefix = f"X-RateLimit-{tier_name.title()}"

            headers.update(
                {
                    f"{tier_prefix}-Limit": str(tier_info.limit),
                    f"{tier_prefix}-Remaining": str(tier_info.remaining),
                    f"{tier_prefix}-Reset": str(tier_info.reset),
                }
            )

        return headers


# Utility functions
def calculate_reset_time(window_start: float, window_duration: int, policy: str = "sliding") -> int:
    """Calculate when the rate limit window resets"""
    now = time.time()

    if policy == "sliding":
        return int(window_start + window_duration)
    else:  # fixed window
        return int(((now // window_duration) + 1) * window_duration)


def calculate_retry_after(reset_time: int, buffer_seconds: int = 1) -> int:
    """Calculate retry-after value with buffer"""
    return max(buffer_seconds, reset_time - int(time.time()))


def is_rate_limited(remaining: int) -> bool:
    """Check if rate limit is exceeded"""
    return remaining <= 0


def get_usage_percentage(limit: int, used: int) -> float:
    """Calculate usage percentage"""
    return (used / limit) * 100 if limit > 0 else 100.0


# Global instances
_header_manager = RateLimitHeaderManager()
_multi_tier_manager = MultiTierRateLimitManager()


def get_rate_limit_header_manager() -> RateLimitHeaderManager:
    """Get global rate limit header manager"""
    return _header_manager


def get_multi_tier_manager() -> MultiTierRateLimitManager:
    """Get global multi-tier rate limit manager"""
    return _multi_tier_manager
