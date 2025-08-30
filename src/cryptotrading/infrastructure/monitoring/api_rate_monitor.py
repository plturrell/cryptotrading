"""
API Rate Limit Monitoring Service
Tracks and monitors rate limits across all external API providers
"""
import asyncio
import json
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration"""

    provider: str
    endpoint: str
    limit: int
    window_seconds: int
    current_usage: int = 0
    reset_time: Optional[datetime] = None


@dataclass
class APIUsage:
    """API usage statistics"""

    provider: str
    endpoint: str
    requests_made: int
    requests_remaining: int
    reset_time: datetime
    usage_percentage: float
    status: str  # 'ok', 'warning', 'critical', 'exceeded'


class APIRateMonitor:
    """Monitors API rate limits across all providers"""

    def __init__(self):
        self.rate_limits: Dict[str, RateLimit] = {}
        self.usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts_sent: Dict[str, datetime] = {}

        # Initialize known API limits
        self._init_known_limits()

    def _init_known_limits(self):
        """Initialize known API rate limits"""
        known_limits = [
            # CoinGecko/GeckoTerminal
            RateLimit("geckoterminal", "trending_pools", 30, 60),
            RateLimit("geckoterminal", "networks", 30, 60),
            RateLimit("geckoterminal", "pools", 30, 60),
            # Web3 Providers
            RateLimit("alchemy", "eth_getBalance", 300, 60),
            RateLimit("infura", "eth_getBalance", 100, 60),
            RateLimit("cloudflare", "eth_getBalance", 100, 60),
            # Market Data
            RateLimit("coingecko", "simple_price", 50, 60),
            RateLimit("cryptocompare", "price", 100, 60),
            # DEX Subgraphs (backup)
            RateLimit("uniswap_subgraph", "pools", 1000, 60),
            RateLimit("sushiswap_subgraph", "pairs", 1000, 60),
        ]

        for limit in known_limits:
            key = f"{limit.provider}:{limit.endpoint}"
            self.rate_limits[key] = limit

    def record_request(self, provider: str, endpoint: str, response_headers: Dict[str, str] = None):
        """Record an API request and update usage tracking"""
        key = f"{provider}:{endpoint}"
        current_time = datetime.now()

        # Get or create rate limit
        if key not in self.rate_limits:
            # Create default rate limit if unknown
            self.rate_limits[key] = RateLimit(
                provider=provider,
                endpoint=endpoint,
                limit=100,  # Default conservative limit
                window_seconds=60,
                current_usage=0,
            )

        rate_limit = self.rate_limits[key]

        # Update usage from response headers if available
        if response_headers:
            self._update_from_headers(rate_limit, response_headers)

        # Increment usage
        rate_limit.current_usage += 1

        # Add to usage history
        self.usage_history[key].append(
            {
                "timestamp": current_time,
                "usage": rate_limit.current_usage,
                "limit": rate_limit.limit,
            }
        )

        # Check if we need to reset the counter
        if rate_limit.reset_time and current_time >= rate_limit.reset_time:
            rate_limit.current_usage = 1  # Reset to 1 (current request)
            rate_limit.reset_time = current_time + timedelta(seconds=rate_limit.window_seconds)
        elif not rate_limit.reset_time:
            rate_limit.reset_time = current_time + timedelta(seconds=rate_limit.window_seconds)

        # Check for alerts
        self._check_usage_alerts(key, rate_limit)

    def _update_from_headers(self, rate_limit: RateLimit, headers: Dict[str, str]):
        """Update rate limit info from response headers"""
        # Common header patterns
        header_mappings = {
            "x-ratelimit-limit": "limit",
            "x-ratelimit-remaining": "remaining",
            "x-ratelimit-reset": "reset",
            "ratelimit-limit": "limit",
            "ratelimit-remaining": "remaining",
            "ratelimit-reset": "reset",
        }

        for header_name, header_value in headers.items():
            header_lower = header_name.lower()

            if header_lower in header_mappings:
                try:
                    value = int(header_value)

                    if "limit" in header_lower:
                        rate_limit.limit = value
                    elif "remaining" in header_lower:
                        rate_limit.current_usage = rate_limit.limit - value
                    elif "reset" in header_lower:
                        # Assume Unix timestamp
                        rate_limit.reset_time = datetime.fromtimestamp(value)

                except (ValueError, TypeError):
                    continue

    def _check_usage_alerts(self, key: str, rate_limit: RateLimit):
        """Check if usage alerts should be sent"""
        usage_percentage = (rate_limit.current_usage / rate_limit.limit) * 100
        current_time = datetime.now()

        # Alert thresholds
        if usage_percentage >= 95:
            status = "critical"
        elif usage_percentage >= 80:
            status = "warning"
        else:
            return  # No alert needed

        # Check if we've already sent an alert recently (within 5 minutes)
        last_alert = self.alerts_sent.get(key)
        if last_alert and (current_time - last_alert) < timedelta(minutes=5):
            return

        # Log alert
        logger.warning(
            f"API Rate Limit Alert - {rate_limit.provider}:{rate_limit.endpoint} "
            f"at {usage_percentage:.1f}% ({rate_limit.current_usage}/{rate_limit.limit})"
        )

        self.alerts_sent[key] = current_time

    def get_usage_stats(self) -> List[APIUsage]:
        """Get current usage statistics for all APIs"""
        stats = []
        current_time = datetime.now()

        for key, rate_limit in self.rate_limits.items():
            # Calculate remaining requests
            remaining = max(0, rate_limit.limit - rate_limit.current_usage)
            usage_percentage = (rate_limit.current_usage / rate_limit.limit) * 100

            # Determine status
            if usage_percentage >= 100:
                status = "exceeded"
            elif usage_percentage >= 95:
                status = "critical"
            elif usage_percentage >= 80:
                status = "warning"
            else:
                status = "ok"

            # Reset time
            reset_time = rate_limit.reset_time or (
                current_time + timedelta(seconds=rate_limit.window_seconds)
            )

            stats.append(
                APIUsage(
                    provider=rate_limit.provider,
                    endpoint=rate_limit.endpoint,
                    requests_made=rate_limit.current_usage,
                    requests_remaining=remaining,
                    reset_time=reset_time,
                    usage_percentage=usage_percentage,
                    status=status,
                )
            )

        return sorted(stats, key=lambda x: x.usage_percentage, reverse=True)

    def get_usage_dict(self) -> Dict[str, Any]:
        """Get usage statistics as dictionary for API responses"""
        stats = self.get_usage_stats()

        return {
            "providers": [asdict(stat) for stat in stats],
            "summary": {
                "total_providers": len(stats),
                "critical_count": len([s for s in stats if s.status == "critical"]),
                "warning_count": len([s for s in stats if s.status == "warning"]),
                "exceeded_count": len([s for s in stats if s.status == "exceeded"]),
                "healthy_count": len([s for s in stats if s.status == "ok"]),
            },
            "last_updated": datetime.now().isoformat(),
            "status": "success",
        }

    def can_make_request(self, provider: str, endpoint: str) -> bool:
        """Check if a request can be made without exceeding rate limits"""
        key = f"{provider}:{endpoint}"

        if key not in self.rate_limits:
            return True  # Unknown API, allow request

        rate_limit = self.rate_limits[key]
        return rate_limit.current_usage < rate_limit.limit

    def get_wait_time(self, provider: str, endpoint: str) -> float:
        """Get recommended wait time before next request (in seconds)"""
        key = f"{provider}:{endpoint}"

        if key not in self.rate_limits:
            return 0.0

        rate_limit = self.rate_limits[key]

        if rate_limit.current_usage < rate_limit.limit:
            return 0.0

        if rate_limit.reset_time:
            wait_time = (rate_limit.reset_time - datetime.now()).total_seconds()
            return max(0.0, wait_time)

        return float(rate_limit.window_seconds)

    async def cleanup_old_history(self):
        """Clean up old usage history (run periodically)"""
        cutoff_time = datetime.now() - timedelta(hours=24)

        for key, history in self.usage_history.items():
            # Remove entries older than 24 hours
            while history and history[0]["timestamp"] < cutoff_time:
                history.popleft()


# Global monitor instance
_rate_monitor = None


def get_rate_monitor() -> APIRateMonitor:
    """Get or create API rate monitor instance"""
    global _rate_monitor
    if _rate_monitor is None:
        _rate_monitor = APIRateMonitor()
    return _rate_monitor
