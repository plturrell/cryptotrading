"""
Rate limiter to stay within free API tiers
"""

import time
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import threading

class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    
    def __init__(self):
        self.limits = {
            # API limits per minute
            "geckoterminal": {"calls": 30, "period": 60},  # 30 calls/min
            "coingecko": {"calls": 10, "period": 60},      # 10-30 calls/min (conservative)
            "coinmarketcap": {"calls": 30, "period": 60},  # Varies by plan
            "bitquery": {"calls": 60, "period": 60},       # 1 call/sec
            "cryptodatadownload": {"calls": 60, "period": 3600},  # 60 downloads/hour
            "bitget": {"calls": 1, "period": 86400},       # 1 per coin per day
            "yahoo": {"calls": 2000, "period": 3600},      # 2000/hour (unofficial)
            "ai_gateway": {"calls": 100, "period": 60},     # Claude-4 rate limit
        }
        
        self.calls = defaultdict(list)
        self.lock = threading.Lock()
    
    def check_limit(self, api_name: str) -> bool:
        """Check if we can make an API call"""
        if api_name not in self.limits:
            return True
        
        with self.lock:
            limit = self.limits[api_name]
            now = time.time()
            
            # Remove old calls outside the time window
            self.calls[api_name] = [
                call_time for call_time in self.calls[api_name]
                if now - call_time < limit["period"]
            ]
            
            # Check if we're within limit
            return len(self.calls[api_name]) < limit["calls"]
    
    def wait_if_needed(self, api_name: str) -> float:
        """Wait if rate limit exceeded, return wait time"""
        if api_name not in self.limits:
            return 0.0
        
        wait_time = 0.0
        with self.lock:
            limit = self.limits[api_name]
            now = time.time()
            
            # Clean old calls
            self.calls[api_name] = [
                call_time for call_time in self.calls[api_name]
                if now - call_time < limit["period"]
            ]
            
            # If at limit, calculate wait time
            if len(self.calls[api_name]) >= limit["calls"]:
                oldest_call = min(self.calls[api_name])
                wait_time = (oldest_call + limit["period"]) - now
                
                if wait_time > 0:
                    print(f"Rate limit reached for {api_name}. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
        
        return wait_time
    
    def record_call(self, api_name: str):
        """Record an API call"""
        with self.lock:
            self.calls[api_name].append(time.time())
    
    def get_remaining_calls(self, api_name: str) -> Dict[str, int]:
        """Get remaining calls in current period"""
        if api_name not in self.limits:
            return {"remaining": -1, "period": 0}
        
        with self.lock:
            limit = self.limits[api_name]
            now = time.time()
            
            # Count recent calls
            recent_calls = [
                call_time for call_time in self.calls[api_name]
                if now - call_time < limit["period"]
            ]
            
            return {
                "remaining": limit["calls"] - len(recent_calls),
                "period": limit["period"],
                "reset_in": min([limit["period"] - (now - call) for call in recent_calls]) if recent_calls else 0
            }
    
    def get_all_limits(self) -> Dict[str, Dict]:
        """Get current status of all API limits"""
        status = {}
        for api_name in self.limits:
            remaining = self.get_remaining_calls(api_name)
            status[api_name] = {
                "limit": self.limits[api_name]["calls"],
                "period": self.limits[api_name]["period"],
                "remaining": remaining["remaining"],
                "reset_in": remaining["reset_in"]
            }
        return status

# Global rate limiter instance
rate_limiter = RateLimiter()