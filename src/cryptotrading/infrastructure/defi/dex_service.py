"""
DEX (Decentralized Exchange) service using GeckoTerminal API
Provides trending pools and liquidity data across multiple DEXs and networks
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)

@dataclass
class DEXPool:
    """DEX pool data structure"""
    address: str
    token0: str
    token1: str
    token0_symbol: str
    token1_symbol: str
    liquidity_usd: float
    volume_24h_usd: float
    fee_tier: float
    dex: str
    network: str
    apy: Optional[float] = None

@dataclass
class DEXTrend:
    """DEX trending data structure"""
    pools: List[DEXPool]
    total_tvl: float
    total_volume_24h: float
    trending_tokens: List[str]
    last_updated: datetime

class DEXService:
    """GeckoTerminal DEX service"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = "https://api.geckoterminal.com/api/v2"
        
        # Rate limiting - GeckoTerminal allows 30 calls/minute
        self.rate_limit_delay = 2.0  # 2 seconds between requests
        self.last_request_time = 0
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Rex-Crypto-Trading/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make GeckoTerminal API request"""
        if not self.session:
            raise RuntimeError("DEXService not initialized. Use async context manager.")
        
        await self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Rate limited - wait longer
                    logger.warning("Rate limited by GeckoTerminal API, waiting 60 seconds")
                    await asyncio.sleep(60)
                    return await self._make_request(endpoint, params)
                else:
                    logger.error("GeckoTerminal API error: %s", response.status)
                    response.raise_for_status()
        
        except aiohttp.ClientError as e:
            logger.error("GeckoTerminal request failed: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error in GeckoTerminal request: %s", e)
            raise
    
    async def get_trending_pools(self, network: str = "eth", limit: int = 20) -> List[DEXPool]:
        """Get trending pools from GeckoTerminal API"""
        try:
            # Use GeckoTerminal trending pools endpoint
            endpoint = f"networks/{network}/trending_pools"
            params = {"page": 1}
            
            data = await self._make_request(endpoint, params)
            pools = []
            
            pool_data_list = data.get('data', [])[:limit]
            
            for pool_info in pool_data_list:
                attributes = pool_info.get('attributes', {})
                relationships = pool_info.get('relationships', {})
                
                # Extract token information
                base_token = relationships.get('base_token', {}).get('data', {})
                quote_token = relationships.get('quote_token', {}).get('data', {})
                dex_info = relationships.get('dex', {}).get('data', {})
                
                pools.append(DEXPool(
                    address=attributes.get('address', ''),
                    token0=base_token.get('id', ''),
                    token1=quote_token.get('id', ''),
                    token0_symbol=attributes.get('base_token_symbol', ''),
                    token1_symbol=attributes.get('quote_token_symbol', ''),
                    liquidity_usd=float(attributes.get('reserve_in_usd', 0)),
                    volume_24h_usd=float(attributes.get('volume_usd', {}).get('h24', 0)),
                    fee_tier=float(attributes.get('pool_created_at', 0)),  # Placeholder
                    dex=dex_info.get('id', 'unknown'),
                    network=network
                ))
            
            return pools
            
        except (aiohttp.ClientError, ValueError) as e:
            logger.error("Failed to get trending pools for %s: %s", network, e)
            return []
        except Exception as e:
            logger.error("Unexpected error getting trending pools for %s: %s", network, e)
            return []
    
    async def get_networks(self) -> List[Dict[str, Any]]:
        """Get available networks from GeckoTerminal"""
        try:
            data = await self._make_request("networks")
            return data.get('data', [])
        except (aiohttp.ClientError, ValueError) as e:
            logger.error("Failed to get networks: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error getting networks: %s", e)
            return []
    
    async def get_trending_pools_aggregated(self, networks: List[str] = None, limit_per_network: int = 10) -> DEXTrend:
        """Get aggregated trending pools from multiple networks using GeckoTerminal"""
        if networks is None:
            networks = ["eth", "bsc", "polygon_pos", "arbitrum", "base"]  # Popular networks
        
        try:
            # Fetch from multiple networks concurrently
            network_tasks = [
                self.get_trending_pools(network, limit_per_network) 
                for network in networks
            ]
            
            results = await asyncio.gather(*network_tasks, return_exceptions=True)
            
            # Combine all pools
            all_pools = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("Network %s query failed: %s", networks[i], result)
                else:
                    all_pools.extend(result)
            
            # Sort by volume
            all_pools.sort(key=lambda p: p.volume_24h_usd, reverse=True)
            
            # Calculate aggregated metrics
            total_tvl = sum(pool.liquidity_usd for pool in all_pools)
            total_volume_24h = sum(pool.volume_24h_usd for pool in all_pools)
            
            # Extract trending tokens
            trending_tokens = []
            for pool in all_pools[:20]:  # Top 20 pools
                trending_tokens.extend([pool.token0_symbol, pool.token1_symbol])
            
            # Remove duplicates and keep order
            seen = set()
            unique_trending = []
            for token in trending_tokens:
                if token not in seen:
                    seen.add(token)
                    unique_trending.append(token)
            
            return DEXTrend(
                pools=all_pools,
                total_tvl=total_tvl,
                total_volume_24h=total_volume_24h,
                trending_tokens=unique_trending[:30],  # Top 30 trending tokens
                last_updated=datetime.now()
            )
            
        except (aiohttp.ClientError, ValueError) as e:
            logger.error("Failed to get aggregated trending pools: %s", e)
        except Exception as e:
            logger.error("Unexpected error getting aggregated trending pools: %s", e)
            return DEXTrend(
                pools=[],
                total_tvl=0,
                total_volume_24h=0,
                trending_tokens=[],
                last_updated=datetime.now()
            )
    
    async def get_trending_dict(self, networks: List[str] = None, limit_per_network: int = 5) -> Dict[str, Any]:
        """Get trending data as dictionary (for API compatibility)"""
        try:
            trend_data = await self.get_trending_pools_aggregated(networks, limit_per_network)
            
            return {
                "pools": [
                    {
                        "address": pool.address,
                        "pair": f"{pool.token0_symbol}/{pool.token1_symbol}",
                        "token0": pool.token0_symbol,
                        "token1": pool.token1_symbol,
                        "liquidity_usd": pool.liquidity_usd,
                        "volume_24h_usd": pool.volume_24h_usd,
                        "fee_tier": pool.fee_tier,
                        "dex": pool.dex,
                        "network": pool.network,
                        "apy": pool.apy
                    }
                    for pool in trend_data.pools[:25]  # Top 25 pools
                ],
                "summary": {
                    "total_tvl": trend_data.total_tvl,
                    "total_volume_24h": trend_data.total_volume_24h,
                    "trending_tokens": trend_data.trending_tokens,
                    "pool_count": len(trend_data.pools),
                    "networks_queried": networks or ["eth", "bsc", "polygon_pos", "arbitrum", "base"]
                },
                "last_updated": trend_data.last_updated.isoformat(),
                "status": "success",
                "source": "geckoterminal_api"
            }
            
        except (aiohttp.ClientError, ValueError) as e:
            logger.error("Failed to get trending dict: %s", e)
        except Exception as e:
            logger.error("Unexpected error getting trending dict: %s", e)
            return {
                "error": str(e),
                "status": "error",
                "source": "geckoterminal_api"
            }

# Global service instance
_dex_service = None

def get_dex_service() -> DEXService:
    """Get or create DEX service instance"""
    return DEXService()
