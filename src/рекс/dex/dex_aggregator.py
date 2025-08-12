"""
DEX Aggregator for best price execution across multiple DEXs
"""

from typing import Dict, List, Tuple
import asyncio
import aiohttp

class DEXAggregator:
    def __init__(self, wallet_address: str = "0x88bE2a6408934e32a0Ad63c368Be5b257ca63cC1"):
        self.wallet = wallet_address
        
        # DEX routers
        self.dexs = {
            "uniswap_v3": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
            "sushiswap": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            "1inch": "0x1111111254fb6c44bAC0beD2854e76F90643097d",
            "balancer": "0xBA12222222228d8Ba445958a75a0704d566BF2C8"
        }
        
        # 1inch API for aggregation
        self.aggregator_api = "https://api.1inch.io/v5.0/1"  # Ethereum mainnet
    
    async def get_best_price(self, from_token: str, to_token: str, amount: float) -> Dict:
        """Get best price across all DEXs"""
        prices = {}
        
        # In production, query each DEX
        # For now, return mock comparison
        mock_prices = {
            "uniswap_v3": amount * 0.997,  # 0.3% fee
            "sushiswap": amount * 0.997,
            "1inch": amount * 0.998,  # Better aggregation
            "balancer": amount * 0.996
        }
        
        best_dex = max(mock_prices.items(), key=lambda x: x[1])
        
        return {
            "best_dex": best_dex[0],
            "best_price": best_dex[1],
            "all_prices": mock_prices,
            "savings": best_dex[1] - min(mock_prices.values())
        }
    
    def get_gas_estimates(self) -> Dict[str, int]:
        """Estimate gas for different DEXs"""
        return {
            "uniswap_v3": 184000,
            "sushiswap": 160000,
            "1inch": 220000,  # More complex routing
            "balancer": 195000
        }
    
    def calculate_effective_price(self, 
                                 dex: str, 
                                 price: float, 
                                 gas_price_gwei: float) -> float:
        """Calculate price including gas costs"""
        gas_estimates = self.get_gas_estimates()
        gas_cost_eth = (gas_estimates[dex] * gas_price_gwei) / 1e9
        
        # Assuming ETH price for gas calculation
        eth_price = 2500  # Would fetch real price
        gas_cost_usd = gas_cost_eth * eth_price
        
        return {
            "output_amount": price,
            "gas_cost_usd": gas_cost_usd,
            "effective_output": price - gas_cost_usd,
            "dex": dex
        }
    
    def get_liquidity_sources(self, token_pair: str) -> List[Dict]:
        """Get liquidity sources for a token pair"""
        return [
            {"dex": "uniswap_v3", "liquidity": "50M", "fee": "0.3%"},
            {"dex": "sushiswap", "liquidity": "20M", "fee": "0.3%"},
            {"dex": "balancer", "liquidity": "15M", "fee": "0.2%"},
            {"dex": "curve", "liquidity": "100M", "fee": "0.04%"}
        ]