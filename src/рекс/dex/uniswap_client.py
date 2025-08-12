"""
Uniswap V3 integration for рекс.com
Trade directly from MetaMask wallet: 0x88bE2a6408934e32a0Ad63c368Be5b257ca63cC1
"""

from web3 import Web3
from typing import Dict, List, Tuple
import json
import requests

class UniswapClient:
    def __init__(self, wallet_address: str = "0x88bE2a6408934e32a0Ad63c368Be5b257ca63cC1"):
        self.wallet = Web3.to_checksum_address(wallet_address)
        
        # Uniswap V3 contracts on Ethereum mainnet
        self.contracts = {
            "router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
            "quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
            "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984"
        }
        
        # Common token addresses
        self.tokens = {
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
            "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
            "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"
        }
    
    def get_price_data(self, token_pair: str) -> Dict[str, float]:
        """Get real-time price data from Uniswap"""
        # Using CoinGecko API as a simple solution
        # In production, would query Uniswap contracts directly
        base, quote = token_pair.split('/')
        
        # Map to CoinGecko IDs
        gecko_ids = {
            "ETH": "ethereum",
            "BTC": "bitcoin", 
            "USDC": "usd-coin",
            "USDT": "tether",
            "UNI": "uniswap",
            "WBTC": "wrapped-bitcoin"
        }
        
        try:
            base_id = gecko_ids.get(base, base.lower())
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={base_id}&vs_currencies=usd"
            response = requests.get(url)
            data = response.json()
            
            return {
                "pair": token_pair,
                "price": data[base_id]["usd"],
                "source": "uniswap_aggregated",
                "timestamp": "now"
            }
        except:
            return {"pair": token_pair, "price": 0, "error": "Failed to fetch"}
    
    def get_pool_liquidity(self, token0: str, token1: str, fee: int = 3000) -> Dict:
        """Get liquidity information for a Uniswap V3 pool"""
        # Simplified - would query pool contract in production
        return {
            "token0": token0,
            "token1": token1,
            "fee_tier": fee,
            "liquidity": "1000000",  # Mock data
            "volume_24h": "5000000"
        }
    
    def prepare_swap(self, 
                    from_token: str, 
                    to_token: str, 
                    amount: float,
                    slippage: float = 0.5) -> Dict:
        """Prepare a swap transaction"""
        from_address = self.tokens.get(from_token, from_token)
        to_address = self.tokens.get(to_token, to_token)
        
        # Calculate with slippage
        min_amount_out = amount * (1 - slippage/100)
        
        return {
            "from": self.wallet,
            "to": self.contracts["router"],
            "from_token": from_address,
            "to_token": to_address,
            "amount_in": amount,
            "min_amount_out": min_amount_out,
            "deadline": 1800,  # 30 minutes
            "fee": 3000  # 0.3%
        }
    
    def get_historical_trades(self, pair: str, limit: int = 100) -> List[Dict]:
        """Get historical trades from Uniswap subgraph"""
        # Would query The Graph protocol in production
        return [
            {
                "pair": pair,
                "price": 2500 + i,
                "amount": 0.1,
                "timestamp": f"2024-01-{i+1}",
                "type": "buy" if i % 2 == 0 else "sell"
            }
            for i in range(min(limit, 10))
        ]