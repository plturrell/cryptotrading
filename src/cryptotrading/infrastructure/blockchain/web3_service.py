"""
Web3 blockchain service for wallet balance and transaction queries
Replaces TODO mock implementations in API endpoints
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware

    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class WalletBalance:
    """Wallet balance structure"""

    address: str
    symbol: str
    balance: Decimal
    balance_usd: Optional[Decimal] = None
    network: str = "ethereum"


@dataclass
class TokenBalance:
    """ERC-20 token balance structure"""

    address: str
    contract_address: str
    symbol: str
    name: str
    balance: Decimal
    decimals: int
    balance_usd: Optional[Decimal] = None


class Web3Service:
    """Web3 blockchain service for wallet operations"""

    def __init__(self):
        if not WEB3_AVAILABLE:
            logger.warning("Web3.py not installed. Install with: pip install web3")
            self.w3 = None
            return

        # Initialize Web3 with multiple provider options
        self.w3 = self._init_web3()

        # Common ERC-20 token contracts on Ethereum mainnet
        self.token_contracts = {
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "USDC": "0xA0b86a33E6441E6C5b4b8b8b8b8b8b8b8b8b8b8b",
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
            "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
            "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        }

        # ERC-20 ABI for balance queries
        self.erc20_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function",
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function",
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function",
            },
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function",
            },
        ]

    def _init_web3(self) -> Optional[Web3]:
        """Initialize Web3 with provider fallback"""
        providers = [
            # Try environment variable first
            os.getenv("WEB3_PROVIDER_URL"),
            # Public endpoints (rate limited)
            "https://eth-mainnet.alchemyapi.io/v2/demo",
            "https://mainnet.infura.io/v3/9aa3d95b3bc440fa88ea12eaa4456161",
            "https://cloudflare-eth.com",
            "https://ethereum.publicnode.com",
        ]

        for provider_url in providers:
            if not provider_url:
                continue

            try:
                if provider_url.startswith("http"):
                    w3 = Web3(Web3.HTTPProvider(provider_url))
                elif provider_url.startswith("ws"):
                    w3 = Web3(Web3.WebsocketProvider(provider_url))
                else:
                    continue

                # Test connection
                if w3.is_connected():
                    logger.info(f"Connected to Web3 provider: {provider_url}")
                    return w3

            except Exception as e:
                logger.warning(f"Failed to connect to {provider_url}: {e}")
                continue

        logger.error("No Web3 providers available")
        return None

    def is_connected(self) -> bool:
        """Check if Web3 is connected"""
        return self.w3 is not None and self.w3.is_connected()

    async def get_eth_balance(self, address: str) -> WalletBalance:
        """Get ETH balance for address"""
        if not self.is_connected():
            raise RuntimeError("Web3 not connected")

        try:
            # Validate address
            if not Web3.is_address(address):
                raise ValueError(f"Invalid Ethereum address: {address}")

            # Get balance in wei
            balance_wei = self.w3.eth.get_balance(Web3.to_checksum_address(address))

            # Convert to ETH
            balance_eth = Web3.from_wei(balance_wei, "ether")

            return WalletBalance(
                address=address, symbol="ETH", balance=Decimal(str(balance_eth)), network="ethereum"
            )

        except Exception as e:
            logger.error(f"Failed to get ETH balance for {address}: {e}")
            raise

    async def get_token_balance(self, address: str, token_symbol: str) -> TokenBalance:
        """Get ERC-20 token balance"""
        if not self.is_connected():
            raise RuntimeError("Web3 not connected")

        contract_address = self.token_contracts.get(token_symbol.upper())
        if not contract_address:
            raise ValueError(f"Unknown token symbol: {token_symbol}")

        try:
            # Create contract instance
            contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(contract_address), abi=self.erc20_abi
            )

            # Get token info
            decimals = contract.functions.decimals().call()
            symbol = contract.functions.symbol().call()
            name = contract.functions.name().call()

            # Get balance
            balance_raw = contract.functions.balanceOf(Web3.to_checksum_address(address)).call()

            # Convert to human readable
            balance = Decimal(balance_raw) / Decimal(10**decimals)

            return TokenBalance(
                address=address,
                contract_address=contract_address,
                symbol=symbol,
                name=name,
                balance=balance,
                decimals=decimals,
            )

        except Exception as e:
            logger.error(f"Failed to get {token_symbol} balance for {address}: {e}")
            raise

    async def get_wallet_summary(
        self, address: str, include_tokens: List[str] = None
    ) -> Dict[str, Any]:
        """Get complete wallet summary with ETH and token balances"""
        if not self.is_connected():
            return {
                "error": "Web3 not connected. Check provider configuration.",
                "status": "connection_error",
                "address": address,
            }

        try:
            summary = {
                "address": address,
                "network": "ethereum",
                "balances": [],
                "total_usd": 0,
                "status": "success",
            }

            # Get ETH balance
            try:
                eth_balance = await self.get_eth_balance(address)
                summary["balances"].append(
                    {
                        "symbol": eth_balance.symbol,
                        "balance": float(eth_balance.balance),
                        "type": "native",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to get ETH balance: {e}")

            # Get token balances
            if include_tokens:
                for token_symbol in include_tokens:
                    try:
                        token_balance = await self.get_token_balance(address, token_symbol)
                        summary["balances"].append(
                            {
                                "symbol": token_balance.symbol,
                                "name": token_balance.name,
                                "balance": float(token_balance.balance),
                                "contract": token_balance.contract_address,
                                "type": "erc20",
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to get {token_symbol} balance: {e}")

            return summary

        except Exception as e:
            logger.error(f"Failed to get wallet summary for {address}: {e}")
            return {"error": str(e), "status": "error", "address": address}

    async def get_transaction_count(self, address: str) -> int:
        """Get transaction count (nonce) for address"""
        if not self.is_connected():
            raise RuntimeError("Web3 not connected")

        try:
            return self.w3.eth.get_transaction_count(Web3.to_checksum_address(address))
        except Exception as e:
            logger.error(f"Failed to get transaction count for {address}: {e}")
            raise


# Global service instance
_web3_service = None


def get_web3_service() -> Web3Service:
    """Get or create Web3 service instance"""
    global _web3_service
    if _web3_service is None:
        _web3_service = Web3Service()
    return _web3_service
