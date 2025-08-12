"""
MetaMask wallet integration for рекс.com
Connected to wallet: 0x88bE2a6408934e32a0Ad63c368Be5b257ca63cC1
"""

import os
from web3 import Web3
from eth_account import Account
from decimal import Decimal
from typing import Dict, Optional, List
import json

class MetaMaskClient:
    def __init__(self, wallet_address: str = "0x88bE2a6408934e32a0Ad63c368Be5b257ca63cC1"):
        """Initialize MetaMask client for the specified wallet"""
        self.wallet_address = Web3.to_checksum_address(wallet_address)
        
        # Connect to Ethereum mainnet via Infura (you can also use Alchemy)
        infura_url = os.getenv('INFURA_URL', 'https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID')
        self.w3 = Web3(Web3.HTTPProvider(infura_url))
        
        # For local testing, you can use Ganache
        if os.getenv('ETH_NETWORK') == 'local':
            self.w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
        
        self.connected = self.w3.is_connected()
        
    def get_balance(self, token_address: Optional[str] = None) -> Dict[str, float]:
        """Get wallet balance in ETH or specific token"""
        if not self.connected:
            return {"error": "Not connected to Ethereum network"}
        
        if token_address is None:
            # Get ETH balance
            balance_wei = self.w3.eth.get_balance(self.wallet_address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            
            return {
                "address": self.wallet_address,
                "balance_eth": float(balance_eth),
                "balance_wei": balance_wei,
                "symbol": "ETH"
            }
        else:
            # Get ERC20 token balance
            return self._get_token_balance(token_address)
    
    def _get_token_balance(self, token_address: str) -> Dict[str, float]:
        """Get ERC20 token balance"""
        # ERC20 ABI for balanceOf function
        erc20_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            }
        ]
        
        try:
            token_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=erc20_abi
            )
            
            balance = token_contract.functions.balanceOf(self.wallet_address).call()
            decimals = token_contract.functions.decimals().call()
            symbol = token_contract.functions.symbol().call()
            
            # Convert to human-readable format
            balance_formatted = balance / (10 ** decimals)
            
            return {
                "address": self.wallet_address,
                "token_address": token_address,
                "balance": float(balance_formatted),
                "balance_raw": balance,
                "symbol": symbol,
                "decimals": decimals
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_transaction_history(self, limit: int = 10) -> List[Dict]:
        """Get recent transactions for the wallet"""
        if not self.connected:
            return [{"error": "Not connected to Ethereum network"}]
        
        # Note: Full transaction history requires Etherscan API
        # This is a simplified version showing recent block transactions
        try:
            latest_block = self.w3.eth.get_block('latest')
            transactions = []
            
            # Get transactions from recent blocks
            for i in range(min(limit, 5)):
                block = self.w3.eth.get_block(latest_block.number - i, full_transactions=True)
                for tx in block.transactions:
                    if tx['from'] == self.wallet_address or tx['to'] == self.wallet_address:
                        transactions.append({
                            'hash': tx['hash'].hex(),
                            'from': tx['from'],
                            'to': tx['to'],
                            'value_eth': float(self.w3.from_wei(tx['value'], 'ether')),
                            'block': tx['blockNumber'],
                            'gas': tx['gas'],
                            'gasPrice': tx['gasPrice']
                        })
            
            return transactions[:limit]
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_gas_price(self) -> Dict[str, float]:
        """Get current gas prices"""
        if not self.connected:
            return {"error": "Not connected to Ethereum network"}
        
        try:
            gas_price_wei = self.w3.eth.gas_price
            gas_price_gwei = self.w3.from_wei(gas_price_wei, 'gwei')
            
            return {
                "gas_price_wei": gas_price_wei,
                "gas_price_gwei": float(gas_price_gwei),
                "gas_price_eth": float(self.w3.from_wei(gas_price_wei, 'ether'))
            }
        except Exception as e:
            return {"error": str(e)}
    
    def monitor_wallet(self) -> Dict[str, any]:
        """Monitor wallet for DeFi opportunities"""
        if not self.connected:
            return {"error": "Not connected to Ethereum network"}
        
        # Get comprehensive wallet status
        eth_balance = self.get_balance()
        gas_price = self.get_gas_price()
        
        # Common DeFi tokens to check
        defi_tokens = {
            "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        }
        
        token_balances = []
        for symbol, address in defi_tokens.items():
            balance = self._get_token_balance(address)
            if "error" not in balance and balance.get("balance", 0) > 0:
                token_balances.append(balance)
        
        return {
            "wallet_address": self.wallet_address,
            "eth_balance": eth_balance,
            "token_balances": token_balances,
            "gas_price": gas_price,
            "network": "mainnet" if "infura" in str(self.w3.provider) else "local",
            "connected": self.connected
        }
    
    def prepare_transaction(self, to_address: str, amount_eth: float) -> Dict:
        """Prepare a transaction (doesn't send it)"""
        if not self.connected:
            return {"error": "Not connected to Ethereum network"}
        
        try:
            nonce = self.w3.eth.get_transaction_count(self.wallet_address)
            gas_price = self.w3.eth.gas_price
            
            transaction = {
                'nonce': nonce,
                'to': Web3.to_checksum_address(to_address),
                'value': self.w3.to_wei(amount_eth, 'ether'),
                'gas': 21000,  # Standard ETH transfer
                'gasPrice': gas_price,
                'chainId': self.w3.eth.chain_id
            }
            
            # Estimate total cost
            total_gas_eth = self.w3.from_wei(transaction['gas'] * gas_price, 'ether')
            total_cost_eth = amount_eth + float(total_gas_eth)
            
            return {
                "transaction": transaction,
                "amount_eth": amount_eth,
                "gas_cost_eth": float(total_gas_eth),
                "total_cost_eth": total_cost_eth,
                "from": self.wallet_address,
                "to": to_address
            }
        except Exception as e:
            return {"error": str(e)}