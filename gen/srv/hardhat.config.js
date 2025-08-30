require("@nomiclabs/hardhat-ethers");
require("@nomiclabs/hardhat-waffle");
require("@nomiclabs/hardhat-etherscan");
require("hardhat-gas-reporter");
require("solidity-coverage");

// Load environment variables
require("dotenv").config();

// Default private key for local development
const PRIVATE_KEY = process.env.PRIVATE_KEY || "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

/**
 * @type import('hardhat/config').HardhatUserConfig
 */
module.exports = {
    solidity: {
        version: "0.8.19",
        settings: {
            optimizer: {
                enabled: true,
                runs: 200
            }
        }
    },
    
    networks: {
        // Local Anvil blockchain
        localhost: {
            url: "http://127.0.0.1:8545",
            chainId: 31337,
            accounts: [PRIVATE_KEY]
        },
        
        // Local Hardhat network
        hardhat: {
            chainId: 31337,
            mining: {
                auto: true,
                interval: 0
            }
        },
        
        // Ethereum Goerli testnet
        goerli: {
            url: process.env.GOERLI_RPC_URL || "https://rpc.ankr.com/eth_goerli",
            chainId: 5,
            accounts: [PRIVATE_KEY],
            gasPrice: 20000000000 // 20 gwei
        },
        
        // Ethereum mainnet
        mainnet: {
            url: process.env.MAINNET_RPC_URL || "https://rpc.ankr.com/eth",
            chainId: 1,
            accounts: [PRIVATE_KEY],
            gasPrice: 30000000000 // 30 gwei
        },
        
        // Polygon Mumbai testnet
        mumbai: {
            url: process.env.MUMBAI_RPC_URL || "https://rpc-mumbai.maticvigil.com",
            chainId: 80001,
            accounts: [PRIVATE_KEY],
            gasPrice: 35000000000 // 35 gwei
        },
        
        // Polygon mainnet
        polygon: {
            url: process.env.POLYGON_RPC_URL || "https://polygon-rpc.com",
            chainId: 137,
            accounts: [PRIVATE_KEY],
            gasPrice: 100000000000 // 100 gwei
        }
    },
    
    etherscan: {
        apiKey: {
            mainnet: process.env.ETHERSCAN_API_KEY || "",
            goerli: process.env.ETHERSCAN_API_KEY || "",
            polygon: process.env.POLYGONSCAN_API_KEY || "",
            polygonMumbai: process.env.POLYGONSCAN_API_KEY || ""
        }
    },
    
    gasReporter: {
        enabled: process.env.REPORT_GAS === "true",
        currency: "USD",
        gasPrice: 30,
        coinmarketcap: process.env.COINMARKETCAP_API_KEY
    },
    
    paths: {
        sources: "./contracts",
        tests: "./test/contracts",
        cache: "./cache",
        artifacts: "./artifacts"
    },
    
    mocha: {
        timeout: 40000
    }
};