#!/usr/bin/env node
/**
 * Deploy A2A Smart Contracts to local Anvil blockchain
 * Supports both local development and mainnet deployment
 */

const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Contract deployment configuration
const CONFIG = {
    local: {
        rpcUrl: "http://localhost:8545",
        chainId: 31337,
        gasPrice: 0,
        confirmations: 1
    },
    testnet: {
        rpcUrl: process.env.TESTNET_RPC_URL || "https://rpc.ankr.com/eth_goerli",
        chainId: 5,
        gasPrice: 20000000000, // 20 gwei
        confirmations: 2
    },
    mainnet: {
        rpcUrl: process.env.MAINNET_RPC_URL || "https://rpc.ankr.com/eth",
        chainId: 1,
        gasPrice: 30000000000, // 30 gwei
        confirmations: 5
    }
};

async function deployContracts(network = "local") {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`Deploying A2A Contracts to ${network}`);
    console.log(`${"=".repeat(60)}\n`);

    try {
        // Get network config
        const config = CONFIG[network];
        if (!config) {
            throw new Error(`Unknown network: ${network}`);
        }

        // Get deployer account
        const [deployer] = await ethers.getSigners();
        console.log("Deploying with account:", deployer.address);
        
        const balance = await deployer.getBalance();
        console.log("Account balance:", ethers.utils.formatEther(balance), "ETH");

        // Deploy A2ARegistry
        console.log("\n1. Deploying A2ARegistry...");
        const Registry = await ethers.getContractFactory("A2ARegistry");
        const registry = await Registry.deploy({
            gasPrice: config.gasPrice
        });
        
        await registry.deployed();
        console.log("   ✅ A2ARegistry deployed to:", registry.address);

        // Wait for confirmations
        if (config.confirmations > 1) {
            console.log(`   ⏳ Waiting for ${config.confirmations} confirmations...`);
            await registry.deployTransaction.wait(config.confirmations);
            console.log("   ✅ Confirmed");
        }

        // Deploy A2AMessaging
        console.log("\n2. Deploying A2AMessaging...");
        const Messaging = await ethers.getContractFactory("A2AMessaging");
        const messaging = await Messaging.deploy(registry.address, {
            gasPrice: config.gasPrice
        });
        
        await messaging.deployed();
        console.log("   ✅ A2AMessaging deployed to:", messaging.address);

        // Wait for confirmations
        if (config.confirmations > 1) {
            console.log(`   ⏳ Waiting for ${config.confirmations} confirmations...`);
            await messaging.deployTransaction.wait(config.confirmations);
            console.log("   ✅ Confirmed");
        }

        // Save deployment addresses
        const deployment = {
            network: network,
            chainId: config.chainId,
            deployedAt: new Date().toISOString(),
            contracts: {
                A2ARegistry: {
                    address: registry.address,
                    deployer: deployer.address,
                    blockNumber: registry.deployTransaction.blockNumber
                },
                A2AMessaging: {
                    address: messaging.address,
                    deployer: deployer.address,
                    blockNumber: messaging.deployTransaction.blockNumber
                }
            }
        };

        const deploymentPath = path.join(__dirname, "..", "deployments", `${network}.json`);
        fs.mkdirSync(path.dirname(deploymentPath), { recursive: true });
        fs.writeFileSync(deploymentPath, JSON.stringify(deployment, null, 2));
        
        console.log(`\n✅ Deployment saved to: ${deploymentPath}`);

        // Register initial agents if local
        if (network === "local") {
            console.log("\n3. Registering initial agents...");
            await registerInitialAgents(registry, deployer);
        }

        // Output summary
        console.log(`\n${"=".repeat(60)}`);
        console.log("DEPLOYMENT SUMMARY");
        console.log(`${"=".repeat(60)}`);
        console.log(`Network:          ${network}`);
        console.log(`Chain ID:         ${config.chainId}`);
        console.log(`A2ARegistry:      ${registry.address}`);
        console.log(`A2AMessaging:     ${messaging.address}`);
        console.log(`Deployer:         ${deployer.address}`);
        console.log(`${"=".repeat(60)}\n`);

        return deployment;

    } catch (error) {
        console.error("❌ Deployment failed:", error.message);
        process.exit(1);
    }
}

async function registerInitialAgents(registry, deployer) {
    const agents = [
        {
            agentId: "agent_manager",
            agentType: "management",
            capabilities: ["registration", "compliance", "monitoring"],
            mcpTools: ["register_agent", "validate_compliance", "send_alert"]
        },
        {
            agentId: "trading_algorithm_agent",
            agentType: "trading",
            capabilities: ["grid_trading", "dca", "momentum"],
            mcpTools: ["grid_trading.create", "dca.execute", "momentum.scan"]
        },
        {
            agentId: "news_intelligence_agent",
            agentType: "intelligence",
            capabilities: ["news_collection", "sentiment_analysis", "translation"],
            mcpTools: ["fetch_news", "analyze_sentiment", "translate_article"]
        }
    ];

    for (const agent of agents) {
        console.log(`   Registering ${agent.agentId}...`);
        
        // Generate a dummy wallet for the agent
        const wallet = ethers.Wallet.createRandom();
        
        const tx = await registry.registerAgent(
            agent.agentId,
            wallet.address,
            agent.agentType,
            agent.capabilities,
            agent.mcpTools,
            "QmEmpty" // Placeholder IPFS hash
        );
        
        await tx.wait();
        console.log(`   ✅ ${agent.agentId} registered with wallet: ${wallet.address}`);
    }
}

// Export for use in other scripts
module.exports = {
    deployContracts,
    CONFIG
};

// Run if called directly
if (require.main === module) {
    const network = process.argv[2] || "local";
    
    deployContracts(network)
        .then(() => {
            console.log("✅ Deployment complete!");
            process.exit(0);
        })
        .catch((error) => {
            console.error("❌ Deployment error:", error);
            process.exit(1);
        });
}