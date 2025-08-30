#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

// Real agent configurations - using actual existing Python agents
const agents = [
    {
        name: 'data-analysis-agent',
        script: 'src/cryptotrading/core/agents/specialized/data_analysis_agent.py',
        env: {
            AGENT_TYPE: 'data-analysis',
            BLOCKCHAIN_URL: 'http://localhost:8545',
            CDS_BACKEND_URL: 'http://localhost:4004',
            MCP_DATA_URL: 'http://localhost:3002'
        }
    },
    {
        name: 'ml-agent',
        script: 'src/cryptotrading/core/agents/specialized/ml_agent.py',
        env: {
            AGENT_TYPE: 'ml-analysis',
            BLOCKCHAIN_URL: 'http://localhost:8545',
            CDS_BACKEND_URL: 'http://localhost:4004',
            MCP_ANALYTICS_URL: 'http://localhost:3003'
        }
    },
    {
        name: 'trading-algorithm-agent',
        script: 'src/cryptotrading/core/agents/specialized/trading_algorithm_agent.py',
        env: {
            AGENT_TYPE: 'trading-algorithm',
            BLOCKCHAIN_URL: 'http://localhost:8545',
            CDS_BACKEND_URL: 'http://localhost:4004',
            MCP_DATA_URL: 'http://localhost:3002'
        }
    }
];

class AgentManager {
    constructor() {
        this.processes = new Map();
    }

    async startAll() {
        console.log('🤖 Starting Real Crypto Trading Agents...\n');
        
        for (const agent of agents) {
            await this.startAgent(agent);
            await this.sleep(1000);
        }

        console.log('\n✅ All real agents started successfully!');
        this.setupGracefulShutdown();
    }

    async startAgent(agentConfig) {
        const { name, script, env } = agentConfig;
        const scriptPath = path.join(__dirname, '..', script);
        
        console.log(`   🔍 Starting ${name} from ${scriptPath}`);
        
        const agentProcess = spawn('python3', [scriptPath], {
            stdio: ['ignore', 'pipe', 'pipe'],
            env: { ...process.env, ...env }
        });

        this.processes.set(name, agentProcess);
        console.log(`   🤖 ${name} started`);

        // Handle process output
        agentProcess.stdout.on('data', (data) => {
            console.log(`[${name}] ${data.toString().trim()}`);
        });

        agentProcess.stderr.on('data', (data) => {
            console.error(`[${name}] ERROR: ${data.toString().trim()}`);
        });

        agentProcess.on('exit', (code) => {
            console.log(`[${name}] Process exited with code ${code}`);
            this.processes.delete(name);
        });
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    setupGracefulShutdown() {
        const shutdown = async () => {
            console.log('\n🛑 Shutting down agents...');
            
            for (const [name, process] of this.processes) {
                console.log(`   Stopping ${name}...`);
                process.kill('SIGTERM');
            }
            
            // Wait for processes to exit
            await this.sleep(2000);
            
            console.log('✅ All agents stopped');
            process.exit(0);
        };

        process.on('SIGINT', shutdown);
        process.on('SIGTERM', shutdown);
    }
}

// Start the agent manager
const manager = new AgentManager();
manager.startAll().catch(console.error);
