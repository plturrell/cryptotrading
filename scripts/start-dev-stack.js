#!/usr/bin/env node

const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs');

// Configuration
const config = {
    anvil: {
        enabled: true,
        port: 8545,
        accounts: 10,
        balance: 10000
    },
    a2aRegistry: {
        enabled: true,
        port: 3001,
        path: '../a2a/a2aNetwork'
    },
    mcpServer: {
        enabled: true,
        port: 3002,
        types: ['trading', 'analytics', 'risk']
    },
    cdsBackend: {
        enabled: true,
        port: 4004
    },
    ui5Frontend: {
        enabled: true,
        port: 8080
    },
    agents: {
        enabled: true,
        types: ['market-monitor', 'risk-analyzer', 'trade-executor']
    }
};

class DevStackManager {
    constructor() {
        this.processes = new Map();
        this.startupOrder = [
            'anvil',
            'a2aRegistry', 
            'mcpServers',
            'agents',
            'cdsBackend',
            'ui5Frontend'
        ];
    }

    async startAll() {
        console.log('🚀 Starting Crypto Trading Development Stack...\n');
        
        for (const service of this.startupOrder) {
            if (config[service]?.enabled || (service === 'mcpServers' && config.mcpServer.enabled) || (service === 'agents' && config.agents.enabled)) {
                await this.startService(service);
                await this.sleep(2000); // Wait between services
            }
        }

        console.log('\n✅ All services started successfully!');
        this.printServiceStatus();
        this.setupGracefulShutdown();
    }

    async startService(serviceName) {
        console.log(`📦 Starting ${serviceName}...`);
        
        switch (serviceName) {
            case 'anvil':
                await this.startAnvil();
                break;
            case 'a2aRegistry':
                await this.startA2ARegistry();
                break;
            case 'mcpServers':
                await this.startMCPServers();
                break;
            case 'agents':
                await this.startAgents();
                break;
            case 'cdsBackend':
                await this.startCDSBackend();
                break;
            case 'ui5Frontend':
                await this.startUI5Frontend();
                break;
        }
    }

    async startAnvil() {
        if (!this.checkCommand('anvil')) {
            console.log('⚠️  Anvil not found. Install with: curl -L https://foundry.paradigm.xyz | bash && foundryup');
            return;
        }

        const anvilProcess = spawn('anvil', [
            '--port', config.anvil.port.toString(),
            '--accounts', config.anvil.accounts.toString(),
            '--balance', config.anvil.balance.toString(),
            '--host', '0.0.0.0'
        ], { stdio: ['ignore', 'pipe', 'pipe'] });

        this.processes.set('anvil', anvilProcess);
        console.log(`   ⛓️  Anvil blockchain running on port ${config.anvil.port}`);
    }

    async startA2ARegistry() {
        const a2aPath = path.resolve(__dirname, config.a2aRegistry.path);
        
        if (!fs.existsSync(a2aPath)) {
            console.log(`⚠️  A2A Network not found at ${a2aPath}`);
            return;
        }

        const registryProcess = spawn('npm', ['run', 'watch'], {
            cwd: a2aPath,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: { ...process.env, PORT: config.a2aRegistry.port.toString() }
        });

        this.processes.set('a2aRegistry', registryProcess);
        console.log(`   🌐 A2A Registry running on port ${config.a2aRegistry.port}`);
    }

    async startMCPServers() {
        for (const serverType of config.mcpServer.types) {
            const mcpProcess = spawn('node', [
                path.join(__dirname, `../src/cryptotrading/core/agents/mcp_tools/${serverType}_mcp_server.js`)
            ], {
                stdio: ['ignore', 'pipe', 'pipe'],
                env: { ...process.env, MCP_PORT: (config.mcpServer.port + config.mcpServer.types.indexOf(serverType)).toString() }
            });

            this.processes.set(`mcp-${serverType}`, mcpProcess);
            console.log(`   🔧 MCP ${serverType} server running on port ${config.mcpServer.port + config.mcpServer.types.indexOf(serverType)}`);
        }
    }

    async startAgents() {
        for (const agentType of config.agents.types) {
            const agentProcess = spawn('node', [
                path.join(__dirname, `../src/cryptotrading/core/agents/${agentType}_agent.js`)
            ], {
                stdio: ['ignore', 'pipe', 'pipe'],
                env: { 
                    ...process.env, 
                    AGENT_TYPE: agentType,
                    BLOCKCHAIN_URL: `http://localhost:${config.anvil.port}`,
                    A2A_REGISTRY_URL: `http://localhost:${config.a2aRegistry.port}`
                }
            });

            this.processes.set(`agent-${agentType}`, agentProcess);
            console.log(`   🤖 ${agentType} agent started`);
        }
    }

    async startCDSBackend() {
        const cdsProcess = spawn('npm', ['run', 'watch'], {
            cwd: process.cwd(),
            stdio: ['ignore', 'pipe', 'pipe']
        });

        this.processes.set('cdsBackend', cdsProcess);
        console.log(`   💾 CDS Backend running on port ${config.cdsBackend.port}`);
    }

    async startUI5Frontend() {
        const ui5Process = spawn('npm', ['run', 'start:ui'], {
            cwd: process.cwd(),
            stdio: ['ignore', 'pipe', 'pipe']
        });

        this.processes.set('ui5Frontend', ui5Process);
        console.log(`   🎨 UI5 Frontend running on port ${config.ui5Frontend.port}`);
    }

    checkCommand(command) {
        try {
            require('child_process').execSync(`which ${command}`, { stdio: 'ignore' });
            return true;
        } catch {
            return false;
        }
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    printServiceStatus() {
        console.log('\n📊 Service Status:');
        console.log('┌─────────────────────┬──────────┬─────────────────────────┐');
        console.log('│ Service             │ Port     │ Status                  │');
        console.log('├─────────────────────┼──────────┼─────────────────────────┤');
        
        const services = [
            ['Anvil Blockchain', config.anvil.port, this.processes.has('anvil') ? '🟢 Running' : '🔴 Stopped'],
            ['A2A Registry', config.a2aRegistry.port, this.processes.has('a2aRegistry') ? '🟢 Running' : '🔴 Stopped'],
            ['CDS Backend', config.cdsBackend.port, this.processes.has('cdsBackend') ? '🟢 Running' : '🔴 Stopped'],
            ['UI5 Frontend', config.ui5Frontend.port, this.processes.has('ui5Frontend') ? '🟢 Running' : '🔴 Stopped']
        ];

        services.forEach(([name, port, status]) => {
            console.log(`│ ${name.padEnd(19)} │ ${port.toString().padEnd(8)} │ ${status.padEnd(23)} │`);
        });
        
        console.log('└─────────────────────┴──────────┴─────────────────────────┘');
        
        console.log('\n🔗 Quick Links:');
        console.log(`   Frontend:    http://localhost:${config.ui5Frontend.port}`);
        console.log(`   Backend:     http://localhost:${config.cdsBackend.port}`);
        console.log(`   Blockchain:  http://localhost:${config.anvil.port}`);
        console.log(`   A2A Registry: http://localhost:${config.a2aRegistry.port}`);
    }

    setupGracefulShutdown() {
        const shutdown = () => {
            console.log('\n🛑 Shutting down all services...');
            
            this.processes.forEach((process, name) => {
                console.log(`   Stopping ${name}...`);
                process.kill('SIGTERM');
            });
            
            setTimeout(() => {
                console.log('✅ All services stopped');
                process.exit(0);
            }, 2000);
        };

        process.on('SIGINT', shutdown);
        process.on('SIGTERM', shutdown);
    }
}

// Start the development stack
const manager = new DevStackManager();
manager.startAll().catch(console.error);
