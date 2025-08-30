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
        
        // Check dependencies first
        await this.checkDependencies();
        
        for (const service of this.startupOrder) {
            if (config[service]?.enabled || (service === 'mcpServers' && config.mcpServer.enabled) || (service === 'agents' && config.agents.enabled)) {
                await this.startService(service);
                await this.sleep(2000); // Wait between services
            }
        }
        
        // Print status and setup shutdown handlers
        this.printServiceStatus();
        this.setupGracefulShutdown();
        
        console.log('\n✅ Development stack started successfully!');
        console.log('Press Ctrl+C to stop all services');
    }

    async startService(serviceName) {
        console.log(`🔄 Starting ${serviceName}...`);
        
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
            default:
                console.log(`⚠️  Unknown service: ${serviceName}`);
        }
    }

    async checkDependencies() {
        console.log('🔍 Checking dependencies...');
        const { execSync } = require('child_process');
        
        // Check Python dependencies for MCP servers
        try {
            execSync('python3 -c "import aiohttp, aiohttp_cors"', { stdio: 'ignore' });
            console.log('   ✅ Python MCP dependencies available');
        } catch (error) {
            console.log('   📦 Installing Python MCP dependencies...');
            try {
                execSync('pip3 install aiohttp aiohttp_cors', { stdio: 'inherit' });
            } catch (installError) {
                console.log('   ⚠️  Failed to install Python dependencies. Install manually: pip3 install aiohttp aiohttp_cors');
            }
        }
        
        // Check Node.js dependencies
        const requiredPackages = ['axios', 'ethers', 'web3'];
        for (const pkg of requiredPackages) {
            try {
                require.resolve(pkg);
                console.log(`   ✅ ${pkg} available`);
            } catch (error) {
                console.log(`   ❌ Missing ${pkg} - run 'npm install'`);
                throw new Error(`Missing required package: ${pkg}`);
            }
        }
        
        // Check external commands
        const commands = [
            { cmd: 'anvil', install: 'curl -L https://foundry.paradigm.xyz | bash && foundryup' },
            { cmd: 'python3', install: 'Install Python 3' }
        ];
        
        for (const { cmd, install } of commands) {
            if (!this.checkCommand(cmd)) {
                console.log(`   ⚠️  ${cmd} not found. Install with: ${install}`);
            } else {
                console.log(`   ✅ ${cmd} available`);
            }
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
        const a2aPath = path.join(__dirname, '../../a2a/a2aNetwork');
        
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
        // Start data analysis MCP server
        const dataProcess = spawn('python3', [
            path.join(__dirname, '../scripts/mcp_data_server.py'),
            '--host', 'localhost',
            '--port', '3002'
        ], {
            stdio: ['ignore', 'pipe', 'pipe'],
            env: { ...process.env }
        });

        this.processes.set('mcp-data', dataProcess);
        console.log(`   🔧 MCP data analysis server running on port 3002`);

        // Start analytics MCP server
        const analyticsProcess = spawn('python3', [
            path.join(__dirname, '../scripts/mcp_analytics_server.py'),
            '--host', 'localhost',
            '--port', '3003'
        ], {
            stdio: ['ignore', 'pipe', 'pipe'],
            env: { ...process.env }
        });

        this.processes.set('mcp-analytics', analyticsProcess);
        console.log(`   🔧 MCP analytics server running on port 3003`);
    }

    async startAgents() {
        // Use the real agent startup script
        const agentProcess = spawn('node', [
            path.join(__dirname, 'start-agents.js')
        ], {
            stdio: ['ignore', 'pipe', 'pipe'],
            env: { 
                ...process.env, 
                BLOCKCHAIN_URL: `http://localhost:${config.anvil.port}`,
                A2A_REGISTRY_URL: `http://localhost:${config.a2aRegistry.port}`,
                MCP_DATA_URL: 'http://localhost:3002',
                MCP_ANALYTICS_URL: 'http://localhost:3003'
            }
        });

        this.processes.set('agents', agentProcess);
        console.log(`   🤖 Real agents started (data-analysis, ml, trading-algorithm)`);
        
        // Wait for agents to start, then register them
        await this.sleep(3000);
        await this.registerAgents();
    }

    async registerAgents() {
        console.log('📝 Registering agents with blockchain and A2A registry...');
        
        try {
            // Use the existing agent registration script
            const registerProcess = spawn('python3', [
                path.join(__dirname, 'start_anvil_and_register_agents.py')
            ], {
                stdio: ['ignore', 'pipe', 'pipe'],
                env: { 
                    ...process.env,
                    SKIP_ANVIL_START: 'true'  // Anvil already running
                }
            });

            registerProcess.stdout.on('data', (data) => {
                console.log(`[registration] ${data.toString().trim()}`);
            });

            registerProcess.stderr.on('data', (data) => {
                console.error(`[registration] ERROR: ${data.toString().trim()}`);
            });

            registerProcess.on('exit', (code) => {
                if (code === 0) {
                    console.log('✅ Agent registration completed');
                } else {
                    console.log('⚠️  Agent registration had issues');
                }
            });

        } catch (error) {
            console.log('⚠️  Agent registration failed:', error.message);
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
