# A2A Blockchain Integration

## Overview
This directory contains the blockchain integration components for the A2A (Agent-to-Agent) protocol, providing cryptographic signatures and on-chain audit trails for all agent communications and data operations.

## Key Components

### 1. Blockchain Signatures (`blockchain_signatures.py`)
- Provides ECDSA signing for all A2A messages
- Includes complete blockchain context (chain ID, contract addresses, agent addresses)
- Enables cryptographic verification of message authenticity

### 2. Workflow Instance Contracts (`workflow_instance_contract.py`)
- Deploys unique smart contracts for each workflow execution
- Records all step executions, messages, and data with signatures
- Provides on-chain verification methods

### 3. Smart Contracts
- `A2ANetwork.sol`: Main registry contract for agents and workflows
- `WorkflowInstance.sol`: Individual workflow execution tracking

## Configuration

### Environment Variables
```bash
# Blockchain Configuration
A2A_CHAIN_ID=31337              # Chain ID (default: Anvil local)
A2A_RPC_URL=http://127.0.0.1:8545  # RPC endpoint
A2A_NETWORK_NAME=local-dev      # Network name

# Account Keys (use Anvil defaults for development)
A2A_DEPLOYER_KEY=               # Deployer private key
A2A_HISTORICAL_LOADER_KEY=      # Historical loader agent key
A2A_DATABASE_AGENT_KEY=         # Database agent key

# Options
A2A_REQUIRE_BLOCKCHAIN=false    # Require blockchain connection
```

### Development Setup
1. Start local Anvil blockchain:
   ```bash
   anvil
   ```

2. Compile contracts:
   ```bash
   # Install solc if needed
   npm install -g solc
   
   # Compile WorkflowInstance contract
   python compile_contracts.py
   ```

3. Deploy contracts:
   ```bash
   python setup_local_blockchain.py
   ```

## Production Considerations

### Current Limitations
1. **Bytecode**: WorkflowInstance bytecode must be compiled before deployment
2. **IPFS Integration**: Currently uses local storage, IPFS integration pending
3. **Gas Optimization**: Contract operations not yet optimized for gas usage

### Security Notes
- Never commit private keys to version control
- Use environment variables for all sensitive configuration
- Verify all blockchain addresses before transactions
- Test thoroughly on testnets before mainnet deployment

## Signature Flow

1. **Message Creation**: Agent creates A2A message
2. **Signature**: Message is signed with agent's private key
3. **Blockchain Context**: Chain ID, contract addresses added
4. **Transmission**: Message sent with full signature data
5. **Verification**: Receiver verifies signature matches sender
6. **Recording**: Message hash recorded on workflow instance contract

## Data Integrity

Every piece of data includes:
- Agent blockchain address
- Digital signature (ECDSA)
- Workflow instance contract address
- Timestamp and message ID
- Complete chain of custody

This ensures full auditability and non-repudiation for all A2A operations.