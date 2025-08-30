/**
 * A2A-Blockchain Bridge
 * Integrates A2A Enhanced Features with Blockchain Data Exchange
 */

const A2AEnhancedFeatures = require("./a2a-enhanced-features");
const Web3 = require("web3");
const { v4: uuidv4 } = require("uuid");

class A2ABlockchainBridge {
    constructor(a2aService, blockchainConfig) {
        this.a2aService = a2aService;
        this.enhancedFeatures = new A2AEnhancedFeatures();

        // Initialize Web3 connection
        this.web3 = new Web3(blockchainConfig.rpcUrl || "http://localhost:8545");
        this.contractAddress = blockchainConfig.contractAddress;
        this.privateKey = blockchainConfig.privateKey;

        // Configuration
        this.ON_CHAIN_THRESHOLD = 256 * 1024; // 256KB - data above this goes to S3
        this.BLOCKCHAIN_CHUNK_SIZE = 32 * 1024; // 32KB chunks for blockchain
    }

    /**
     * Process A2A message and determine storage strategy
     */
    async processMessage(message) {
        const payload = JSON.parse(message.payload);
        const payloadSize = JSON.stringify(payload).length;

        // Determine storage strategy based on size and importance
        if (payloadSize > this.ON_CHAIN_THRESHOLD) {
            // Large data: Store in S3, put reference on-chain
            return await this.handleLargeDataExchange(message);
        } else if (message.priority === "critical" || payload.requiresAudit) {
            // Critical data: Store on-chain for immutability
            return await this.handleOnChainStorage(message);
        } else {
            // Standard data: Use regular A2A messaging
            return await this.handleStandardMessage(message);
        }
    }

    /**
     * Handle large data exchange via S3 with blockchain reference
     */
    async handleLargeDataExchange(message) {
        try {
            const payload = JSON.parse(message.payload);

            // Step 1: Upload to S3 using A2A Enhanced Features
            const s3Result = await this.enhancedFeatures.sendLargeFile(
                message.fromAgentId,
                message.toAgentId,
                `data-${Date.now()}.json`,
                Buffer.from(JSON.stringify(payload)).toString("base64"),
                "application/json"
            );

            if (!s3Result.success) {
                throw new Error("S3 upload failed");
            }

            // Step 2: Store reference on blockchain
            const blockchainRef = {
                messageId: message.id,
                fromAgent: message.fromAgentId,
                toAgent: message.toAgentId,
                s3Url: s3Result.s3Url,
                s3Key: s3Result.message.payload,
                dataHash: this.calculateHash(payload),
                timestamp: Date.now(),
                dataType: "large_file_reference"
            };

            // Store reference on-chain via Python blockchain_data_exchange
            const onChainId = await this.storeOnChain(blockchainRef);

            // Step 3: Send notification via A2A
            const notification = {
                messageType: "LARGE_DATA_READY",
                onChainId: onChainId,
                s3Url: s3Result.s3Url,
                dataHash: blockchainRef.dataHash
            };

            await this.a2aService.sendMessage(
                message.fromAgentId,
                message.toAgentId,
                "data_notification",
                JSON.stringify(notification),
                "high"
            );

            return {
                success: true,
                storageType: "hybrid",
                s3Url: s3Result.s3Url,
                onChainId: onChainId
            };

        } catch (error) {
            console.error("Large data exchange failed:", error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Handle on-chain storage for critical data
     */
    async handleOnChainStorage(message) {
        try {
            const payload = JSON.parse(message.payload);

            // Compress if beneficial
            const compressed = this.enhancedFeatures.compressPayload(payload);

            // Encrypt if required
            let dataToStore = compressed;
            if (message.encrypted || payload.sensitive) {
                // Get recipient's public key from blockchain or A2A registry
                const recipientKey = await this.getAgentPublicKey(message.toAgentId);
                if (recipientKey) {
                    dataToStore = this.enhancedFeatures.encryptPayload(
                        compressed.data || compressed,
                        recipientKey
                    );
                }
            }

            // Store on blockchain
            const blockchainData = {
                fromAgent: message.fromAgentId,
                toAgent: message.toAgentId,
                messageType: message.messageType,
                data: dataToStore,
                timestamp: Date.now(),
                compressed: compressed.compressed || false,
                encrypted: dataToStore.encrypted || false
            };

            const onChainId = await this.storeOnChain(blockchainData);

            // Send confirmation via A2A
            await this.a2aService.sendMessage(
                message.fromAgentId,
                message.toAgentId,
                "on_chain_confirmation",
                JSON.stringify({
                    originalMessageId: message.id,
                    onChainId: onChainId,
                    txHash: this.lastTxHash
                }),
                "high"
            );

            return {
                success: true,
                storageType: "blockchain",
                onChainId: onChainId
            };

        } catch (error) {
            console.error("On-chain storage failed:", error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Handle standard A2A message with optional blockchain audit
     */
    async handleStandardMessage(message) {
        try {
            // Process through standard A2A
            const result = await this.a2aService.processEnhancedMessage(message);

            // Optionally log hash on-chain for audit trail
            if (message.auditRequired) {
                const auditEntry = {
                    messageId: message.id,
                    fromAgent: message.fromAgentId,
                    toAgent: message.toAgentId,
                    messageHash: this.calculateHash(message.payload),
                    timestamp: Date.now(),
                    dataType: "audit_hash"
                };

                await this.storeOnChain(auditEntry);
            }

            return {
                success: true,
                storageType: "a2a",
                result: result
            };

        } catch (error) {
            console.error("Standard message processing failed:", error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Store data on blockchain via Python service
     */
    async storeOnChain(data) {
        // This would call the Python blockchain_data_exchange service
        // via HTTP API or direct integration
        try {
            const response = await fetch("http://localhost:8000/blockchain/store", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    sender_agent_id: data.fromAgent || data.sender_agent_id,
                    receiver_agent_id: data.toAgent || data.receiver_agent_id,
                    data: data,
                    data_type: data.dataType || "general",
                    is_encrypted: data.encrypted || false,
                    compress: true
                })
            });

            const result = await response.json();
            return result.data_id;

        } catch (error) {
            console.error("Blockchain storage failed:", error);
            // Fallback to storing hash only
            return this.storeHashOnChain(data);
        }
    }

    /**
     * Store only hash on-chain (gas-efficient fallback)
     */
    async storeHashOnChain(data) {
        const dataHash = this.calculateHash(data);
        const account = this.web3.eth.accounts.privateKeyToAccount(this.privateKey);

        // Simple hash storage transaction
        const tx = {
            from: account.address,
            to: this.contractAddress,
            gas: 100000,
            data: this.web3.utils.sha3(JSON.stringify({
                hash: dataHash,
                timestamp: Date.now(),
                agents: [data.fromAgent, data.toAgent]
            }))
        };

        const signedTx = await account.signTransaction(tx);
        const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
        this.lastTxHash = receipt.transactionHash;

        return receipt.blockNumber;
    }

    /**
     * Retrieve data from hybrid storage
     */
    async retrieveData(storageRef) {
        try {
            if (storageRef.storageType === "hybrid") {
                // Get S3 data
                const s3Data = await this.enhancedFeatures.downloadLargeFile(storageRef.s3Key);

                // Verify against blockchain hash
                const onChainData = await this.retrieveFromChain(storageRef.onChainId);
                const dataHash = this.calculateHash(s3Data.data);

                if (dataHash !== onChainData.dataHash) {
                    throw new Error("Data integrity check failed");
                }

                return s3Data.data;

            } else if (storageRef.storageType === "blockchain") {
                // Get directly from blockchain
                return await this.retrieveFromChain(storageRef.onChainId);

            } else {
                // Standard A2A retrieval
                return await this.a2aService.getStoredMessage(storageRef.messageId);
            }

        } catch (error) {
            console.error("Data retrieval failed:", error);
            return null;
        }
    }

    /**
     * Retrieve from blockchain via Python service
     */
    async retrieveFromChain(dataId) {
        try {
            const response = await fetch(`http://localhost:8000/blockchain/retrieve/${dataId}`);
            const result = await response.json();

            // Handle decompression if needed
            if (result.data_type && result.data_type.startsWith("compressed:")) {
                result.data = this.enhancedFeatures.decompressPayload({
                    compressed: true,
                    data: result.data,
                    algorithm: "gzip"
                });
            }

            // Handle decryption if needed
            if (result.is_encrypted) {
                // Would need private key of recipient
                console.log("Data is encrypted - decryption required");
            }

            return result;

        } catch (error) {
            console.error("Blockchain retrieval failed:", error);
            return null;
        }
    }

    /**
     * Get agent's public key from registry
     */
    async getAgentPublicKey(agentId) {
        try {
            // First try A2A registry
            const keyData = await this.a2aService.db.run(
                "SELECT publicKey FROM agent_keys WHERE agentId = ?",
                [agentId]
            );

            if (keyData && keyData.publicKey) {
                return keyData.publicKey;
            }

            // Fallback to blockchain registry
            // This would call smart contract to get public key
            return null;

        } catch (error) {
            console.error("Failed to get agent public key:", error);
            return null;
        }
    }

    /**
     * Calculate hash of data
     */
    calculateHash(data) {
        const crypto = require("crypto");
        const dataString = typeof data === "string" ? data : JSON.stringify(data);
        return crypto.createHash("sha256").update(dataString).digest("hex");
    }

    /**
     * Create workflow combining A2A and blockchain
     */
    async createHybridWorkflow(workflowType, participants, initialData) {
        try {
            // Create blockchain workflow
            const workflowId = await this.createBlockchainWorkflow(workflowType, participants);

            // Store initial data
            const dataId = await this.storeOnChain({
                fromAgent: participants[0],
                toAgent: "WORKFLOW",
                data: initialData,
                dataType: "workflow_init",
                workflowId: workflowId
            });

            // Notify participants via A2A
            for (const participant of participants) {
                await this.a2aService.sendMessage(
                    "WORKFLOW_COORDINATOR",
                    participant,
                    "workflow_invitation",
                    JSON.stringify({
                        workflowId: workflowId,
                        workflowType: workflowType,
                        dataId: dataId,
                        participants: participants
                    }),
                    "high"
                );
            }

            return {
                success: true,
                workflowId: workflowId,
                dataId: dataId
            };

        } catch (error) {
            console.error("Hybrid workflow creation failed:", error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Create blockchain workflow
     */
    async createBlockchainWorkflow(workflowType, participants) {
        const response = await fetch("http://localhost:8000/blockchain/workflow/create", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                workflow_type: workflowType,
                participants: participants
            })
        });

        const result = await response.json();
        return result.workflow_id;
    }
}

module.exports = A2ABlockchainBridge;
