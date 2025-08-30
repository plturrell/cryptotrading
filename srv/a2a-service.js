const cds = require("@sap/cds");
const { v4: uuidv4 } = require("uuid");
const axios = require("axios");
const zlib = require("zlib");
const crypto = require("crypto");
// AWS SDK would be imported here if needed
// const AWS = require("aws-sdk");
// File system utilities removed - not currently used

// Import CDS query builders
const { INSERT, SELECT, UPDATE } = cds.ql;

// S3 client would be initialized here if AWS SDK is available
let s3 = null;
try {
    const AWS = require("aws-sdk");
    s3 = new AWS.S3({
        region: process.env.AWS_REGION || "us-east-1",
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
    });
} catch (e) {
    // AWS SDK not available - S3 features disabled
}

/**
 * A2A Agent Service Implementation
 * Provides seamless integration between CDS and A2A agents
 */
module.exports = class A2AService extends cds.ApplicationService {

    async init() {
        const {
            A2AAgents,
            A2AConnections,
            A2AMessages,
            A2AWorkflows,
            A2AWorkflowExecutions
        } = this.entities;
        // Initialize message queue for async processing
        this.messageQueue = [];
        this.agentRegistry = new Map();
        this.activeConnections = new Map();

        // Initialize enhanced A2A features
        this.chunkStore = new Map(); // For message chunking
        this.encryptionKeys = new Map(); // For agent public keys
        this.transactionStore = new Map(); // For distributed transactions
        this.compressionThreshold = 10240; // Compress payloads > 10KB

        // Initialize blockchain bridge for hybrid storage
        const A2ABlockchainBridge = require("./a2a-blockchain-bridge");
        this.blockchainBridge = new A2ABlockchainBridge(this, {
            rpcUrl: process.env.BLOCKCHAIN_RPC_URL || "http://localhost:8545",
            contractAddress: process.env.A2A_CONTRACT_ADDRESS,
            privateKey: process.env.A2A_PRIVATE_KEY
        });
        this.s3Bucket = s3 ? (process.env.A2A_S3_BUCKET || "cryptotrading-a2a-files") : null;

        // Initialize WebSocket server for real-time agent communication
        this.initWebSocketServer();

        // ============== AGENT MANAGEMENT ==============

        // Register a new agen
        this.on("registerAgent", async (req) => {
            const { agentName, agentType, capabilities } = req.data;

            try {
                // Create agent record in database
                const agent = await INSERT.into(A2AAgents).entries({
                    ID: uuidv4(),
                    agentName,
                    agentType,
                    capabilities: JSON.stringify(capabilities),
                    status: "INITIALIZING",
                    lastHeartbeat: new Date(),
                    version: "1.0"
                });

                // Register in memory for fast lookup
                this.agentRegistry.set(agent.ID, {
                    ...agent,
                    capabilities: JSON.parse(agent.capabilities || "[]"),
                    wsConnection: null
                });

                // Emit registration even
                await this.emit("agent.registered", {
                    agentId: agent.ID,
                    agentName,
                    agentType,
                    timestamp: new Date().toISOString()
                });

                // Notify Python agents via HTTP callback
                await this.notifyPythonAgents("agent_registered", {
                    agentId: agent.ID,
                    agentName,
                    agentType
                });

                return {
                    agentId: agent.ID,
                    status: "SUCCESS",
                    message: `Agent ${agentName} registered successfully`
                };

            } catch (error) {
                cds.log("error", "Agent registration failed:", error);
                return {
                    agentId: null,
                    status: "ERROR",
                    message: error.message
                };
            }
        });

        // Connect two agents
        this.on("connectAgents", async (req) => {
            const { fromAgentId, toAgentId, protocol } = req.data;

            try {
                // Verify both agents exis
                const [fromAgent, toAgent] = await Promise.all([
                    SELECT.one.from(A2AAgents).where({ ID: fromAgentId }),
                    SELECT.one.from(A2AAgents).where({ ID: toAgentId })
                ]);

                if (!fromAgent || !toAgent) {
                    throw new Error("One or both agents not found");
                }

                // Create connection record
                const connection = await INSERT.into(A2AConnections).entries({
                    ID: uuidv4(),
                    fromAgent_ID: fromAgentId,
                    toAgent_ID: toAgentId,
                    connectionType: "BIDIRECTIONAL",
                    status: "PENDING",
                    protocol: protocol || "A2A",
                    establishedAt: new Date(),
                    lastActivity: new Date()
                });

                // Store in active connections
                const connectionKey = `${fromAgentId}-${toAgentId}`;
                this.activeConnections.set(connectionKey, connection);

                // Emit connection even
                await this.emit("agents.connected", {
                    connectionId: connection.ID,
                    fromAgentId,
                    toAgentId,
                    protocol
                });

                // Update connection status to CONNECTED
                await UPDATE(A2AConnections)
                    .set({ status: "CONNECTED" })
                    .where({ ID: connection.ID });

                return {
                    connectionId: connection.ID,
                    status: "SUCCESS",
                    message: "Agents connected successfully"
                };

            } catch (error) {
                cds.log("error", "Agent connection failed:", error);
                return {
                    connectionId: null,
                    status: "ERROR",
                    message: error.message
                };
            }
        });

        // ============== MESSAGE HANDLING ==============

        // Send message between agents
        this.on("sendMessage", async (req) => {
            const { fromAgentId, toAgentId, messageType, payload, priority } = req.data;

            try {
                let processedPayload = payload;

                // ENHANCED FEATURE: Auto-compress large payloads
                if (payload && payload.length > this.compressionThreshold) {
                    const compressed = zlib.gzipSync(payload);
                    processedPayload = JSON.stringify({
                        compressed: true,
                        algorithm: "gzip",
                        data: compressed.toString("base64"),
                        originalSize: payload.length,
                        compressedSize: compressed.length
                    });
                }

                // ENHANCED FEATURE: Handle chunking for very large messages
                if (processedPayload.length > 524288) { // > 512KB
                    const correlationId = uuidv4();
                    const chunks = [];
                    const chunkSize = 65536; // 64KB chunks

                    for (let i = 0; i < processedPayload.length; i += chunkSize) {
                        chunks.push(processedPayload.slice(i, i + chunkSize));
                    }

                    // Send all chunks
                    for (let i = 0; i < chunks.length; i++) {
                        const chunkType = i === 0 ? "CHUNK_START"
                            : i === chunks.length - 1 ? "CHUNK_END"
                                : "CHUNK_DATA";

                        await INSERT.into(A2AMessages).entries({
                            ID: uuidv4(),
                            fromAgent_ID: fromAgentId,
                            toAgent_ID: toAgentId,
                            messageType: chunkType,
                            priority: priority || "MEDIUM",
                            payload: JSON.stringify({
                                correlationId: correlationId,
                                chunkIndex: i,
                                totalChunks: chunks.length,
                                originalType: messageType,
                                data: chunks[i]
                            }),
                            status: "SENT",
                            sentAt: new Date(),
                            retryCount: 0
                        });
                    }

                    return {
                        messageId: correlationId,
                        status: "CHUNKED",
                        chunks: chunks.length,
                        deliveryTime: new Date()
                    };
                }

                // Create message record
                const message = await INSERT.into(A2AMessages).entries({
                    ID: uuidv4(),
                    fromAgent_ID: fromAgentId,
                    toAgent_ID: toAgentId,
                    messageType,
                    priority: priority || "MEDIUM",
                    payload: processedPayload,
                    status: "SENT",
                    sentAt: new Date(),
                    retryCount: 0
                });

                // Add to message queue for processing
                this.messageQueue.push({
                    ...message,
                    processingAttempts: 0
                });

                // Process message asynchronously
                setImmediate(() => this.processMessageQueue());

                // Emit message even
                await this.emit("message.sent", {
                    messageId: message.ID,
                    fromAgentId,
                    toAgentId,
                    messageType
                });

                // Send via WebSocket if agent is connected
                const toAgentInfo = this.agentRegistry.get(toAgentId);
                if (toAgentInfo?.wsConnection) {
                    toAgentInfo.wsConnection.send(JSON.stringify({
                        type: "MESSAGE",
                        data: message
                    }));

                    // Update status to DELIVERED
                    await UPDATE(A2AMessages)
                        .set({
                            status: "DELIVERED",
                            deliveredAt: new Date()
                        })
                        .where({ ID: message.ID });
                }

                return {
                    messageId: message.ID,
                    status: "SUCCESS",
                    deliveryTime: new Date()
                };

            } catch (error) {
                cds.log("error", "Message send failed:", error);
                return {
                    messageId: null,
                    status: "ERROR",
                    deliveryTime: null
                };
            }
        });

        // ============== ENHANCED A2A FEATURES ==============

        // Send large file via S3
        this.on("sendLargeFile", async (req) => {
            const { fromAgentId, toAgentId, fileName, fileData, contentType } = req.data;

            if (!s3) {
                return { success: false, error: "S3 not configured" };
            }

            try {
                const s3Key = `a2a-transfers/${fromAgentId}/${Date.now()}-${fileName}`;

                const uploadParams = {
                    Bucket: this.s3Bucket,
                    Key: s3Key,
                    Body: Buffer.from(fileData, "base64"),
                    ContentType: contentType || "application/octet-stream",
                    Metadata: {
                        fromAgent: fromAgentId,
                        toAgent: toAgentId,
                        timestamp: new Date().toISOString()
                    }
                };

                const uploadResult = await s3.upload(uploadParams).promise();

                // Send notification message with S3 URL
                const message = await INSERT.into(A2AMessages).entries({
                    ID: uuidv4(),
                    fromAgent_ID: fromAgentId,
                    toAgent_ID: toAgentId,
                    messageType: "FILE_TRANSFER",
                    priority: "HIGH",
                    payload: JSON.stringify({
                        s3Url: uploadResult.Location,
                        s3Key: s3Key,
                        fileName: fileName,
                        fileSize: Buffer.byteLength(fileData, "base64"),
                        contentType: contentType,
                        checksum: crypto.createHash("sha256").update(fileData).digest("hex")
                    }),
                    status: "SENT",
                    sentAt: new Date(),
                    retryCount: 0
                });

                return {
                    success: true,
                    s3Url: uploadResult.Location,
                    messageId: message.ID
                };

            } catch (error) {
                cds.log("error", "Large file transfer failed:", error);
                return { success: false, error: error.message };
            }
        });

        // Send encrypted message
        this.on("sendEncryptedMessage", async (req) => {
            const { fromAgentId, toAgentId, messageType, payload, priority, recipientPublicKey } = req.data;

            try {
                // Store public key if provided
                if (recipientPublicKey) {
                    this.encryptionKeys.set(toAgentId, recipientPublicKey);
                }

                const publicKey = this.encryptionKeys.get(toAgentId);
                if (!publicKey) {
                    throw new Error("Recipient public key not available");
                }

                // Generate AES key for this message
                const aesKey = crypto.randomBytes(32);
                const iv = crypto.randomBytes(16);

                // Encrypt payload with AES
                const cipher = crypto.createCipheriv("aes-256-gcm", aesKey, iv);
                let encrypted = cipher.update(payload, "utf8", "hex");
                encrypted += cipher.final("hex");
                const authTag = cipher.getAuthTag();

                // Encrypt AES key with recipient's public key
                const encryptedKey = crypto.publicEncrypt(
                    publicKey,
                    aesKey
                ).toString("base64");

                // Send encrypted message
                const message = await INSERT.into(A2AMessages).entries({
                    ID: uuidv4(),
                    fromAgent_ID: fromAgentId,
                    toAgent_ID: toAgentId,
                    messageType: messageType,
                    priority: priority || "HIGH",
                    payload: JSON.stringify({
                        encrypted: true,
                        algorithm: "aes-256-gcm",
                        data: encrypted,
                        key: encryptedKey,
                        iv: iv.toString("base64"),
                        authTag: authTag.toString("base64")
                    }),
                    status: "SENT",
                    sentAt: new Date(),
                    retryCount: 0
                });

                return {
                    messageId: message.ID,
                    encrypted: true,
                    status: "SUCCESS"
                };

            } catch (error) {
                cds.log("error", "Encrypted message send failed:", error);
                return { messageId: null, encrypted: false, error: error.message };
            }
        });

        // Register agent public key
        this.on("registerPublicKey", async (req) => {
            const { agentId, publicKey } = req.data;

            try {
                this.encryptionKeys.set(agentId, publicKey);

                // Update agent record with key hash
                await UPDATE(A2AAgents)
                    .set({
                        configuration: JSON.stringify({
                            publicKeyHash: crypto.createHash("sha256").update(publicKey).digest("hex")
                        })
                    })
                    .where({ ID: agentId });

                return {
                    success: true,
                    agentId: agentId,
                    keyHash: crypto.createHash("sha256").update(publicKey).digest("hex")
                };

            } catch (error) {
                cds.log("error", "Public key registration failed:", error);
                return { success: false, error: error.message };
            }
        });

        // ============== WORKFLOW MANAGEMENT ==============

        // Execute workflow
        this.on("executeWorkflow", async (req) => {
            const { workflowId, inputData } = req.data;

            try {
                // Get workflow definition
                const workflow = await SELECT.one.from(A2AWorkflows)
                    .where({ ID: workflowId });

                if (!workflow) {
                    throw new Error("Workflow not found");
                }

                // Create execution record
                const execution = await INSERT.into(A2AWorkflowExecutions).entries({
                    ID: uuidv4(),
                    workflow_ID: workflowId,
                    executionId: `EXEC-${Date.now()}`,
                    status: "RUNNING",
                    startedAt: new Date(),
                    inputData,
                    currentStep: "INITIALIZATION",
                    totalSteps: JSON.parse(workflow.definition || "{}").steps?.length || 0,
                    completedSteps: 0
                });

                // Start workflow execution asynchronously
                this.executeWorkflowAsync(execution.ID, workflow, inputData);

                // Emit workflow started even
                await this.emit("workflow.started", {
                    executionId: execution.ID,
                    workflowId,
                    timestamp: new Date().toISOString()
                });

                return {
                    executionId: execution.ID,
                    status: "STARTED",
                    estimatedTime: 60 // seconds
                };

            } catch (error) {
                cds.log("error", "Workflow execution failed:", error);
                return {
                    executionId: null,
                    status: "ERROR",
                    estimatedTime: 0
                };
            }
        });

        // Stop workflow
        this.on("stopWorkflow", async (req) => {
            const { executionId, reason } = req.data;

            try {
                // Update execution status
                await UPDATE(A2AWorkflowExecutions)
                    .set({
                        status: "CANCELLED",
                        completedAt: new Date(),
                        errorMessage: reason
                    })
                    .where({ ID: executionId });

                // Emit workflow stopped even
                await this.emit("workflow.stopped", {
                    executionId,
                    reason,
                    timestamp: new Date().toISOString()
                });

                return {
                    success: true,
                    finalStatus: "CANCELLED",
                    message: `Workflow stopped: ${reason}`
                };

            } catch (error) {
                cds.log("error", "Workflow stop failed:", error);
                return {
                    success: false,
                    finalStatus: "ERROR",
                    message: error.message
                };
            }
        });

        // ============== QUERY FUNCTIONS ==============

        // Get agent status
        this.on("getAgentStatus", async (req) => {
            const { agentId } = req.data;

            try {
                const agent = await SELECT.one.from(A2AAgents)
                    .where({ ID: agentId });

                if (!agent) {
                    throw new Error("Agent not found");
                }

                // Count active connections
                const connections = await SELECT.from(A2AConnections)
                    .where({
                        or: [
                            { fromAgent_ID: agentId },
                            { toAgent_ID: agentId }
                        ],
                        status: "CONNECTED"
                    });

                // Count pending messages
                const pendingMessages = await SELECT.from(A2AMessages)
                    .where({
                        toAgent_ID: agentId,
                        status: { in: ["SENT", "DELIVERED"] }
                    });

                // Count running workflows
                const runningWorkflows = await SELECT.from(A2AWorkflowExecutions)
                    .columns("workflow_ID")
                    .join(A2AWorkflows)
                    .on("A2AWorkflowExecutions.workflow_ID = A2AWorkflows.ID")
                    .where({
                        "A2AWorkflows.ownerAgent_ID": agentId,
                        "A2AWorkflowExecutions.status": "RUNNING"
                    });

                return {
                    status: agent.status,
                    lastHeartbeat: agent.lastHeartbeat,
                    activeConnections: connections.length,
                    pendingMessages: pendingMessages.length,
                    runningWorkflows: runningWorkflows.length
                };

            } catch (error) {
                cds.log("error", "Get agent status failed:", error);
                throw error;
            }
        });

        // Get agent metrics
        this.on("getAgentMetrics", async (req) => {
            const { agentId, period } = req.data;

            try {
                // Calculate time window
                const now = new Date();
                const startTime = new Date(now - this.parsePeriod(period));

                // Get message statistics
                const messages = await SELECT.from(A2AMessages)
                    .where({
                        or: [
                            { fromAgent_ID: agentId },
                            { toAgent_ID: agentId }
                        ],
                        createdAt: { ">=": startTime }
                    });

                const successfulMessages = messages.filter(m =>
                    m.status === "READ" || m.status === "DELIVERED"
                ).length;

                const failedMessages = messages.filter(m =>
                    m.status === "FAILED" || m.status === "EXPIRED"
                ).length;

                // Calculate metrics
                const totalMessages = messages.length;
                const successRate = totalMessages > 0 ?
                    (successfulMessages / totalMessages) * 100 : 100;

                // Calculate average response time from actual data
                const responseMessages = await SELECT.from("A2AMessages")
                    .columns("responseTime")
                    .where({ status: "DELIVERED", receiver: agentId })
                    .limit(100);

                const avgResponseTime = responseMessages.length > 0
                    ? responseMessages.reduce((sum, m) => sum + (m.responseTime || 250), 0) / responseMessages.length
                    : 250; // Default if no data

                // Calculate uptime
                const agent = await SELECT.one.from(A2AAgents)
                    .where({ ID: agentId });

                const uptime = agent.createdAt ?
                    ((now - new Date(agent.createdAt)) / 1000 / 3600).toFixed(2) : 0;

                return {
                    messagesProcessed: totalMessages,
                    avgResponseTime,
                    successRate: successRate.toFixed(2),
                    errorCount: failedMessages,
                    uptime: parseFloat(uptime)
                };

            } catch (error) {
                cds.log("error", "Get agent metrics failed:", error);
                throw error;
            }
        });

        // Get workflow status
        this.on("getWorkflowStatus", async (req) => {
            const { executionId } = req.data;

            try {
                const execution = await SELECT.one.from(A2AWorkflowExecutions)
                    .where({ ID: executionId });

                if (!execution) {
                    throw new Error("Execution not found");
                }

                const progress = execution.totalSteps > 0 ?
                    (execution.completedSteps / execution.totalSteps) * 100 : 0;

                // Estimate completion time
                const estimatedCompletion = execution.status === "RUNNING" ?
                    new Date(Date.now() + 60000) : execution.completedAt;

                return {
                    status: execution.status,
                    currentStep: execution.currentStep,
                    progress: progress.toFixed(2),
                    estimatedCompletion,
                    errors: execution.errorMessage ? [execution.errorMessage] : []
                };

            } catch (error) {
                cds.log("error", "Get workflow status failed:", error);
                throw error;
            }
        });

        // Get agent network
        this.on("getAgentNetwork", async () => {
            try {
                const agents = await SELECT.from(A2AAgents)
                    .where({ status: "ACTIVE" });

                const network = await Promise.all(agents.map(async (agent) => {
                    const connections = await SELECT.from(A2AConnections)
                        .where({
                            or: [
                                { fromAgent_ID: agent.ID },
                                { toAgent_ID: agent.ID }
                            ],
                            status: "CONNECTED"
                        });

                    return {
                        agentId: agent.ID,
                        agentName: agent.agentName,
                        connections: connections.map(conn => ({
                            targetAgent: conn.fromAgent_ID === agent.ID ?
                                conn.toAgent_ID : conn.fromAgent_ID,
                            status: conn.status,
                            protocol: conn.protocol
                        }))
                    };
                }));

                return network;

            } catch (error) {
                cds.log("error", "Get agent network failed:", error);
                throw error;
            }
        });

        // Get message queue
        this.on("getMessageQueue", async (req) => {
            const { agentId } = req.data;

            try {
                const messages = await SELECT.from(A2AMessages)
                    .where({
                        toAgent_ID: agentId,
                        status: { in: ["SENT", "DELIVERED"] }
                    })
                    .orderBy("priority desc", "sentAt asc")
                    .limit(100);

                return messages.map(msg => ({
                    messageId: msg.ID,
                    fromAgent: msg.fromAgent_ID,
                    messageType: msg.messageType,
                    priority: msg.priority,
                    status: msg.status,
                    sentAt: msg.sentA
                }));

            } catch (error) {
                cds.log("error", "Get message queue failed:", error);
                throw error;
            }
        });

        // ============== EVENT HANDLERS ==============

        // Handle agent heartbea
        this.on("UPDATE", A2AAgents, async (req, next) => {
            // Update last heartbeat timestamp
            if (req.data.status === "ACTIVE") {
                req.data.lastHeartbeat = new Date();
            }
            return next();
        });

        // Handle message status updates
        this.after("UPDATE", A2AMessages, async (data) => {
            if (data.status === "READ") {
                await this.emit("message.read", {
                    messageId: data.ID,
                    readAt: data.readA
                });
            }
        });

        // Handle workflow completion
        this.after("UPDATE", A2AWorkflowExecutions, async (data) => {
            if (data.status === "COMPLETED" || data.status === "FAILED") {
                await this.emit("workflow.completed", {
                    executionId: data.ID,
                    status: data.status,
                    completedAt: data.completedA
                });
            }
        });

        // ENHANCED FEATURE: Hybrid blockchain/S3 storage
        this.on("storeHybridData", async (req) => {
            const { fromAgentId, toAgentId, data, dataType, priority, requiresAudit } = req.data;

            try {
                // Create message object
                const message = {
                    id: uuidv4(),
                    fromAgentId,
                    toAgentId,
                    messageType: dataType || "hybrid_data",
                    payload: JSON.stringify(data),
                    priority: priority || "medium",
                    auditRequired: requiresAudit || false
                };

                // Process through blockchain bridge
                const result = await this.blockchainBridge.processMessage(message);

                if (result.success) {
                    // Store reference in A2A database
                    await INSERT.into(A2AMessages).entries({
                        ID: message.id,
                        fromAgent_ID: fromAgentId,
                        toAgent_ID: toAgentId,
                        messageType: `hybrid:${result.storageType}`,
                        payload: JSON.stringify({
                            storageRef: result,
                            originalData: data.length > 1000 ? "[LARGE_DATA]" : data
                        }),
                        status: "STORED",
                        priority: priority || "medium"
                    });

                    return {
                        success: true,
                        messageId: message.id,
                        storageType: result.storageType,
                        s3Url: result.s3Url,
                        onChainId: result.onChainId
                    };
                }

                return { success: false, error: result.error };

            } catch (error) {
                cds.log("error", "Hybrid storage failed:", error);
                return { success: false, error: error.message };
            }
        });

        // ENHANCED FEATURE: Retrieve hybrid data
        this.on("retrieveHybridData", async (req) => {
            const { messageId, storageRef } = req.data;

            try {
                let ref = storageRef;

                // If only messageId provided, get storage reference from database
                if (!ref && messageId) {
                    const message = await SELECT.one.from(A2AMessages)
                        .where({ ID: messageId });

                    if (message && message.payload) {
                        const payload = JSON.parse(message.payload);
                        ref = payload.storageRef;
                    }
                }

                if (!ref) {
                    return { success: false, error: "Storage reference not found" };
                }

                // Retrieve through blockchain bridge
                const data = await this.blockchainBridge.retrieveData(ref);

                if (data) {
                    return {
                        success: true,
                        data: data,
                        storageType: ref.storageType
                    };
                }

                return { success: false, error: "Data retrieval failed" };

            } catch (error) {
                cds.log("error", "Hybrid retrieval failed:", error);
                return { success: false, error: error.message };
            }
        });

        // ENHANCED FEATURE: Create hybrid workflow
        this.on("createHybridWorkflow", async (req) => {
            const { workflowType, participants, initialData } = req.data;

            try {
                const result = await this.blockchainBridge.createHybridWorkflow(
                    workflowType,
                    participants,
                    initialData
                );

                if (result.success) {
                    // Store workflow reference in A2A database
                    await INSERT.into(A2AWorkflows).entries({
                        ID: uuidv4(),
                        name: `Hybrid: ${workflowType}`,
                        type: "HYBRID",
                        config: JSON.stringify({
                            blockchainWorkflowId: result.workflowId,
                            dataId: result.dataId,
                            participants: participants
                        }),
                        createdBy: participants[0],
                        isActive: true
                    });

                    return result;
                }

                return { success: false, error: result.error };

            } catch (error) {
                cds.log("error", "Hybrid workflow creation failed:", error);
                return { success: false, error: error.message };
            }
        });

        await super.init();
    }

    // ============== HELPER METHODS ==============

    initWebSocketServer() {
        // WebSocket server will be initialized in server.js
        this.wsClients = new Map();
    }

    async processMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();

            try {
                // Attempt to deliver message
                await this.deliverMessage(message);
            } catch (error) {
                cds.log("error", "Message delivery failed:", error);

                // Retry logic
                message.processingAttempts++;
                if (message.processingAttempts < 3) {
                    // Re-queue for retry
                    setTimeout(() => {
                        this.messageQueue.push(message);
                        this.processMessageQueue();
                    }, 5000 * message.processingAttempts);
                } else {
                    // Mark as failed
                    await UPDATE(this.entities.A2AMessages)
                        .set({ status: "FAILED" })
                        .where({ ID: message.ID });
                }
            }
        }
    }

    async deliverMessage(message) {
        // ENHANCED FEATURE: Handle chunk reassembly
        if (message.messageType === "CHUNK_START" ||
            message.messageType === "CHUNK_DATA" ||
            message.messageType === "CHUNK_END") {

            const chunkData = JSON.parse(message.payload);
            const correlationId = chunkData.correlationId;

            if (!this.chunkStore.has(correlationId)) {
                this.chunkStore.set(correlationId, {
                    chunks: [],
                    originalType: chunkData.originalType,
                    receivedAt: Date.now()
                });
            }

            const chunkInfo = this.chunkStore.get(correlationId);
            chunkInfo.chunks[chunkData.chunkIndex] = chunkData.data;

            // If this is the last chunk, reassemble
            if (message.messageType === "CHUNK_END") {
                const reassembled = chunkInfo.chunks.join("");
                this.chunkStore.delete(correlationId);

                // Create reassembled message
                message = {
                    ...message,
                    messageType: chunkInfo.originalType,
                    payload: reassembled
                };
            } else {
                return; // Wait for more chunks
            }
        }

        // ENHANCED FEATURE: Handle decompression
        try {
            const payloadData = JSON.parse(message.payload);
            if (payloadData.compressed) {
                const compressed = Buffer.from(payloadData.data, "base64");
                const decompressed = zlib.gunzipSync(compressed);
                message.payload = decompressed.toString("utf8");
            }
        } catch (e) {
            // Payload is not JSON or not compressed, use as-is
        }

        // Try WebSocket delivery first
        const toAgent = this.agentRegistry.get(message.toAgent_ID);
        if (toAgent?.wsConnection) {
            toAgent.wsConnection.send(JSON.stringify({
                type: "MESSAGE",
                data: message
            }));
            return;
        }

        // Fallback to HTTP callback
        await this.notifyPythonAgents("message_received", message);
    }

    async notifyPythonAgents(eventType, data) {
        // Notify Python agents via HTTP
        try {
            await axios.post("http://localhost:8000/a2a/events", {
                event: eventType,
                data: data,
                timestamp: new Date().toISOString()
            });
        } catch (error) {
            cds.log("error", "Failed to notify Python agents:", error.message);
        }
    }

    async executeWorkflowAsync(executionId, workflow, inputData) {
        try {
            const definition = JSON.parse(workflow.definition || "{}");
            const steps = definition.steps || [];

            for (let i = 0; i < steps.length; i++) {
                const step = steps[i];

                // Update current step
                await UPDATE(this.entities.A2AWorkflowExecutions)
                    .set({
                        currentStep: step.name,
                        completedSteps: i
                    })
                    .where({ ID: executionId });

                // Execute step (simplified)
                await this.executeWorkflowStep(step, inputData);

                // Check if workflow was cancelled
                const execution = await SELECT.one
                    .from(this.entities.A2AWorkflowExecutions)
                    .where({ ID: executionId });

                if (execution.status === "CANCELLED") {
                    break;
                }
            }

            // Mark as completed
            await UPDATE(this.entities.A2AWorkflowExecutions)
                .set({
                    status: "COMPLETED",
                    completedAt: new Date(),
                    completedSteps: steps.length
                })
                .where({ ID: executionId });

        } catch (error) {
            cds.log("error", "Workflow execution error:", error);

            await UPDATE(this.entities.A2AWorkflowExecutions)
                .set({
                    status: "FAILED",
                    completedAt: new Date(),
                    errorMessage: error.message
                })
                .where({ ID: executionId });
        }
    }

    async executeWorkflowStep() {
        // Execute actual workflow step
        // This would integrate with the actual agent operations
        const startTime = Date.now();

        try {
            // Execute the step logic here
            // For now, we'll do a basic processing delay
            // In production, this would call actual agent methods
            await new Promise(resolve => setTimeout(resolve, 100));

            // Return execution time
            return {
                success: true,
                executionTime: Date.now() - startTime
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                executionTime: Date.now() - startTime
            };
        }
    }

    parsePeriod(period) {
        const units = {
            "1h": 3600000,
            "24h": 86400000,
            "7d": 604800000,
            "30d": 2592000000
        };
        return units[period] || 86400000; // Default to 24h
    }

    // Register WebSocket connection for an agen
    registerAgentWebSocket(agentId, ws) {
        const agent = this.agentRegistry.get(agentId);
        if (agent) {
            agent.wsConnection = ws;
            this.wsClients.set(agentId, ws);
        }
    }

    // Unregister WebSocket connection
    unregisterAgentWebSocket(agentId) {
        const agent = this.agentRegistry.get(agentId);
        if (agent) {
            agent.wsConnection = null;
        }
        this.wsClients.delete(agentId);
    }
};
