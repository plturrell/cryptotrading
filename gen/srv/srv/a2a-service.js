const cds = require("@sap/cds");
const { v4: uuidv4 } = require("uuid");
const axios = require("axios");

// Import CDS query builders
const { INSERT, SELECT, UPDATE } = cds.ql;

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
                // Create message record
                const message = await INSERT.into(A2AMessages).entries({
                    ID: uuidv4(),
                    fromAgent_ID: fromAgentId,
                    toAgent_ID: toAgentId,
                    messageType,
                    priority: priority || "MEDIUM",
                    payload,
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

                // Calculate average response time (mock for now)
                const avgResponseTime = 250; // milliseconds

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
        // Try WebSocket delivery firs
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
        // Simulate step execution
        return new Promise(resolve => {
            setTimeout(resolve, 1000);
        });
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
