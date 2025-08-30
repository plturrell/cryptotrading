const cds = require("@sap/cds");
const cors = require("cors");
const helmet = require("helmet");
const compression = require("compression");
const rateLimit = require("express-rate-limit");
const winston = require("winston");
const WebSocket = require("ws");
const axios = require("axios");

// Import CDS query builder
const { UPDATE } = cds.ql;

// Initialize logger
const logger = winston.createLogger({
    level: "info",
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: "logs/error.log", level: "error" }),
        new winston.transports.File({ filename: "logs/combined.log" }),
        new winston.transports.Console({
            format: winston.format.simple()
        })
    ]
});

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 1000, // limit each IP to 1000 requests per windowMs
    message: "Too many requests from this IP, please try again later."
});

module.exports = cds.server;

cds.on("bootstrap", (app) => {
    // Security middleware
    app.use(helmet({
        contentSecurityPolicy: {
            directives: {
                defaultSrc: ["'self'"],
                scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'", "https://ui5.sap.com", "http:", "ws:"],
                styleSrc: ["'self'", "'unsafe-inline'", "https://ui5.sap.com"],
                fontSrc: ["'self'", "https://ui5.sap.com"],
                imgSrc: ["'self'", "data:", "https:"],
                connectSrc: ["'self'", "https://ui5.sap.com", "http:", "ws:", "wss:"]
            }
        }
    }));

    app.use(compression());
    app.use(cors({
        origin: process.env.ALLOWED_ORIGINS?.split(",") || ["http://localhost:8080"],
        credentials: true
    }));
    app.use(limiter);

    // Health check endpoin
    app.get("/health", (req, res) => {
        res.status(200).json({
            status: "healthy",
            timestamp: new Date().toISOString(),
            uptime: process.uptime(),
            version: require("../package.json").version
        });
    });

    // API documentation endpoin
    app.get("/api-docs", (req, res) => {
        res.json({
            message: "Crypto Trading Platform API",
            version: "1.0.0",
            endpoints: {
                health: "/health",
                marketData: "/api/market-data",
                trading: "/api/trading",
                portfolio: "/api/portfolio",
                analytics: "/api/analytics",
                risk: "/api/risk"
            }
        });
    });

    logger.info("Crypto Trading Platform server initialized");
});

cds.on("listening", ({ server, url }) => {
    logger.info(`Crypto Trading Platform server listening at ${url}`);
    logger.info("Available CDS services:");

    // Get actual registered services
    const services = cds.services;
    Object.keys(services).forEach(serviceName => {
        const service = services[serviceName];
        if (service.path && service.path !== "/") {
            logger.info(`- ${serviceName} → ${service.path}`);
        }
    });

    // Initialize WebSocket server for A2A agent communication
    initWebSocketServer(server);

    // Initialize event bridge
    initEventBridge();

    logger.info("🔗 A2A-CDS Event Bridge initialized");
    logger.info("🔌 WebSocket server ready for agent connections");
});

// ============== WEBSOCKET SERVER ==============

function initWebSocketServer(httpServer) {
    const wss = new WebSocket.Server({
        server: httpServer,
        path: "/a2a/ws"
    });

    wss.on("connection", (ws, req) => {
        logger.info("New WebSocket connection from:", req.socket.remoteAddress);

        ws.on("message", async (message) => {
            try {
                const data = JSON.parse(message);
                await handleWebSocketMessage(ws, data);
            } catch (error) {
                logger.error("WebSocket message error:", error);
                ws.send(JSON.stringify({
                    type: "ERROR",
                    message: error.message
                }));
            }
        });

        ws.on("close", () => {
            // Unregister agent connection
            const a2aService = cds.services["A2AService"];
            if (a2aService && ws.agentId) {
                a2aService.unregisterAgentWebSocket(ws.agentId);
                logger.info(`Agent ${ws.agentId} disconnected`);
            }
        });

        ws.on("error", (error) => {
            logger.error("WebSocket error:", error);
        });
    });

    return wss;
}

async function handleWebSocketMessage(ws, data) {
    const { type, payload } = data;

    switch (type) {
    case "AGENT_REGISTER":
        // Register agent WebSocket connection
        const { agentId } = payload;
        ws.agentId = agentId;

        const a2aService = cds.services["A2AService"];
        if (a2aService) {
            a2aService.registerAgentWebSocket(agentId, ws);
        }

        ws.send(JSON.stringify({
            type: "AGENT_REGISTERED",
            agentId: agentId
        }));

        logger.info(`Agent ${agentId} registered WebSocket connection`);
        break;

    case "HEARTBEAT":
        // Update agent heartbea
        if (ws.agentId) {
            await updateAgentHeartbeat(ws.agentId);
        }

        ws.send(JSON.stringify({
            type: "HEARTBEAT_ACK",
            timestamp: new Date().toISOString()
        }));
        break;

    case "MESSAGE_ACK":
        // Mark message as read
        if (payload.messageId) {
            await markMessageAsRead(payload.messageId);
        }
        break;

    default:
        logger.warn("Unknown WebSocket message type:", type);
    }
}

async function updateAgentHeartbeat(agentId) {
    try {
        const db = await cds.connect.to("db");
        await UPDATE(db.entities["com.rex.cryptotrading.a2a.A2AAgents"])
            .set({ lastHeartbeat: new Date() })
            .where({ ID: agentId });
    } catch (error) {
        logger.error("Failed to update agent heartbeat:", error);
    }
}

async function markMessageAsRead(messageId) {
    try {
        const db = await cds.connect.to("db");
        await UPDATE(db.entities["com.rex.cryptotrading.a2a.A2AMessages"])
            .set({
                status: "READ",
                readAt: new Date()
            })
            .where({ ID: messageId });
    } catch (error) {
        logger.error("Failed to mark message as read:", error);
    }
}

// ============== EVENT BRIDGE ==============

function initEventBridge() {
    // Listen to CDS events and forward to Python agents
    cds.on("agent.*", async (event) => {
        await forwardEventToPythonAgents("agent_event", event);
    });

    cds.on("message.*", async (event) => {
        await forwardEventToPythonAgents("message_event", event);
    });

    cds.on("workflow.*", async (event) => {
        await forwardEventToPythonAgents("workflow_event", event);
    });

    // Set up HTTP endpoint for Python agents to send events to CDS
    const app = cds.app;
    if (app) {
        setupPythonEventEndpoint(app);
    }
}

function setupPythonEventEndpoint(app) {
    // Endpoint for Python agents to send events to CDS
    app.post("/a2a/events", async (req, res) => {
        try {
            const { event, data } = req.body || {};

            logger.info(`Received event from Python agent: ${event}`);

            // Emit event to CDS event system
            await cds.emit(event, data);

            // Forward to WebSocket clients if needed
            await broadcastToWebSocketClients(event, data);

            res.status(200).json({
                status: "SUCCESS",
                message: "Event processed",
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            logger.error("Failed to process Python event:", error);
            res.status(500).json({
                status: "ERROR",
                message: error.message
            });
        }
    });

    // Health check for Python agents
    app.get("/a2a/health", (req, res) => {
        res.json({
            status: "HEALTHY",
            timestamp: new Date().toISOString(),
            services: {
                cds: "RUNNING",
                websocket: "RUNNING",
                eventBridge: "RUNNING"
            }
        });
    });

    logger.info("📡 Python event endpoints configured:");
    logger.info("  POST /a2a/events - Receive events from Python agents");
    logger.info("  GET /a2a/health - Health check for Python agents");
}

async function forwardEventToPythonAgents(eventType, eventData) {
    const pythonAgentUrl = process.env.PYTHON_AGENT_URL || "http://localhost:8000";

    try {
        await axios.post(`${pythonAgentUrl}/cds/events`, {
            event: eventType,
            data: eventData,
            timestamp: new Date().toISOString(),
            source: "CDS"
        }, {
            timeout: 5000
        });

        logger.debug(`Event forwarded to Python agents: ${eventType}`);

    } catch (error) {
        if (error.code !== "ECONNREFUSED") {
            logger.error(`Failed to forward event to Python agents: ${error.message}`);
        }
    }
}

async function broadcastToWebSocketClients(event, data) {
    const a2aService = cds.services["A2AService"];
    if (a2aService && a2aService.wsClients) {
        const message = JSON.stringify({
            type: "EVENT",
            event: event,
            data: data,
            timestamp: new Date().toISOString()
        });

        a2aService.wsClients.forEach((ws) => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(message);
            }
        });
    }
}

// Error handling
process.on("unhandledRejection", (reason, promise) => {
    logger.error("Unhandled Rejection at:", promise, "reason:", reason);
});

process.on("uncaughtException", (error) => {
    logger.error("Uncaught Exception:", error);
    process.exit(1);
});
