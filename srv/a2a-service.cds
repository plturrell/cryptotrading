using { com.rex.cryptotrading.a2a as a2a } from './a2a-model';

namespace com.rex.cryptotrading.a2a.service;

/**
 * A2A Agent Service - RESTful API for Agent Management
 */
@path: '/api/odata/v4/A2AService'
service A2AService {
    
    // Core Agent Entities
    @odata.draft.enabled
    @cds.redirection.target
    entity A2AAgents as projection on a2a.A2AAgents;
    
    @cds.redirection.target
    entity A2AConnections as projection on a2a.A2AConnections;
    
    @cds.redirection.target
    entity A2AMessages as projection on a2a.A2AMessages;
    
    @odata.draft.enabled
    entity A2AWorkflows as projection on a2a.A2AWorkflows;
    
    @cds.redirection.target
    entity A2AWorkflowExecutions as projection on a2a.A2AWorkflowExecutions;
    
    entity AgentContexts as projection on a2a.AgentContexts;
    
    // Analytics Views
    @readonly
    entity ActiveAgents as projection on a2a.ActiveAgents;
    
    @readonly
    entity RecentMessages as projection on a2a.RecentMessages;
    
    @readonly
    entity RunningWorkflows as projection on a2a.RunningWorkflows;
    
    @readonly
    entity AgentPerformance as projection on a2a.AgentPerformance;
    
    // Agent Management Actions
    action registerAgent(
        agentName: String,
        agentType: String,
        capabilities: String
    ) returns {
        agentId: String;
        status: String;
        message: String;
    };
    
    action connectAgents(
        fromAgentId: String,
        toAgentId: String,
        protocol: String
    ) returns {
        connectionId: String;
        status: String;
        message: String;
    };
    
    action sendMessage(
        fromAgentId: String,
        toAgentId: String,
        messageType: String,
        payload: String,
        priority: String
    ) returns {
        messageId: String;
        status: String;
        deliveryTime: DateTime;
    };
    
    action executeWorkflow(
        workflowId: String,
        inputData: String
    ) returns {
        executionId: String;
        status: String;
        estimatedTime: Integer;
    };
    
    action stopWorkflow(
        executionId: String,
        reason: String
    ) returns {
        success: Boolean;
        finalStatus: String;
        message: String;
    };
    
    // Agent Query Functions
    function getAgentStatus(agentId: String) returns {
        status: String;
        lastHeartbeat: DateTime;
        activeConnections: Integer;
        pendingMessages: Integer;
        runningWorkflows: Integer;
    };
    
    function getAgentMetrics(
        agentId: String,
        period: String
    ) returns {
        messagesProcessed: Integer;
        avgResponseTime: Decimal;
        successRate: Decimal;
        errorCount: Integer;
        uptime: Decimal;
    };
    
    function getWorkflowStatus(executionId: String) returns {
        status: String;
        currentStep: String;
        progress: Decimal;
        estimatedCompletion: DateTime;
        errors: array of String;
    };
    
    function getAgentNetwork() returns array of {
        agentId: String;
        agentName: String;
        connections: array of {
            targetAgent: String;
            status: String;
            protocol: String;
        };
    };
    
    function getMessageQueue(agentId: String) returns array of {
        messageId: String;
        fromAgent: String;
        messageType: String;
        priority: String;
        status: String;
        sentAt: DateTime;
    };
}