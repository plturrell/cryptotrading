namespace com.rex.cryptotrading.a2a;

using { managed, cuid } from '@sap/cds/common';

/**
 * A2A Agents Entity
 * Represents autonomous agents in the system
 */
entity A2AAgents : cuid, managed {
    agentName       : String(100) @title: 'Agent Name';
    agentType       : String(50) @title: 'Agent Type';
    description     : String(500) @title: 'Description';
    status          : String(20) @title: 'Status' @assert.range enum {
        ACTIVE;
        INACTIVE;
        SUSPENDED;
        INITIALIZING;
        ERROR;
    } default 'INACTIVE';
    capabilities    : LargeString @title: 'Capabilities (JSON)';
    configuration   : LargeString @title: 'Configuration (JSON)';
    lastHeartbeat   : Timestamp @title: 'Last Heartbeat';
    version         : String(20) @title: 'Version';
    
    // Navigation
    connections     : Composition of many A2AConnections on connections.fromAgent = $self or connections.toAgent = $self;
    messages        : Composition of many A2AMessages on messages.fromAgent = $self or messages.toAgent = $self;
    workflows       : Association to many A2AWorkflows on workflows.ownerAgent = $self;
}

/**
 * A2A Connections Entity
 * Manages connections between agents
 */
entity A2AConnections : cuid, managed {
    fromAgent       : Association to A2AAgents @title: 'From Agent';
    toAgent         : Association to A2AAgents @title: 'To Agent';
    connectionType  : String(50) @title: 'Connection Type';
    status          : String(20) @title: 'Status' @assert.range enum {
        CONNECTED;
        DISCONNECTED;
        PENDING;
        FAILED;
    } default 'PENDING';
    protocol        : String(50) @title: 'Protocol';
    metadata        : LargeString @title: 'Metadata (JSON)';
    establishedAt   : Timestamp @title: 'Established At';
    lastActivity    : Timestamp @title: 'Last Activity';
}

/**
 * A2A Messages Entity
 * Stores messages between agents
 */
entity A2AMessages : cuid, managed {
    fromAgent       : Association to A2AAgents @title: 'From Agent';
    toAgent         : Association to A2AAgents @title: 'To Agent';
    messageType     : String(50) @title: 'Message Type';
    priority        : String(20) @title: 'Priority' @assert.range enum {
        CRITICAL;
        HIGH;
        MEDIUM;
        LOW;
    } default 'MEDIUM';
    payload         : LargeString @title: 'Message Payload';
    status          : String(20) @title: 'Status' @assert.range enum {
        SENT;
        DELIVERED;
        READ;
        FAILED;
        EXPIRED;
    } default 'SENT';
    sentAt          : Timestamp @title: 'Sent At';
    deliveredAt     : Timestamp @title: 'Delivered At';
    readAt          : Timestamp @title: 'Read At';
    expiresAt       : Timestamp @title: 'Expires At';
    retryCount      : Integer @title: 'Retry Count' default 0;
}

/**
 * A2A Workflows Entity
 * Defines workflows for agent coordination
 */
entity A2AWorkflows : cuid, managed {
    workflowName    : String(100) @title: 'Workflow Name';
    description     : String(500) @title: 'Description';
    ownerAgent      : Association to A2AAgents @title: 'Owner Agent';
    workflowType    : String(50) @title: 'Workflow Type';
    definition      : LargeString @title: 'Workflow Definition (JSON)';
    status          : String(20) @title: 'Status' @assert.range enum {
        DRAFT;
        ACTIVE;
        PAUSED;
        COMPLETED;
        ARCHIVED;
    } default 'DRAFT';
    version         : String(20) @title: 'Version';
    
    // Navigation
    executions      : Composition of many A2AWorkflowExecutions on executions.workflow = $self;
}

/**
 * A2A Workflow Executions Entity
 * Tracks workflow execution instances
 */
entity A2AWorkflowExecutions : cuid, managed {
    workflow        : Association to A2AWorkflows @title: 'Workflow';
    executionId     : String(50) @title: 'Execution ID';
    status          : String(20) @title: 'Status' @assert.range enum {
        RUNNING;
        COMPLETED;
        FAILED;
        CANCELLED;
        TIMEOUT;
    } default 'RUNNING';
    startedAt       : Timestamp @title: 'Started At';
    completedAt     : Timestamp @title: 'Completed At';
    inputData       : LargeString @title: 'Input Data (JSON)';
    outputData      : LargeString @title: 'Output Data (JSON)';
    errorMessage    : String(1000) @title: 'Error Message';
    currentStep     : String(100) @title: 'Current Step';
    totalSteps      : Integer @title: 'Total Steps';
    completedSteps  : Integer @title: 'Completed Steps';
}

/**
 * Agent Contexts Entity
 * Stores contextual information for agents
 */
entity AgentContexts : cuid, managed {
    agent           : Association to A2AAgents @title: 'Agent';
    contextType     : String(50) @title: 'Context Type';
    contextKey      : String(100) @title: 'Context Key';
    contextValue    : LargeString @title: 'Context Value';
    metadata        : LargeString @title: 'Metadata (JSON)';
    validFrom       : Timestamp @title: 'Valid From';
    validTo         : Timestamp @title: 'Valid To';
    isActive        : Boolean @title: 'Is Active' default true;
}

// Analytics Views
view ActiveAgents as select from A2AAgents {
    *
} where status = 'ACTIVE';

view RecentMessages as select from A2AMessages {
    *
} where createdAt >= $now - 86400000; // Last 24 hours

view RunningWorkflows as select from A2AWorkflowExecutions {
    *
} where status = 'RUNNING';

view AgentPerformance as select from A2AAgents {
    ID,
    agentName,
    agentType,
    status,
    count(messages.ID) as totalMessages : Integer,
    count(workflows.ID) as totalWorkflows : Integer
} group by ID, agentName, agentType, status;