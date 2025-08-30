// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./A2ARegistry.sol";

/**
 * @title A2A Messaging Protocol
 * @notice On-chain messaging system for A2A agent communication
 * @dev Implements secure, auditable agent-to-agent messaging with routing and validation
 */
contract A2AMessaging {
    
    // Message types matching A2A protocol
    enum MessageType {
        DATA_LOAD_REQUEST,
        DATA_LOAD_RESPONSE,
        ANALYSIS_REQUEST,
        ANALYSIS_RESPONSE,
        DATA_QUERY,
        DATA_QUERY_RESPONSE,
        TRADE_EXECUTION,
        TRADE_RESPONSE,
        WORKFLOW_REQUEST,
        WORKFLOW_RESPONSE,
        WORKFLOW_STATUS,
        HEARTBEAT,
        ERROR,
        MEMORY_SHARE,
        MEMORY_REQUEST,
        MEMORY_RESPONSE
    }
    
    // Message priority levels
    enum Priority {
        LOW,
        NORMAL,
        HIGH,
        CRITICAL
    }
    
    // Message structure
    struct Message {
        uint256 messageId;
        string senderId;
        string receiverId;
        MessageType messageType;
        string payload; // JSON or IPFS hash for large payloads
        Priority priority;
        uint256 timestamp;
        uint256 expiresAt;
        bool isProcessed;
        string responseHash; // IPFS hash of response if any
        uint256 gasUsed;
    }
    
    // Workflow context for multi-step operations
    struct WorkflowContext {
        string workflowId;
        string[] participatingAgents;
        uint256 currentStep;
        uint256 totalSteps;
        bool isActive;
        uint256 startedAt;
        uint256 completedAt;
    }
    
    // State variables
    A2ARegistry public registry;
    mapping(uint256 => Message) public messages;
    mapping(string => uint256[]) public agentInbox; // agentId => messageIds
    mapping(string => uint256[]) public agentOutbox; // agentId => messageIds
    mapping(string => WorkflowContext) public workflows;
    mapping(string => mapping(string => bool)) public trustedPairs; // sender => receiver => trusted
    
    uint256 public messageCounter;
    uint256 public constant MAX_PAYLOAD_SIZE = 1024; // For on-chain storage
    uint256 public constant MESSAGE_RETENTION_DAYS = 30;
    
    // Events
    event MessageSent(
        uint256 indexed messageId,
        string indexed senderId,
        string indexed receiverId,
        MessageType messageType,
        Priority priority,
        uint256 timestamp
    );
    
    event MessageProcessed(
        uint256 indexed messageId,
        string indexed processorId,
        string responseHash,
        uint256 timestamp
    );
    
    event WorkflowStarted(
        string indexed workflowId,
        string[] agents,
        uint256 totalSteps,
        uint256 timestamp
    );
    
    event WorkflowCompleted(
        string indexed workflowId,
        uint256 completedAt,
        uint256 totalGasUsed
    );
    
    event TrustEstablished(
        string indexed senderId,
        string indexed receiverId,
        uint256 timestamp
    );
    
    // Modifiers
    modifier onlyActiveAgent(string memory agentId) {
        require(registry.isAgentActive(agentId), "Agent not active");
        _;
    }
    
    modifier onlyAgentWallet(string memory agentId) {
        A2ARegistry.Agent memory agent = registry.getAgent(agentId);
        require(msg.sender == agent.walletAddress, "Not agent wallet");
        _;
    }
    
    constructor(address _registryAddress) {
        registry = A2ARegistry(_registryAddress);
    }
    
    /**
     * @notice Send a message from one agent to another
     * @param senderId ID of sending agent
     * @param receiverId ID of receiving agent
     * @param messageType Type of message
     * @param payload Message payload (JSON or IPFS hash)
     * @param priority Message priority
     * @param expiresInHours Hours until message expires (0 for no expiry)
     */
    function sendMessage(
        string memory senderId,
        string memory receiverId,
        MessageType messageType,
        string memory payload,
        Priority priority,
        uint256 expiresInHours
    ) external onlyAgentWallet(senderId) onlyActiveAgent(senderId) onlyActiveAgent(receiverId) returns (uint256) {
        
        // Validate payload size for on-chain storage
        require(bytes(payload).length <= MAX_PAYLOAD_SIZE || _isIPFSHash(payload), 
                "Payload too large - use IPFS");
        
        messageCounter++;
        
        Message storage newMessage = messages[messageCounter];
        newMessage.messageId = messageCounter;
        newMessage.senderId = senderId;
        newMessage.receiverId = receiverId;
        newMessage.messageType = messageType;
        newMessage.payload = payload;
        newMessage.priority = priority;
        newMessage.timestamp = block.timestamp;
        newMessage.expiresAt = expiresInHours > 0 ? 
            block.timestamp + (expiresInHours * 1 hours) : 0;
        newMessage.isProcessed = false;
        newMessage.gasUsed = gasleft();
        
        // Add to inboxes/outboxes
        agentInbox[receiverId].push(messageCounter);
        agentOutbox[senderId].push(messageCounter);
        
        emit MessageSent(
            messageCounter,
            senderId,
            receiverId,
            messageType,
            priority,
            block.timestamp
        );
        
        // Calculate gas used
        newMessage.gasUsed = newMessage.gasUsed - gasleft();
        
        return messageCounter;
    }
    
    /**
     * @notice Process a received message
     * @param messageId ID of message to process
     * @param processorId ID of processing agent
     * @param responseHash IPFS hash of response (optional)
     */
    function processMessage(
        uint256 messageId,
        string memory processorId,
        string memory responseHash
    ) external onlyAgentWallet(processorId) {
        Message storage message = messages[messageId];
        
        require(keccak256(bytes(message.receiverId)) == keccak256(bytes(processorId)), 
                "Not the intended receiver");
        require(!message.isProcessed, "Already processed");
        require(message.expiresAt == 0 || block.timestamp <= message.expiresAt, 
                "Message expired");
        
        message.isProcessed = true;
        message.responseHash = responseHash;
        
        emit MessageProcessed(messageId, processorId, responseHash, block.timestamp);
    }
    
    /**
     * @notice Start a multi-agent workflow
     * @param workflowId Unique workflow identifier
     * @param agents Array of participating agent IDs
     * @param totalSteps Number of steps in workflow
     */
    function startWorkflow(
        string memory workflowId,
        string[] memory agents,
        uint256 totalSteps
    ) external {
        require(!workflows[workflowId].isActive, "Workflow already exists");
        require(agents.length >= 2, "Need at least 2 agents");
        require(totalSteps > 0, "Invalid step count");
        
        // Verify all agents are active
        for (uint i = 0; i < agents.length; i++) {
            require(registry.isAgentActive(agents[i]), "Agent not active");
        }
        
        WorkflowContext storage workflow = workflows[workflowId];
        workflow.workflowId = workflowId;
        workflow.participatingAgents = agents;
        workflow.currentStep = 0;
        workflow.totalSteps = totalSteps;
        workflow.isActive = true;
        workflow.startedAt = block.timestamp;
        
        emit WorkflowStarted(workflowId, agents, totalSteps, block.timestamp);
    }
    
    /**
     * @notice Advance workflow to next step
     * @param workflowId Workflow to advance
     * @param agentId Agent advancing the workflow
     */
    function advanceWorkflow(
        string memory workflowId,
        string memory agentId
    ) external onlyAgentWallet(agentId) {
        WorkflowContext storage workflow = workflows[workflowId];
        
        require(workflow.isActive, "Workflow not active");
        require(_isParticipant(workflow, agentId), "Not a participant");
        require(workflow.currentStep < workflow.totalSteps, "Workflow complete");
        
        workflow.currentStep++;
        
        if (workflow.currentStep >= workflow.totalSteps) {
            workflow.isActive = false;
            workflow.completedAt = block.timestamp;
            
            emit WorkflowCompleted(workflowId, block.timestamp, 0);
        }
    }
    
    /**
     * @notice Establish trust between two agents for priority messaging
     * @param senderId Sender agent ID
     * @param receiverId Receiver agent ID
     */
    function establishTrust(
        string memory senderId,
        string memory receiverId
    ) external onlyAgentWallet(senderId) {
        trustedPairs[senderId][receiverId] = true;
        
        emit TrustEstablished(senderId, receiverId, block.timestamp);
    }
    
    /**
     * @notice Get inbox messages for an agent
     * @param agentId Agent to query
     * @param limit Maximum messages to return
     * @return Array of message IDs
     */
    function getInbox(string memory agentId, uint256 limit) 
        external 
        view 
        returns (uint256[] memory) 
    {
        uint256[] memory inbox = agentInbox[agentId];
        
        if (inbox.length <= limit) {
            return inbox;
        }
        
        // Return latest messages up to limit
        uint256[] memory latestMessages = new uint256[](limit);
        uint256 start = inbox.length - limit;
        
        for (uint256 i = 0; i < limit; i++) {
            latestMessages[i] = inbox[start + i];
        }
        
        return latestMessages;
    }
    
    /**
     * @notice Get unprocessed messages for an agent
     * @param agentId Agent to query
     * @return Array of unprocessed message IDs
     */
    function getUnprocessedMessages(string memory agentId) 
        external 
        view 
        returns (uint256[] memory) 
    {
        uint256[] memory inbox = agentInbox[agentId];
        uint256 unprocessedCount = 0;
        
        // Count unprocessed
        for (uint256 i = 0; i < inbox.length; i++) {
            if (!messages[inbox[i]].isProcessed) {
                unprocessedCount++;
            }
        }
        
        // Collect unprocessed
        uint256[] memory unprocessed = new uint256[](unprocessedCount);
        uint256 index = 0;
        
        for (uint256 i = 0; i < inbox.length; i++) {
            if (!messages[inbox[i]].isProcessed) {
                unprocessed[index++] = inbox[i];
            }
        }
        
        return unprocessed;
    }
    
    /**
     * @notice Clean up old messages (can be called by anyone for gas rewards)
     * @param maxMessages Maximum messages to clean in one transaction
     */
    function cleanupOldMessages(uint256 maxMessages) external {
        uint256 cleaned = 0;
        uint256 cutoffTime = block.timestamp - (MESSAGE_RETENTION_DAYS * 1 days);
        
        for (uint256 i = 1; i <= messageCounter && cleaned < maxMessages; i++) {
            Message storage message = messages[i];
            
            if (message.timestamp > 0 && message.timestamp < cutoffTime) {
                delete messages[i];
                cleaned++;
            }
        }
    }
    
    // Internal functions
    
    function _isIPFSHash(string memory str) private pure returns (bool) {
        bytes memory b = bytes(str);
        // IPFS hashes typically start with "Qm" and are 46 characters
        return b.length == 46 && b[0] == 0x51 && b[1] == 0x6d;
    }
    
    function _isParticipant(WorkflowContext storage workflow, string memory agentId) 
        private 
        view 
        returns (bool) 
    {
        for (uint i = 0; i < workflow.participatingAgents.length; i++) {
            if (keccak256(bytes(workflow.participatingAgents[i])) == 
                keccak256(bytes(agentId))) {
                return true;
            }
        }
        return false;
    }
}