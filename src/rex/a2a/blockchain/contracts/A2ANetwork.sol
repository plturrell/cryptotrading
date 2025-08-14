// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title A2A Network - On-chain Agent Registry and Workflow Management
 * @notice Manages agent registration, capabilities, and workflow execution on Vercel blockchain
 */
contract A2ANetwork {
    
    // Agent status enum
    enum AgentStatus { INACTIVE, ACTIVE, BUSY, ERROR }
    
    // Message types enum
    enum MessageType {
        DATA_LOAD_REQUEST,
        DATA_LOAD_RESPONSE,
        ANALYSIS_REQUEST,
        ANALYSIS_RESPONSE,
        WORKFLOW_REQUEST,
        WORKFLOW_RESPONSE,
        ERROR
    }
    
    // Agent struct
    struct Agent {
        string agentId;
        address agentAddress;
        string agentType;
        string[] capabilities;
        AgentStatus status;
        uint256 registeredAt;
        uint256 lastActiveAt;
        string ipfsMetadata; // IPFS hash for extended metadata
    }
    
    // Workflow step struct
    struct WorkflowStep {
        string stepId;
        string agentId;
        string action;
        string inputDataHash; // IPFS hash of input data
        string[] dependsOn;
        uint256 timeout;
        uint8 retryCount;
        bool completed;
        string resultHash; // IPFS hash of result
    }
    
    // Workflow struct
    struct Workflow {
        string workflowId;
        string name;
        string description;
        WorkflowStep[] steps;
        address creator;
        uint256 createdAt;
        bool active;
        uint256 executionCount;
    }
    
    // A2A Message struct
    struct A2AMessage {
        string messageId;
        string senderId;
        string receiverId;
        MessageType messageType;
        string payloadHash; // IPFS hash of payload
        uint256 timestamp;
        uint256 priority;
        bool processed;
    }
    
    // State variables
    mapping(string => Agent) public agents;
    mapping(address => string) public addressToAgentId;
    mapping(string => Workflow) public workflows;
    mapping(string => A2AMessage[]) public agentMessages;
    mapping(string => mapping(string => bool)) public agentConnections;
    
    string[] public registeredAgentIds;
    string[] public registeredWorkflowIds;
    
    // Events
    event AgentRegistered(string indexed agentId, address indexed agentAddress, string agentType);
    event AgentStatusUpdated(string indexed agentId, AgentStatus newStatus);
    event WorkflowRegistered(string indexed workflowId, address indexed creator);
    event WorkflowExecuted(string indexed workflowId, string executionId);
    event MessageSent(string indexed messageId, string indexed senderId, string indexed receiverId);
    event ConnectionEstablished(string indexed agent1Id, string indexed agent2Id);
    
    // Modifiers
    modifier onlyRegisteredAgent(string memory agentId) {
        require(agents[agentId].agentAddress != address(0), "Agent not registered");
        require(agents[agentId].agentAddress == msg.sender, "Unauthorized agent");
        _;
    }
    
    modifier agentExists(string memory agentId) {
        require(agents[agentId].agentAddress != address(0), "Agent does not exist");
        _;
    }
    
    /**
     * @notice Register a new agent on-chain
     * @param agentId Unique identifier for the agent
     * @param agentType Type of agent (historical-loader, database, etc.)
     * @param capabilities Array of agent capabilities
     * @param ipfsMetadata IPFS hash containing extended agent metadata
     */
    function registerAgent(
        string memory agentId,
        string memory agentType,
        string[] memory capabilities,
        string memory ipfsMetadata
    ) external {
        require(agents[agentId].agentAddress == address(0), "Agent already registered");
        require(bytes(addressToAgentId[msg.sender]).length == 0, "Address already has agent");
        
        Agent storage newAgent = agents[agentId];
        newAgent.agentId = agentId;
        newAgent.agentAddress = msg.sender;
        newAgent.agentType = agentType;
        newAgent.capabilities = capabilities;
        newAgent.status = AgentStatus.ACTIVE;
        newAgent.registeredAt = block.timestamp;
        newAgent.lastActiveAt = block.timestamp;
        newAgent.ipfsMetadata = ipfsMetadata;
        
        addressToAgentId[msg.sender] = agentId;
        registeredAgentIds.push(agentId);
        
        emit AgentRegistered(agentId, msg.sender, agentType);
    }
    
    /**
     * @notice Update agent status
     * @param status New status for the agent
     */
    function updateAgentStatus(AgentStatus status) external {
        string memory agentId = addressToAgentId[msg.sender];
        require(bytes(agentId).length > 0, "Agent not found");
        
        agents[agentId].status = status;
        agents[agentId].lastActiveAt = block.timestamp;
        
        emit AgentStatusUpdated(agentId, status);
    }
    
    /**
     * @notice Register a new workflow
     * @param workflowId Unique identifier for the workflow
     * @param name Workflow name
     * @param description Workflow description
     * @param stepData Encoded workflow steps data
     */
    function registerWorkflow(
        string memory workflowId,
        string memory name,
        string memory description,
        bytes memory stepData
    ) external {
        require(bytes(workflows[workflowId].workflowId).length == 0, "Workflow already exists");
        
        Workflow storage newWorkflow = workflows[workflowId];
        newWorkflow.workflowId = workflowId;
        newWorkflow.name = name;
        newWorkflow.description = description;
        newWorkflow.creator = msg.sender;
        newWorkflow.createdAt = block.timestamp;
        newWorkflow.active = true;
        
        // Steps would be decoded from stepData
        // For now, this is a placeholder
        
        registeredWorkflowIds.push(workflowId);
        
        emit WorkflowRegistered(workflowId, msg.sender);
    }
    
    /**
     * @notice Send A2A message between agents
     * @param receiverId Target agent ID
     * @param messageType Type of message
     * @param payloadHash IPFS hash of message payload
     * @param priority Message priority (0-3)
     */
    function sendMessage(
        string memory receiverId,
        MessageType messageType,
        string memory payloadHash,
        uint256 priority
    ) external {
        string memory senderId = addressToAgentId[msg.sender];
        require(bytes(senderId).length > 0, "Sender not registered");
        require(agents[receiverId].agentAddress != address(0), "Receiver not registered");
        
        string memory messageId = string(abi.encodePacked(
            senderId, "_", 
            uint2str(block.timestamp), "_",
            uint2str(agentMessages[receiverId].length)
        ));
        
        A2AMessage memory newMessage = A2AMessage({
            messageId: messageId,
            senderId: senderId,
            receiverId: receiverId,
            messageType: messageType,
            payloadHash: payloadHash,
            timestamp: block.timestamp,
            priority: priority,
            processed: false
        });
        
        agentMessages[receiverId].push(newMessage);
        
        emit MessageSent(messageId, senderId, receiverId);
    }
    
    /**
     * @notice Establish connection between two agents
     * @param otherAgentId ID of the other agent to connect with
     */
    function establishConnection(string memory otherAgentId) external {
        string memory myAgentId = addressToAgentId[msg.sender];
        require(bytes(myAgentId).length > 0, "Agent not registered");
        require(agents[otherAgentId].agentAddress != address(0), "Other agent not registered");
        
        agentConnections[myAgentId][otherAgentId] = true;
        agentConnections[otherAgentId][myAgentId] = true;
        
        emit ConnectionEstablished(myAgentId, otherAgentId);
    }
    
    /**
     * @notice Get agent details
     * @param agentId ID of the agent
     * @return Agent struct with all details
     */
    function getAgent(string memory agentId) 
        external 
        view 
        agentExists(agentId) 
        returns (Agent memory) 
    {
        return agents[agentId];
    }
    
    /**
     * @notice Get pending messages for an agent
     * @param agentId ID of the agent
     * @return Array of pending messages
     */
    function getPendingMessages(string memory agentId) 
        external 
        view 
        returns (A2AMessage[] memory) 
    {
        uint256 pendingCount = 0;
        
        // Count pending messages
        for (uint i = 0; i < agentMessages[agentId].length; i++) {
            if (!agentMessages[agentId][i].processed) {
                pendingCount++;
            }
        }
        
        // Create array of pending messages
        A2AMessage[] memory pending = new A2AMessage[](pendingCount);
        uint256 index = 0;
        
        for (uint i = 0; i < agentMessages[agentId].length; i++) {
            if (!agentMessages[agentId][i].processed) {
                pending[index] = agentMessages[agentId][i];
                index++;
            }
        }
        
        return pending;
    }
    
    /**
     * @notice Mark message as processed
     * @param messageId ID of the message to mark as processed
     */
    function markMessageProcessed(string memory messageId) external {
        string memory agentId = addressToAgentId[msg.sender];
        require(bytes(agentId).length > 0, "Agent not registered");
        
        for (uint i = 0; i < agentMessages[agentId].length; i++) {
            if (keccak256(bytes(agentMessages[agentId][i].messageId)) == 
                keccak256(bytes(messageId))) {
                agentMessages[agentId][i].processed = true;
                break;
            }
        }
    }
    
    /**
     * @notice Get all registered agents
     * @return Array of agent IDs
     */
    function getAllAgents() external view returns (string[] memory) {
        return registeredAgentIds;
    }
    
    /**
     * @notice Get all registered workflows
     * @return Array of workflow IDs
     */
    function getAllWorkflows() external view returns (string[] memory) {
        return registeredWorkflowIds;
    }
    
    // Utility function to convert uint to string
    function uint2str(uint256 _i) internal pure returns (string memory) {
        if (_i == 0) {
            return "0";
        }
        uint256 j = _i;
        uint256 length;
        while (j != 0) {
            length++;
            j /= 10;
        }
        bytes memory bstr = new bytes(length);
        uint256 k = length;
        while (_i != 0) {
            k = k-1;
            uint8 temp = (48 + uint8(_i - _i / 10 * 10));
            bytes1 b1 = bytes1(temp);
            bstr[k] = b1;
            _i /= 10;
        }
        return string(bstr);
    }
}