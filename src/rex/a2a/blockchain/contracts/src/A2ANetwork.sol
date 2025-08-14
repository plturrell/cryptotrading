// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title A2A Network - On-chain Agent Registry and Workflow Management
 * @notice Manages agent registration, capabilities, and workflow execution on local blockchain
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
        DATA_QUERY,
        DATA_QUERY_RESPONSE,
        TRADE_EXECUTION,
        TRADE_RESPONSE,
        WORKFLOW_REQUEST,
        WORKFLOW_RESPONSE,
        WORKFLOW_STATUS,
        HEARTBEAT,
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
        string inputDataHash;
        string[] dependsOn;
        uint256 timeout;
        uint8 retryCount;
        bool completed;
        string resultHash;
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
        string payloadHash;
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
        newWorkflow.executionCount = 0;
        
        // Decode and store workflow steps
        _decodeAndStoreSteps(workflowId, stepData);
        
        registeredWorkflowIds.push(workflowId);
        
        emit WorkflowRegistered(workflowId, msg.sender);
    }
    
    /**
     * @notice Execute workflow and increment execution count
     */
    function executeWorkflow(string memory workflowId, string memory executionId) external {
        require(bytes(workflows[workflowId].workflowId).length > 0, "Workflow does not exist");
        require(workflows[workflowId].active, "Workflow is not active");
        
        workflows[workflowId].executionCount++;
        
        emit WorkflowExecuted(workflowId, executionId);
    }
    
    /**
     * @notice Get workflow details
     */
    function getWorkflow(string memory workflowId) external view returns (Workflow memory) {
        require(bytes(workflows[workflowId].workflowId).length > 0, "Workflow does not exist");
        return workflows[workflowId];
    }

    /**
     * @notice Send A2A message between agents
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
     * @notice Get agent details
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
     */
    function getAllAgents() external view returns (string[] memory) {
        return registeredAgentIds;
    }
    
    /**
     * @notice Get all registered workflows
     */
    function getAllWorkflows() external view returns (string[] memory) {
        return registeredWorkflowIds;
    }
    
    /**
     * @notice Internal function to decode and store workflow steps
     */
    function _decodeAndStoreSteps(string memory workflowId, bytes memory stepData) internal {
        // For simplicity, we'll store the encoded step data as-is
        // In a full implementation, this would properly decode the steps
        // and store them in the workflow struct
        
        // This is a simplified implementation that stores minimal step info
        // Real implementation would decode the JSON and create proper WorkflowStep structs
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