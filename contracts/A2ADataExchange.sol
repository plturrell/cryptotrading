// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./A2ARegistry.sol";

/**
 * @title A2A On-Chain Data Exchange
 * @notice Secure on-chain data storage and exchange between A2A agents
 * @dev Stores data directly on-chain with access control and workflow integration
 */
contract A2ADataExchange {
    
    // Data structure for on-chain storage
    struct DataPacket {
        uint256 dataId;
        string senderAgentId;
        string receiverAgentId;
        bytes data;  // Actual data stored on-chain
        string dataType;  // e.g., "market_data", "analysis_result", "prediction"
        uint256 timestamp;
        bool isEncrypted;
        bytes32 dataHash;  // For integrity verification
        DataStatus status;
        uint256 expiresAt;  // Data expiration timestamp
    }
    
    // Workflow data structure
    struct WorkflowData {
        uint256 workflowId;
        string workflowType;
        string[] participantAgents;
        uint256[] dataPacketIds;
        WorkflowStatus status;
        uint256 createdAt;
        uint256 completedAt;
        mapping(string => bool) agentApprovals;
        bytes workflowResult;
    }
    
    // Data access permission structure
    struct DataAccess {
        string agentId;
        bool canRead;
        bool canWrite;
        bool canDelete;
        uint256 grantedAt;
        uint256 expiresAt;
    }
    
    enum DataStatus {
        PENDING,
        AVAILABLE,
        PROCESSING,
        CONSUMED,
        EXPIRED,
        DELETED
    }
    
    enum WorkflowStatus {
        INITIATED,
        IN_PROGRESS,
        AWAITING_DATA,
        PROCESSING,
        COMPLETED,
        FAILED,
        CANCELLED
    }
    
    // State variables
    A2ARegistry public registry;
    address public dataManager;
    uint256 public nextDataId = 1;
    uint256 public nextWorkflowId = 1;
    uint256 public maxDataSize = 256 * 1024; // 256KB max per data packet
    uint256 public dataRetentionPeriod = 30 days;
    
    // Storage mappings
    mapping(uint256 => DataPacket) public dataPackets;
    mapping(uint256 => WorkflowData) public workflows;
    mapping(string => uint256[]) public agentDataIndex;  // agentId => dataIds
    mapping(string => uint256[]) public agentWorkflows;  // agentId => workflowIds
    mapping(uint256 => mapping(string => DataAccess)) public dataAccessControl;  // dataId => agentId => access
    mapping(bytes32 => uint256) public dataHashIndex;  // dataHash => dataId for deduplication
    
    // Events
    event DataStored(
        uint256 indexed dataId,
        string indexed senderAgentId,
        string indexed receiverAgentId,
        string dataType,
        uint256 dataSize,
        bytes32 dataHash,
        uint256 timestamp
    );
    
    event DataAccessed(
        uint256 indexed dataId,
        string indexed agentId,
        uint256 timestamp
    );
    
    event WorkflowCreated(
        uint256 indexed workflowId,
        string workflowType,
        string[] participants,
        uint256 timestamp
    );
    
    event WorkflowCompleted(
        uint256 indexed workflowId,
        bytes result,
        uint256 timestamp
    );
    
    event DataAccessGranted(
        uint256 indexed dataId,
        string indexed agentId,
        bool canRead,
        bool canWrite,
        uint256 expiresAt
    );
    
    event DataDeleted(
        uint256 indexed dataId,
        string indexed deletedBy,
        uint256 timestamp
    );
    
    // Modifiers
    modifier onlyDataManager() {
        require(msg.sender == dataManager, "Only data manager can call");
        _;
    }
    
    modifier onlyActiveAgent(string memory agentId) {
        require(registry.isAgentActive(agentId), "Agent must be active");
        _;
    }
    
    modifier dataExists(uint256 dataId) {
        require(dataPackets[dataId].dataId != 0, "Data does not exist");
        _;
    }
    
    modifier hasReadAccess(uint256 dataId, string memory agentId) {
        DataPacket memory packet = dataPackets[dataId];
        require(
            keccak256(bytes(packet.senderAgentId)) == keccak256(bytes(agentId)) ||
            keccak256(bytes(packet.receiverAgentId)) == keccak256(bytes(agentId)) ||
            dataAccessControl[dataId][agentId].canRead,
            "No read access"
        );
        _;
    }
    
    constructor(address _registry) {
        registry = A2ARegistry(_registry);
        dataManager = msg.sender;
    }
    
    /**
     * @notice Store data on-chain for agent-to-agent exchange
     * @param receiverAgentId Target agent ID
     * @param data Raw data to store
     * @param dataType Type of data being stored
     * @param isEncrypted Whether data is encrypted
     * @param ttl Time to live in seconds (0 for default retention)
     */
    function storeData(
        string memory senderAgentId,
        string memory receiverAgentId,
        bytes memory data,
        string memory dataType,
        bool isEncrypted,
        uint256 ttl
    ) external onlyActiveAgent(senderAgentId) returns (uint256) {
        require(data.length > 0, "Data cannot be empty");
        require(data.length <= maxDataSize, "Data exceeds size limit");
        
        // Calculate data hash for integrity and deduplication
        bytes32 dataHash = keccak256(data);
        
        // Check for duplicate data
        if (dataHashIndex[dataHash] != 0) {
            // Data already exists, return existing ID
            return dataHashIndex[dataHash];
        }
        
        uint256 dataId = nextDataId++;
        uint256 expiresAt = ttl > 0 ? 
            block.timestamp + ttl : 
            block.timestamp + dataRetentionPeriod;
        
        // Store data packet
        DataPacket storage packet = dataPackets[dataId];
        packet.dataId = dataId;
        packet.senderAgentId = senderAgentId;
        packet.receiverAgentId = receiverAgentId;
        packet.data = data;
        packet.dataType = dataType;
        packet.timestamp = block.timestamp;
        packet.isEncrypted = isEncrypted;
        packet.dataHash = dataHash;
        packet.status = DataStatus.AVAILABLE;
        packet.expiresAt = expiresAt;
        
        // Update indices
        agentDataIndex[senderAgentId].push(dataId);
        agentDataIndex[receiverAgentId].push(dataId);
        dataHashIndex[dataHash] = dataId;
        
        // Grant default access
        dataAccessControl[dataId][senderAgentId] = DataAccess({
            agentId: senderAgentId,
            canRead: true,
            canWrite: true,
            canDelete: true,
            grantedAt: block.timestamp,
            expiresAt: expiresAt
        });
        
        dataAccessControl[dataId][receiverAgentId] = DataAccess({
            agentId: receiverAgentId,
            canRead: true,
            canWrite: false,
            canDelete: false,
            grantedAt: block.timestamp,
            expiresAt: expiresAt
        });
        
        emit DataStored(
            dataId,
            senderAgentId,
            receiverAgentId,
            dataType,
            data.length,
            dataHash,
            block.timestamp
        );
        
        return dataId;
    }
    
    /**
     * @notice Retrieve data from on-chain storage
     * @param dataId ID of the data packet
     * @param agentId Agent requesting the data
     */
    function retrieveData(
        uint256 dataId,
        string memory agentId
    ) external 
      dataExists(dataId) 
      onlyActiveAgent(agentId)
      hasReadAccess(dataId, agentId)
      returns (bytes memory data, string memory dataType, bool isEncrypted) 
    {
        DataPacket storage packet = dataPackets[dataId];
        
        require(packet.status == DataStatus.AVAILABLE, "Data not available");
        require(block.timestamp < packet.expiresAt, "Data has expired");
        
        emit DataAccessed(dataId, agentId, block.timestamp);
        
        return (packet.data, packet.dataType, packet.isEncrypted);
    }
    
    /**
     * @notice Create a new workflow involving multiple agents
     * @param workflowType Type of workflow
     * @param participants Array of participating agent IDs
     */
    function createWorkflow(
        string memory workflowType,
        string[] memory participants
    ) external returns (uint256) {
        require(participants.length >= 2, "Workflow needs at least 2 agents");
        
        uint256 workflowId = nextWorkflowId++;
        
        WorkflowData storage workflow = workflows[workflowId];
        workflow.workflowId = workflowId;
        workflow.workflowType = workflowType;
        workflow.participantAgents = participants;
        workflow.status = WorkflowStatus.INITIATED;
        workflow.createdAt = block.timestamp;
        
        // Register workflow for each participant
        for (uint i = 0; i < participants.length; i++) {
            agentWorkflows[participants[i]].push(workflowId);
            workflow.agentApprovals[participants[i]] = false;
        }
        
        emit WorkflowCreated(workflowId, workflowType, participants, block.timestamp);
        
        return workflowId;
    }
    
    /**
     * @notice Add data to a workflow
     * @param workflowId Workflow to add data to
     * @param dataId Data packet ID to add
     */
    function addDataToWorkflow(
        uint256 workflowId,
        uint256 dataId,
        string memory agentId
    ) external 
      dataExists(dataId)
      onlyActiveAgent(agentId)
    {
        WorkflowData storage workflow = workflows[workflowId];
        require(workflow.workflowId != 0, "Workflow does not exist");
        require(workflow.status == WorkflowStatus.IN_PROGRESS || 
                workflow.status == WorkflowStatus.AWAITING_DATA, 
                "Workflow not accepting data");
        
        // Verify agent is participant
        bool isParticipant = false;
        for (uint i = 0; i < workflow.participantAgents.length; i++) {
            if (keccak256(bytes(workflow.participantAgents[i])) == keccak256(bytes(agentId))) {
                isParticipant = true;
                break;
            }
        }
        require(isParticipant, "Agent not in workflow");
        
        workflow.dataPacketIds.push(dataId);
        
        if (workflow.status == WorkflowStatus.AWAITING_DATA) {
            workflow.status = WorkflowStatus.PROCESSING;
        }
    }
    
    /**
     * @notice Complete a workflow with results
     * @param workflowId Workflow to complete
     * @param result Result data
     */
    function completeWorkflow(
        uint256 workflowId,
        bytes memory result,
        string memory agentId
    ) external onlyActiveAgent(agentId) {
        WorkflowData storage workflow = workflows[workflowId];
        require(workflow.workflowId != 0, "Workflow does not exist");
        require(workflow.status == WorkflowStatus.PROCESSING, "Workflow not ready for completion");
        
        workflow.status = WorkflowStatus.COMPLETED;
        workflow.completedAt = block.timestamp;
        workflow.workflowResult = result;
        
        emit WorkflowCompleted(workflowId, result, block.timestamp);
    }
    
    /**
     * @notice Grant data access to another agent
     * @param dataId Data packet ID
     * @param targetAgentId Agent to grant access to
     * @param canRead Read permission
     * @param canWrite Write permission
     * @param duration Access duration in seconds
     */
    function grantDataAccess(
        uint256 dataId,
        string memory granterAgentId,
        string memory targetAgentId,
        bool canRead,
        bool canWrite,
        uint256 duration
    ) external 
      dataExists(dataId)
      onlyActiveAgent(granterAgentId)
      onlyActiveAgent(targetAgentId)
    {
        DataPacket memory packet = dataPackets[dataId];
        require(
            keccak256(bytes(packet.senderAgentId)) == keccak256(bytes(granterAgentId)) ||
            dataAccessControl[dataId][granterAgentId].canWrite,
            "No permission to grant access"
        );
        
        uint256 expiresAt = duration > 0 ? 
            block.timestamp + duration : 
            packet.expiresAt;
        
        dataAccessControl[dataId][targetAgentId] = DataAccess({
            agentId: targetAgentId,
            canRead: canRead,
            canWrite: canWrite,
            canDelete: false,
            grantedAt: block.timestamp,
            expiresAt: expiresAt
        });
        
        emit DataAccessGranted(dataId, targetAgentId, canRead, canWrite, expiresAt);
    }
    
    /**
     * @notice Delete data from on-chain storage
     * @param dataId Data packet ID to delete
     * @param agentId Agent requesting deletion
     */
    function deleteData(
        uint256 dataId,
        string memory agentId
    ) external 
      dataExists(dataId)
      onlyActiveAgent(agentId)
    {
        require(
            dataAccessControl[dataId][agentId].canDelete,
            "No delete permission"
        );
        
        DataPacket storage packet = dataPackets[dataId];
        packet.status = DataStatus.DELETED;
        delete packet.data;  // Free up storage
        
        emit DataDeleted(dataId, agentId, block.timestamp);
    }
    
    /**
     * @notice Get workflow details
     * @param workflowId Workflow ID
     */
    function getWorkflow(uint256 workflowId) 
        external 
        view 
        returns (
            string memory workflowType,
            string[] memory participants,
            uint256[] memory dataIds,
            WorkflowStatus status,
            uint256 createdAt,
            bytes memory result
        ) 
    {
        WorkflowData storage workflow = workflows[workflowId];
        return (
            workflow.workflowType,
            workflow.participantAgents,
            workflow.dataPacketIds,
            workflow.status,
            workflow.createdAt,
            workflow.workflowResult
        );
    }
    
    /**
     * @notice Get agent's data packets
     * @param agentId Agent ID
     */
    function getAgentData(string memory agentId) 
        external 
        view 
        returns (uint256[] memory) 
    {
        return agentDataIndex[agentId];
    }
    
    /**
     * @notice Clean up expired data
     */
    function cleanupExpiredData() external {
        // This would typically be called by a keeper or automated process
        // Implementation would iterate through data and delete expired entries
        // Keeping simple for demo
    }
}