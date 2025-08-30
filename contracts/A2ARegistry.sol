// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title A2A Agent Registry
 * @notice On-chain registry for A2A agents with capability management
 * @dev Manages agent registration, capability tracking, and status updates
 */
contract A2ARegistry {
    
    // Agent status enum
    enum AgentStatus {
        INACTIVE,
        ACTIVE,
        SUSPENDED,
        TERMINATED
    }
    
    // Agent registration structure
    struct Agent {
        string agentId;
        address walletAddress;
        string agentType;
        string[] capabilities;
        string[] mcpTools;
        AgentStatus status;
        uint256 registeredAt;
        uint256 lastUpdated;
        string ipfsSkillCard; // IPFS hash of skill card
        uint256 complianceScore;
    }
    
    // State variables
    mapping(string => Agent) public agents;
    mapping(address => string) public addressToAgentId;
    mapping(string => bool) public agentExists;
    string[] public registeredAgentIds;
    
    address public agentManager;
    uint256 public totalAgents;
    
    // Events
    event AgentRegistered(
        string indexed agentId,
        address indexed walletAddress,
        string agentType,
        uint256 timestamp
    );
    
    event AgentStatusUpdated(
        string indexed agentId,
        AgentStatus oldStatus,
        AgentStatus newStatus,
        uint256 timestamp
    );
    
    event CapabilitiesUpdated(
        string indexed agentId,
        string[] newCapabilities,
        uint256 timestamp
    );
    
    event SkillCardUpdated(
        string indexed agentId,
        string ipfsHash,
        uint256 complianceScore,
        uint256 timestamp
    );
    
    // Modifiers
    modifier onlyAgentManager() {
        require(msg.sender == agentManager, "Only Agent Manager can call");
        _;
    }
    
    modifier agentMustExist(string memory agentId) {
        require(agentExists[agentId], "Agent does not exist");
        _;
    }
    
    constructor() {
        agentManager = msg.sender;
    }
    
    /**
     * @notice Register a new A2A agent
     * @param agentId Unique identifier for the agent
     * @param walletAddress Blockchain wallet address for the agent
     * @param agentType Type of agent (e.g., trading_algorithm, data_analysis)
     * @param capabilities Array of agent capabilities
     * @param mcpTools Array of MCP tools the agent uses
     * @param ipfsSkillCard IPFS hash of the agent's skill card
     */
    function registerAgent(
        string memory agentId,
        address walletAddress,
        string memory agentType,
        string[] memory capabilities,
        string[] memory mcpTools,
        string memory ipfsSkillCard
    ) external onlyAgentManager {
        require(!agentExists[agentId], "Agent already registered");
        require(walletAddress != address(0), "Invalid wallet address");
        
        Agent storage newAgent = agents[agentId];
        newAgent.agentId = agentId;
        newAgent.walletAddress = walletAddress;
        newAgent.agentType = agentType;
        newAgent.capabilities = capabilities;
        newAgent.mcpTools = mcpTools;
        newAgent.status = AgentStatus.ACTIVE;
        newAgent.registeredAt = block.timestamp;
        newAgent.lastUpdated = block.timestamp;
        newAgent.ipfsSkillCard = ipfsSkillCard;
        newAgent.complianceScore = 100; // Start with perfect compliance
        
        agentExists[agentId] = true;
        addressToAgentId[walletAddress] = agentId;
        registeredAgentIds.push(agentId);
        totalAgents++;
        
        emit AgentRegistered(agentId, walletAddress, agentType, block.timestamp);
    }
    
    /**
     * @notice Update agent status
     * @param agentId Agent to update
     * @param newStatus New status for the agent
     */
    function updateAgentStatus(
        string memory agentId,
        AgentStatus newStatus
    ) external onlyAgentManager agentMustExist(agentId) {
        Agent storage agent = agents[agentId];
        AgentStatus oldStatus = agent.status;
        
        agent.status = newStatus;
        agent.lastUpdated = block.timestamp;
        
        emit AgentStatusUpdated(agentId, oldStatus, newStatus, block.timestamp);
    }
    
    /**
     * @notice Update agent capabilities
     * @param agentId Agent to update
     * @param newCapabilities New capability array
     */
    function updateCapabilities(
        string memory agentId,
        string[] memory newCapabilities
    ) external onlyAgentManager agentMustExist(agentId) {
        Agent storage agent = agents[agentId];
        
        agent.capabilities = newCapabilities;
        agent.lastUpdated = block.timestamp;
        
        emit CapabilitiesUpdated(agentId, newCapabilities, block.timestamp);
    }
    
    /**
     * @notice Update agent skill card and compliance score
     * @param agentId Agent to update
     * @param ipfsHash New IPFS hash for skill card
     * @param complianceScore New compliance score (0-100)
     */
    function updateSkillCard(
        string memory agentId,
        string memory ipfsHash,
        uint256 complianceScore
    ) external onlyAgentManager agentMustExist(agentId) {
        require(complianceScore <= 100, "Invalid compliance score");
        
        Agent storage agent = agents[agentId];
        
        agent.ipfsSkillCard = ipfsHash;
        agent.complianceScore = complianceScore;
        agent.lastUpdated = block.timestamp;
        
        emit SkillCardUpdated(agentId, ipfsHash, complianceScore, block.timestamp);
    }
    
    /**
     * @notice Get agent details
     * @param agentId Agent to query
     * @return Agent struct with all details
     */
    function getAgent(string memory agentId) 
        external 
        view 
        agentMustExist(agentId) 
        returns (Agent memory) 
    {
        return agents[agentId];
    }
    
    /**
     * @notice Get agent by wallet address
     * @param walletAddress Wallet address to query
     * @return Agent struct with all details
     */
    function getAgentByAddress(address walletAddress) 
        external 
        view 
        returns (Agent memory) 
    {
        string memory agentId = addressToAgentId[walletAddress];
        require(bytes(agentId).length > 0, "No agent for this address");
        return agents[agentId];
    }
    
    /**
     * @notice Get all registered agent IDs
     * @return Array of agent IDs
     */
    function getAllAgentIds() external view returns (string[] memory) {
        return registeredAgentIds;
    }
    
    /**
     * @notice Check if agent is active
     * @param agentId Agent to check
     * @return Boolean indicating if agent is active
     */
    function isAgentActive(string memory agentId) 
        external 
        view 
        agentMustExist(agentId) 
        returns (bool) 
    {
        return agents[agentId].status == AgentStatus.ACTIVE;
    }
    
    /**
     * @notice Transfer Agent Manager role
     * @param newManager New Agent Manager address
     */
    function transferAgentManager(address newManager) external onlyAgentManager {
        require(newManager != address(0), "Invalid address");
        agentManager = newManager;
    }
}