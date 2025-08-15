// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title WorkflowInstance
 * @notice Individual workflow instance contract that tracks execution with full audit trail
 */
contract WorkflowInstance {
    address public parentContract;
    string public workflowId;
    string public executionId;
    address public creator;
    uint256 public createdAt;
    
    struct StepExecution {
        string stepId;
        string agentId;
        address agentAddress;
        string messageId;
        uint8 status; // 0=pending, 1=running, 2=completed, 3=failed
        uint256 timestamp;
        bytes32 dataHash; // Hash of step result data
    }
    
    struct MessageRecord {
        string messageId;
        string senderId;
        address senderAddress;
        string receiverId;
        bytes32 messageHash;
        uint256 timestamp;
        bytes signature; // Agent's signature of the message
    }
    
    struct DataRecord {
        string dataId;
        string agentId;
        address agentAddress;
        bytes32 dataHash;
        uint256 timestamp;
        bytes signature; // Agent's signature of the data
    }
    
    mapping(string => StepExecution) public steps;
    mapping(string => MessageRecord) public messages;
    mapping(string => DataRecord) public dataRecords;
    
    string[] public stepIds;
    string[] public messageIds;
    string[] public dataIds;
    
    event StepExecuted(
        string indexed stepId,
        string indexed agentId,
        string messageId,
        uint8 status,
        bytes32 dataHash
    );
    
    event MessageRecorded(
        string indexed messageId,
        string indexed senderId,
        string indexed receiverId,
        bytes signature
    );
    
    event DataRecorded(
        string indexed dataId,
        string indexed agentId,
        bytes32 dataHash,
        bytes signature
    );
    
    modifier onlyAuthorizedAgent() {
        require(
            IParentContract(parentContract).isRegisteredAgent(msg.sender),
            "Only registered agents can interact"
        );
        _;
    }
    
    constructor(
        address _parentContract,
        string memory _workflowId,
        string memory _executionId,
        address _creator
    ) {
        parentContract = _parentContract;
        workflowId = _workflowId;
        executionId = _executionId;
        creator = _creator;
        createdAt = block.timestamp;
    }
    
    /**
     * @notice Record step execution with result data hash
     */
    function recordStepExecution(
        string memory _stepId,
        string memory _agentId,
        string memory _messageId,
        uint8 _status,
        bytes32 _dataHash
    ) external onlyAuthorizedAgent {
        require(_status <= 3, "Invalid status");
        
        steps[_stepId] = StepExecution({
            stepId: _stepId,
            agentId: _agentId,
            agentAddress: msg.sender,
            messageId: _messageId,
            status: _status,
            timestamp: block.timestamp,
            dataHash: _dataHash
        });
        
        stepIds.push(_stepId);
        emit StepExecuted(_stepId, _agentId, _messageId, _status, _dataHash);
    }
    
    /**
     * @notice Record message with signature
     */
    function recordMessage(
        string memory _messageId,
        string memory _senderId,
        string memory _receiverId,
        bytes32 _messageHash,
        bytes memory _signature
    ) external onlyAuthorizedAgent {
        messages[_messageId] = MessageRecord({
            messageId: _messageId,
            senderId: _senderId,
            senderAddress: msg.sender,
            receiverId: _receiverId,
            messageHash: _messageHash,
            timestamp: block.timestamp,
            signature: _signature
        });
        
        messageIds.push(_messageId);
        emit MessageRecorded(_messageId, _senderId, _receiverId, _signature);
    }
    
    /**
     * @notice Record data with signature
     */
    function recordData(
        string memory _dataId,
        string memory _agentId,
        bytes32 _dataHash,
        bytes memory _signature
    ) external onlyAuthorizedAgent {
        dataRecords[_dataId] = DataRecord({
            dataId: _dataId,
            agentId: _agentId,
            agentAddress: msg.sender,
            dataHash: _dataHash,
            timestamp: block.timestamp,
            signature: _signature
        });
        
        dataIds.push(_dataId);
        emit DataRecorded(_dataId, _agentId, _dataHash, _signature);
    }
    
    /**
     * @notice Verify message signature matches sender
     */
    function verifyMessageSignature(
        string memory _messageId,
        bytes32 _messageHash,
        bytes memory _signature
    ) external view returns (bool) {
        MessageRecord memory record = messages[_messageId];
        require(bytes(record.messageId).length > 0, "Message not found");
        
        // Verify hash matches
        if (record.messageHash != _messageHash) {
            return false;
        }
        
        // Recover signer from signature
        address signer = recoverSigner(_messageHash, _signature);
        
        // Verify signer matches sender address
        return signer == record.senderAddress;
    }
    
    /**
     * @notice Verify data signature matches agent
     */
    function verifyDataSignature(
        string memory _dataId,
        bytes32 _dataHash,
        bytes memory _signature
    ) external view returns (bool) {
        DataRecord memory record = dataRecords[_dataId];
        require(bytes(record.dataId).length > 0, "Data not found");
        
        // Verify hash matches
        if (record.dataHash != _dataHash) {
            return false;
        }
        
        // Recover signer from signature
        address signer = recoverSigner(_dataHash, _signature);
        
        // Verify signer matches agent address
        return signer == record.agentAddress;
    }
    
    /**
     * @notice Get execution summary
     */
    function getExecutionSummary() external view returns (
        uint256 totalSteps,
        uint256 completedSteps,
        uint256 failedSteps,
        uint256 totalMessages,
        uint256 totalDataRecords
    ) {
        totalSteps = stepIds.length;
        
        for (uint i = 0; i < stepIds.length; i++) {
            StepExecution memory step = steps[stepIds[i]];
            if (step.status == 2) {
                completedSteps++;
            } else if (step.status == 3) {
                failedSteps++;
            }
        }
        
        totalMessages = messageIds.length;
        totalDataRecords = dataIds.length;
    }
    
    /**
     * @notice Recover signer address from signature
     */
    function recoverSigner(bytes32 _hash, bytes memory _signature) internal pure returns (address) {
        require(_signature.length == 65, "Invalid signature length");
        
        bytes32 r;
        bytes32 s;
        uint8 v;
        
        assembly {
            r := mload(add(_signature, 32))
            s := mload(add(_signature, 64))
            v := byte(0, mload(add(_signature, 96)))
        }
        
        return ecrecover(_hash, v, r, s);
    }
    
    function getStepCount() external view returns (uint256) {
        return stepIds.length;
    }
    
    function getMessageCount() external view returns (uint256) {
        return messageIds.length;
    }
    
    function getDataCount() external view returns (uint256) {
        return dataIds.length;
    }
}

interface IParentContract {
    function isRegisteredAgent(address agent) external view returns (bool);
}