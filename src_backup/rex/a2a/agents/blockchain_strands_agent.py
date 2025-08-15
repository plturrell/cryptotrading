"""
Blockchain-integrated Strands Agent
Extends A2AStrandsAgent with blockchain transport capabilities
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from .a2a_strands_agent import A2AStrandsAgent
from ..transports import get_blockchain_transport
from ..blockchain.blockchain_registry import get_blockchain_registries
from ..protocols.a2a_protocol import A2AMessage, MessageType, A2AProtocol, AgentStatus

logger = logging.getLogger(__name__)

class BlockchainStrandsAgent(A2AStrandsAgent):
    """Blockchain-integrated Strands Agent"""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        private_key: str,
        model_provider: str = "grok4",
        version: str = "1.0",
        auto_register: bool = True
    ):
        """
        Initialize blockchain-integrated Strands agent
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (e.g., 'historical_loader', 'database')
            capabilities: List of agent capabilities
            private_key: Ethereum private key for blockchain operations
            model_provider: AI model provider ('grok4' only)
            version: Agent version
            auto_register: Whether to automatically register on blockchain
        """
        # Create blockchain transport
        self.private_key = private_key
        BlockchainTransport = get_blockchain_transport()
        blockchain_transport = BlockchainTransport(private_key)
        
        # Initialize A2A agent with blockchain transport
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities,
            model_provider=model_provider,
            version=version,
            transport=blockchain_transport
        )
        
        # Blockchain-specific attributes
        self.blockchain_registry, self.blockchain_workflow_registry = get_blockchain_registries()
        self.blockchain_address = None
        self.blockchain_registered = False
        
        # Register on blockchain if requested
        if auto_register and self.blockchain_registry:
            self._register_on_blockchain()
    
    def _register_on_blockchain(self):
        """Register agent on blockchain"""
        try:
            if not self.blockchain_registry:
                logger.error("No blockchain registry available")
                return
            
            # Register on blockchain
            tx_hash = self.blockchain_registry.register_agent_blockchain(
                self.agent_id,
                self.agent_type,
                self.capabilities,
                self.private_key,
                ipfs_metadata=""  # Could store extended metadata here
            )
            
            if tx_hash:
                self.blockchain_registered = True
                self.blockchain_address = self.blockchain_registry.agent_accounts[self.agent_id]['address']
                
                # Register address with transport
                BlockchainTransport = get_blockchain_transport()
                if isinstance(self.transport, BlockchainTransport):
                    self.transport.register_agent_address(self.agent_id, self.blockchain_address)
                
                logger.info(f"Agent {self.agent_id} registered on blockchain with tx: {tx_hash}")
            else:
                logger.error(f"Failed to register agent {self.agent_id} on blockchain")
                
        except Exception as e:
            logger.error(f"Blockchain registration failed for {self.agent_id}: {e}")
    
    def _create_tools(self):
        """Create blockchain-specific tools for Strands framework"""
        from strands import tool
        
        # Get base A2A tools from parent
        base_tools = super()._create_tools()
        
        @tool
        def register_agent_on_blockchain() -> Dict[str, Any]:
            """Register this agent on the blockchain"""
            try:
                self._register_on_blockchain()
                return {"success": True, "registered": self.blockchain_registered, "address": self.blockchain_address}
                
            except Exception as e:
                logger.error(f"Blockchain registration error: {e}")
                return {"success": False, "error": str(e)}
        
        @tool
        def get_blockchain_status() -> Dict[str, Any]:
            """Get blockchain connection and agent registration status"""
            try:
                status = {
                    "agent_registered": self.blockchain_registered,
                    "blockchain_address": self.blockchain_address,
                    "private_key_configured": bool(self.private_key),
                    "transport_type": type(self.transport).__name__,
                    "agent_id": self.agent_id
                }
                return {"success": True, "blockchain_status": status}
                
            except Exception as e:
                logger.error(f"Blockchain status error: {e}")
                return {"success": False, "error": str(e)}
        
        # Combine base tools with blockchain-specific tools
        blockchain_tools = [register_agent_on_blockchain, get_blockchain_status]
        return base_tools + blockchain_tools
    
    # Blockchain message handling is now handled by A2A layer via transport
    # These methods remain for backward compatibility and blockchain-specific operations
    
    async def send_blockchain_message(
        self,
        target_agent_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 0,
        workflow_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Send A2A message via blockchain (delegates to A2A layer)"""
        if not self.blockchain_registered:
            logger.error(f"Agent {self.agent_id} not registered on blockchain")
            return None
        
        # Use A2A layer's send method which will use blockchain transport
        return await self.send_a2a_message(
            target_agent_id=target_agent_id,
            message_type=message_type,
            payload=payload,
            priority=priority,
            workflow_context=workflow_context
        )
    
    def update_blockchain_status(self, status: str):
        """Update agent status on blockchain and A2A layer"""
        if not self.blockchain_registered:
            return
        
        try:
            # Update blockchain status
            tx_hash = self.blockchain_registry.update_agent_status_blockchain(
                self.agent_id, status
            )
            
            if tx_hash:
                logger.info(f"Status updated on blockchain: {status}")
                
                # Also update A2A agent status
                if status == "active":
                    self.update_agent_status(AgentStatus.ACTIVE)
                elif status == "busy":
                    self.update_agent_status(AgentStatus.BUSY)
                elif status == "inactive":
                    self.update_agent_status(AgentStatus.INACTIVE)
                elif status == "error":
                    self.update_agent_status(AgentStatus.ERROR)
            
        except Exception as e:
            logger.error(f"Failed to update blockchain status: {e}")
    
    async def execute_blockchain_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow via blockchain coordination"""
        if not self.blockchain_workflow_registry:
            logger.error("No blockchain workflow registry available")
            return {"success": False, "error": "No blockchain workflow registry"}
        
        try:
            # Get workflow definition
            workflow = self.blockchain_workflow_registry.get(workflow_id)
            if not workflow:
                return {"success": False, "error": f"Workflow {workflow_id} not found"}
            
            # Update status to busy
            self.update_blockchain_status("busy")
            
            # Execute workflow steps via blockchain messages
            execution_results = {}
            completed_steps = set()
            
            for step in workflow.steps:
                # Check dependencies
                if all(dep in completed_steps for dep in step.depends_on):
                    # Send step execution request via blockchain
                    step_payload = {
                        "workflow_id": workflow_id,
                        "step_id": step.step_id,
                        "action": step.action,
                        "input_data": {**step.input_data, **input_data},
                        "execution_context": {
                            "executor": self.agent_id,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    # Send message to target agent via A2A
                    message_id = await self.send_a2a_message(
                        target_agent_id=step.agent_id,
                        message_type=MessageType.WORKFLOW_REQUEST,
                        payload=step_payload,
                        priority=1,
                        workflow_context={"workflow_id": workflow_id, "step_id": step.step_id}
                    )
                    
                    if message_id:
                        execution_results[step.step_id] = {
                            "status": "sent",
                            "message_id": message_id
                        }
                        completed_steps.add(step.step_id)
            
            # Update status back to active
            self.update_blockchain_status("active")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "execution_results": execution_results,
                "completed_steps": list(completed_steps)
            }
            
        except Exception as e:
            logger.error(f"Blockchain workflow execution failed: {e}")
            self.update_blockchain_status("error")
            return {"success": False, "error": str(e)}
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get blockchain integration status"""
        if not self.blockchain_registry:
            return {"blockchain_enabled": False}
        
        blockchain_info = self.blockchain_registry.get_blockchain_info()
        a2a_info = self.get_a2a_info()
        
        return {
            "blockchain_enabled": True,
            "registered": self.blockchain_registered,
            "address": self.blockchain_address,
            "agent_id": self.agent_id,
            "blockchain_info": blockchain_info,
            "a2a_info": a2a_info,
            "transport_type": "BlockchainTransport"
        }
    
    async def shutdown(self):
        """Gracefully shutdown agent and blockchain connections"""
        logger.info(f"Shutting down blockchain agent {self.agent_id}")
        
        # Update status to inactive
        if self.blockchain_registered:
            self.update_blockchain_status("inactive")
        
        # Call A2A shutdown
        await super().shutdown()
        
        logger.info(f"Blockchain agent {self.agent_id} shutdown complete")


# Factory function for creating blockchain agents
def create_blockchain_agent(
    agent_id: str,
    agent_type: str,
    capabilities: List[str],
    private_key: str,
    model_provider: str = "deepseek"
) -> BlockchainStrandsAgent:
    """Factory function to create blockchain-integrated agents"""
    
    agent = BlockchainStrandsAgent(
        agent_id=agent_id,
        agent_type=agent_type,
        capabilities=capabilities,
        private_key=private_key,
        model_provider=model_provider,
        auto_register=True
    )
    
    logger.info(f"Created blockchain agent: {agent_id}")
    return agent