"""
CDS Client for A2A Agents
Provides seamless integration between Python agents and CDS services
"""

import asyncio
import aiohttp
import json
import websockets
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urljoin
import uuid

logger = logging.getLogger(__name__)

@dataclass
class CDSServiceConfig:
    """Configuration for CDS service connection"""
    base_url: str = "http://localhost:4004"  # Changed from 4005 to 4004
    odata_path: str = "/api/odata/v4"
    websocket_path: str = "/a2a/ws"
    service_name: str = "A2AService"
    timeout: int = 30
    retry_attempts: int = 3

@dataclass
class CDSTransaction:
    """CDS transaction context"""
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=datetime.utcnow)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    committed: bool = False
    rolled_back: bool = False

class CDSClient:
    """
    CDS Client for A2A Agents
    Provides OData, WebSocket, and event integration with CDS services
    """
    
    def __init__(self, config: CDSServiceConfig = None):
        self.config = config or CDSServiceConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.agent_id: Optional[str] = None
        self.event_handlers: Dict[str, Callable] = {}
        self.connected = False
        self.current_transaction: Optional[CDSTransaction] = None
        
        # Service URLs
        self.odata_url = urljoin(self.config.base_url, self.config.odata_path)
        self.service_url = urljoin(self.odata_url, self.config.service_name)
        self.websocket_url = self.config.base_url.replace('http', 'ws') + self.config.websocket_path
        
        logger.info(f"CDS Client initialized - Service: {self.service_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self, agent_id: str = None):
        """Connect to CDS services and establish WebSocket connection"""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
            # Test HTTP connection
            await self._test_connection()
            
            # Establish WebSocket connection if agent_id provided
            if agent_id:
                await self._connect_websocket(agent_id)
            
            self.connected = True
            logger.info("CDS Client connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to CDS services: {e}")
            await self.disconnect()
            raise
    
    async def disconnect(self):
        """Disconnect from CDS services"""
        try:
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            if self.session:
                await self.session.close()
                self.session = None
            
            self.connected = False
            logger.info("CDS Client disconnected")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def _test_connection(self):
        """Test HTTP connection to CDS service"""
        try:
            url = urljoin(self.config.base_url, "/health")
            async with self.session.get(url) as response:
                if response.status == 200:
                    logger.debug("CDS service health check passed")
                else:
                    raise Exception(f"Health check failed with status {response.status}")
        except Exception as e:
            logger.error(f"CDS service health check failed: {e}")
            raise
    
    async def _connect_websocket(self, agent_id: str):
        """Establish WebSocket connection for real-time events"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            self.agent_id = agent_id
            
            # Register agent with WebSocket server
            await self.websocket.send(json.dumps({
                'type': 'AGENT_REGISTER',
                'payload': {'agentId': agent_id}
            }))
            
            # Wait for registration confirmation
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get('type') == 'AGENT_REGISTERED':
                logger.info(f"Agent {agent_id} registered with CDS WebSocket")
                
                # Start listening for events
                asyncio.create_task(self._websocket_listener())
                
                # Start heartbeat
                asyncio.create_task(self._heartbeat_sender())
            else:
                raise Exception(f"Agent registration failed: {response_data}")
                
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.websocket = None
            raise
    
    async def _websocket_listener(self):
        """Listen for WebSocket messages from CDS"""
        try:
            while self.websocket and not self.websocket.closed:
                try:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    
                    message_type = data.get('type')
                    if message_type in self.event_handlers:
                        await self.event_handlers[message_type](data)
                    elif message_type == 'MESSAGE':
                        await self._handle_agent_message(data)
                    elif message_type == 'EVENT':
                        await self._handle_cds_event(data)
                    else:
                        logger.debug(f"Unhandled WebSocket message: {message_type}")
                        
                except websockets.exceptions.ConnectionClosed:
                    logger.info("WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"WebSocket listener error: {e}")
                    
        except Exception as e:
            logger.error(f"WebSocket listener failed: {e}")
    
    async def _heartbeat_sender(self):
        """Send periodic heartbeat to CDS"""
        while self.websocket and hasattr(self.websocket, 'close_code') and self.websocket.close_code is None:
            try:
                await self.websocket.send(json.dumps({
                    'type': 'HEARTBEAT',
                    'payload': {'timestamp': datetime.utcnow().isoformat()}
                }))
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                break
    
    async def _handle_agent_message(self, data: Dict[str, Any]):
        """Handle incoming agent messages"""
        message_data = data.get('data', {})
        logger.info(f"Received agent message: {message_data.get('messageType')}")
        
        # Acknowledge message receipt
        if message_data.get('ID'):
            await self.websocket.send(json.dumps({
                'type': 'MESSAGE_ACK',
                'payload': {'messageId': message_data['ID']}
            }))
    
    async def _handle_cds_event(self, data: Dict[str, Any]):
        """Handle CDS events"""
        event_type = data.get('event')
        event_data = data.get('data', {})
        
        logger.info(f"Received CDS event: {event_type}")
        
        # Forward to specific event handlers
        handler_name = f"on_{event_type.replace('.', '_')}"
        if hasattr(self, handler_name):
            handler = getattr(self, handler_name)
            await handler(event_data)
    
    def on_event(self, event_type: str, handler: Callable):
        """Register event handler"""
        self.event_handlers[event_type] = handler
    
    # ============== OData Operations ==============
    
    async def get(self, entity_set: str, filters: Dict[str, Any] = None, 
                  select: List[str] = None, expand: List[str] = None,
                  top: int = None, skip: int = None, count: bool = False) -> Dict[str, Any]:
        """Get entities from CDS service"""
        if not self.connected:
            raise Exception("Not connected to CDS service")
        
        url = f"{self.service_url}/{entity_set}"
        params = {}
        
        # Build OData query parameters
        if filters:
            filter_parts = []
            for key, value in filters.items():
                if isinstance(value, str):
                    filter_parts.append(f"{key} eq '{value}'")
                else:
                    filter_parts.append(f"{key} eq {value}")
            if filter_parts:
                params['$filter'] = ' and '.join(filter_parts)
        
        if select:
            params['$select'] = ','.join(select)
        
        if expand:
            params['$expand'] = ','.join(expand)
        
        if top:
            params['$top'] = str(top)
        
        if skip:
            params['$skip'] = str(skip)
        
        if count:
            params['$count'] = 'true'
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"GET request failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"GET operation failed: {e}")
            raise
    
    async def create(self, entity_set: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new entity in CDS service"""
        if not self.connected:
            raise Exception("Not connected to CDS service")
        
        url = f"{self.service_url}/{entity_set}"
        
        # Add to current transaction if exists
        if self.current_transaction:
            self.current_transaction.operations.append({
                'operation': 'CREATE',
                'entity_set': entity_set,
                'data': data
            })
        
        try:
            async with self.session.post(url, json=data) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"CREATE request failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"CREATE operation failed: {e}")
            raise
    
    async def update(self, entity_set: str, key: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update entity in CDS service"""
        if not self.connected:
            raise Exception("Not connected to CDS service")
        
        url = f"{self.service_url}/{entity_set}('{key}')"
        
        # Add to current transaction if exists
        if self.current_transaction:
            self.current_transaction.operations.append({
                'operation': 'UPDATE',
                'entity_set': entity_set,
                'key': key,
                'data': data
            })
        
        try:
            async with self.session.patch(url, json=data) as response:
                if response.status in [200, 204]:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    return {'success': True}
                else:
                    error_text = await response.text()
                    raise Exception(f"UPDATE request failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"UPDATE operation failed: {e}")
            raise
    
    async def delete(self, entity_set: str, key: str) -> bool:
        """Delete entity from CDS service"""
        if not self.connected:
            raise Exception("Not connected to CDS service")
        
        url = f"{self.service_url}/{entity_set}('{key}')"
        
        # Add to current transaction if exists
        if self.current_transaction:
            self.current_transaction.operations.append({
                'operation': 'DELETE',
                'entity_set': entity_set,
                'key': key
            })
        
        try:
            async with self.session.delete(url) as response:
                if response.status == 204:
                    return True
                else:
                    error_text = await response.text()
                    raise Exception(f"DELETE request failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"DELETE operation failed: {e}")
            raise
    
    # ============== CDS Actions ==============
    
    async def call_action(self, action_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call CDS service action"""
        if not self.connected:
            raise Exception("Not connected to CDS service")
        
        url = f"{self.service_url}/{action_name}"
        
        try:
            async with self.session.post(url, json=parameters or {}) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Action {action_name} failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Action {action_name} failed: {e}")
            raise
    
    async def call_function(self, function_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call CDS service function"""
        if not self.connected:
            raise Exception("Not connected to CDS service")
        
        # Build function URL with parameters
        param_string = ""
        if parameters:
            param_parts = []
            for key, value in parameters.items():
                if isinstance(value, str):
                    param_parts.append(f"{key}='{value}'")
                else:
                    param_parts.append(f"{key}={value}")
            param_string = f"({','.join(param_parts)})"
        
        url = f"{self.service_url}/{function_name}{param_string}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Function {function_name} failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Function {function_name} failed: {e}")
            raise
    
    # ============== A2A Service Specific Methods ==============
    
    async def register_agent(self, agent_name: str, agent_type: str, 
                           capabilities: List[str]) -> Dict[str, Any]:
        """Register agent with A2A service"""
        return await self.call_action('registerAgent', {
            'agentName': agent_name,
            'agentType': agent_type,
            'capabilities': capabilities
        })
    
    async def send_message(self, from_agent_id: str, to_agent_id: str,
                          message_type: str, payload: str, 
                          priority: str = 'MEDIUM') -> Dict[str, Any]:
        """Send message between agents"""
        return await self.call_action('sendMessage', {
            'fromAgentId': from_agent_id,
            'toAgentId': to_agent_id,
            'messageType': message_type,
            'payload': payload,
            'priority': priority
        })
    
    async def connect_agents(self, from_agent_id: str, to_agent_id: str,
                           protocol: str = 'A2A') -> Dict[str, Any]:
        """Connect two agents"""
        return await self.call_action('connectAgents', {
            'fromAgentId': from_agent_id,
            'toAgentId': to_agent_id,
            'protocol': protocol
        })
    
    async def execute_workflow(self, workflow_id: str, 
                             input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow"""
        return await self.call_action('executeWorkflow', {
            'workflowId': workflow_id,
            'inputData': json.dumps(input_data)
        })
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent status"""
        return await self.call_function('getAgentStatus', {
            'agentId': agent_id
        })
    
    async def get_agent_metrics(self, agent_id: str, 
                              period: str = '24h') -> Dict[str, Any]:
        """Get agent metrics"""
        return await self.call_function('getAgentMetrics', {
            'agentId': agent_id,
            'period': period
        })
    
    # ============== Transaction Support ==============
    
    async def begin_transaction(self) -> CDSTransaction:
        """Begin CDS transaction"""
        if self.current_transaction:
            raise Exception("Transaction already in progress")
        
        self.current_transaction = CDSTransaction()
        logger.info(f"Transaction started: {self.current_transaction.transaction_id}")
        return self.current_transaction
    
    async def commit_transaction(self) -> bool:
        """Commit current transaction"""
        if not self.current_transaction:
            raise Exception("No transaction in progress")
        
        try:
            # In a real implementation, this would submit all operations as a batch
            logger.info(f"Committing transaction: {self.current_transaction.transaction_id}")
            logger.info(f"Operations: {len(self.current_transaction.operations)}")
            
            self.current_transaction.committed = True
            self.current_transaction = None
            return True
            
        except Exception as e:
            logger.error(f"Transaction commit failed: {e}")
            await self.rollback_transaction()
            raise
    
    async def rollback_transaction(self) -> bool:
        """Rollback current transaction"""
        if not self.current_transaction:
            raise Exception("No transaction in progress")
        
        logger.info(f"Rolling back transaction: {self.current_transaction.transaction_id}")
        
        self.current_transaction.rolled_back = True
        self.current_transaction = None
        return True
    
    def transaction(self):
        """Transaction context manager"""
        return CDSTransactionContext(self)
    
    # ============== Event Publishing ==============
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to CDS event system"""
        if not self.connected:
            raise Exception("Not connected to CDS service")
        
        try:
            url = urljoin(self.config.base_url, "/a2a/events")
            payload = {
                'event': event_type,
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'source': self.agent_id or 'python_agent'
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Event emission failed: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to emit event {event_type}: {e}")
            raise

class CDSTransactionContext:
    """Transaction context manager"""
    
    def __init__(self, client: CDSClient):
        self.client = client
        self.transaction: Optional[CDSTransaction] = None
    
    async def __aenter__(self):
        self.transaction = await self.client.begin_transaction()
        return self.transaction
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self.client.commit_transaction()
        else:
            await self.client.rollback_transaction()

# ============== Agent Integration Helpers ==============

class A2AAgentCDSMixin:
    """Mixin class to add CDS integration to A2A agents with monitoring"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cds_client: Optional[CDSClient] = None
        self.cds_config = CDSServiceConfig()
        self._cds_monitor = None
    
    async def initialize_cds(self, agent_id: str = None):
        """Initialize CDS client connection with monitoring"""
        try:
            # Initialize monitoring
            from ...infrastructure.monitoring.cds_integration_monitor import (
                get_cds_monitor, CDSIntegrationStatus, CDSOperationType
            )
            self._cds_monitor = get_cds_monitor()
            
            actual_agent_id = agent_id or getattr(self, 'agent_id', 'unknown')
            self._cds_monitor.register_agent(actual_agent_id)
            self._cds_monitor.update_agent_status(actual_agent_id, CDSIntegrationStatus.CONNECTING)
            
            # Track connection operation
            async with self._cds_monitor.track_operation(actual_agent_id, CDSOperationType.CONNECTION) as operation:
                self.cds_client = CDSClient(self.cds_config)
                await self.cds_client.connect(actual_agent_id)
                
                # Register event handlers
                await self._setup_cds_event_handlers()
                
                operation.method_used = "CDS"
                self._cds_monitor.update_agent_status(actual_agent_id, CDSIntegrationStatus.CONNECTED)
            
            logger.info(f"CDS integration initialized for agent {actual_agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"CDS initialization failed: {e}")
            if self._cds_monitor:
                self._cds_monitor.update_agent_status(
                    agent_id or getattr(self, 'agent_id', 'unknown'), 
                    CDSIntegrationStatus.ERROR
                )
            return False
    
    async def _setup_cds_event_handlers(self):
        """Setup CDS event handlers"""
        if self.cds_client:
            # Register for agent events
            self.cds_client.on_event('MESSAGE', self._handle_cds_message)
            self.cds_client.on_event('WORKFLOW_UPDATE', self._handle_workflow_update)
    
    async def _handle_cds_message(self, data: Dict[str, Any]):
        """Handle incoming CDS messages"""
        logger.info(f"Agent received CDS message: {data}")
        
        # Override in subclasses to handle specific message types
        message_type = data.get('messageType')
        if hasattr(self, f'handle_{message_type}'):
            handler = getattr(self, f'handle_{message_type}')
            await handler(data)
    
    async def _handle_workflow_update(self, data: Dict[str, Any]):
        """Handle workflow updates from CDS"""
        logger.info(f"Workflow update: {data}")
    
    async def register_with_cds(self, capabilities: List[str]):
        """Register agent with CDS service with monitoring"""
        if not self.cds_client:
            raise Exception("CDS client not initialized")
        
        agent_id = getattr(self, 'agent_id', 'unknown')
        agent_type = getattr(self, 'agent_type', 'generic')
        
        if self._cds_monitor:
            from ...infrastructure.monitoring.cds_integration_monitor import CDSOperationType
            async with self._cds_monitor.track_operation(agent_id, CDSOperationType.AGENT_REGISTRATION) as operation:
                result = await self.cds_client.register_agent(
                    agent_name=agent_id,
                    agent_type=agent_type,
                    capabilities=capabilities
                )
                
                operation.method_used = "CDS"
                operation.payload_size = len(json.dumps({'capabilities': capabilities}).encode())
                operation.response_size = len(json.dumps(result).encode()) if result else 0
                
                logger.info(f"Agent registered with CDS: {result}")
                return result
        else:
            result = await self.cds_client.register_agent(
                agent_name=agent_id,
                agent_type=agent_type,
                capabilities=capabilities
            )
            logger.info(f"Agent registered with CDS: {result}")
            return result
    
    async def send_cds_message(self, to_agent_id: str, message_type: str, 
                              payload: Dict[str, Any], priority: str = 'MEDIUM'):
        """Send message via CDS with monitoring"""
        if not self.cds_client:
            raise Exception("CDS client not initialized")
        
        from_agent_id = getattr(self, 'agent_id', 'unknown')
        
        if self._cds_monitor:
            from ...infrastructure.monitoring.cds_integration_monitor import CDSOperationType
            async with self._cds_monitor.track_operation(from_agent_id, CDSOperationType.MESSAGE_SEND) as operation:
                payload_json = json.dumps(payload)
                result = await self.cds_client.send_message(
                    from_agent_id=from_agent_id,
                    to_agent_id=to_agent_id,
                    message_type=message_type,
                    payload=payload_json,
                    priority=priority
                )
                
                operation.method_used = "CDS"
                operation.payload_size = len(payload_json.encode())
                operation.response_size = len(json.dumps(result).encode()) if result else 0
                
                return result
        else:
            result = await self.cds_client.send_message(
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                message_type=message_type,
                payload=json.dumps(payload),
                priority=priority
            )
            return result
    
    async def emit_cds_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to CDS event system"""
        if not self.cds_client:
            raise Exception("CDS client not initialized")
        
        await self.cds_client.emit_event(event_type, data)
    
    async def cleanup_cds(self):
        """Cleanup CDS connection"""
        if self.cds_client:
            await self.cds_client.disconnect()
            self.cds_client = None

# Factory function
async def create_cds_client(agent_id: str = None, config: CDSServiceConfig = None) -> CDSClient:
    """Create and connect CDS client"""
    client = CDSClient(config or CDSServiceConfig())
    await client.connect(agent_id)
    return client