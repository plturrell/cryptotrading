"""
Production-ready persistent registry using existing database with Redis caching
Integrates with your SQLite database and uses Redis only for performance caching
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from ...data.database.cache import cache_manager, cache_result
from ...data.database.client import get_db
from ...data.database.models import (
    A2AAgent,
    A2AConnection,
    A2AMessage,
    A2AWorkflow,
    A2AWorkflowExecution,
)
from ..security.auth import Permission, auth_manager, require_auth
from ..security.validation import ValidationSchemas, request_validator

logger = logging.getLogger(__name__)


class PersistentAgentRegistry:
    """Production agent registry with database persistence and Redis caching"""

    def __init__(self):
        self.db = get_db()
        self.cache = cache_manager

    @cache_result(ttl=1800, key_func=lambda self, agent_id: f"agent_{agent_id}")
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information with caching"""
        # Try cache first (handled by decorator)

        # Query database
        with self.db.get_session() as session:
            agent = (
                session.query(A2AAgent)
                .filter(A2AAgent.agent_id == agent_id, A2AAgent.is_active == True)
                .first()
            )

            if agent:
                agent_data = {
                    "id": agent.agent_id,
                    "type": agent.agent_type,
                    "capabilities": json.loads(agent.capabilities),
                    "config": json.loads(agent.config) if agent.config else {},
                    "status": agent.status,
                    "blockchain_address": agent.blockchain_address,
                    "registered_at": agent.registered_at.isoformat(),
                    "last_updated": agent.last_updated.isoformat(),
                    "last_heartbeat": agent.last_heartbeat.isoformat()
                    if agent.last_heartbeat
                    else None,
                }
                return agent_data

        return None

    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        config: Dict[str, Any] = None,
        blockchain_address: str = None,
        current_user=None,
    ) -> bool:
        """Register agent with database persistence"""
        try:
            with self.db.get_session() as session:
                # Check if agent already exists
                existing = session.query(A2AAgent).filter(A2AAgent.agent_id == agent_id).first()

                if existing:
                    if existing.is_active:
                        logger.warning(f"Agent {agent_id} already registered and active")
                        return False
                    else:
                        # Reactivate existing agent
                        existing.is_active = True
                        existing.agent_type = agent_type
                        existing.capabilities = json.dumps(capabilities)
                        existing.config = json.dumps(config or {})
                        existing.status = "active"
                        existing.blockchain_address = blockchain_address
                        existing.last_updated = datetime.utcnow()
                else:
                    # Create new agent
                    agent = A2AAgent(
                        agent_id=agent_id,
                        agent_type=agent_type,
                        capabilities=json.dumps(capabilities),
                        config=json.dumps(config or {}),
                        blockchain_address=blockchain_address,
                        status="active",
                    )
                    session.add(agent)

                session.commit()

                # Invalidate cache
                self.cache.cache.delete(f"agent_{agent_id}")
                self.cache.cache.clear_pattern("agents_by_capability:*")

                logger.info(f"Agent {agent_id} registered successfully")
                return True

        except IntegrityError as e:
            logger.error(f"Database integrity error registering agent {agent_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error registering agent {agent_id}: {e}")
            return False

    @cache_result(ttl=900, key_func=lambda self, capability: f"capability_{capability}")
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents with specific capability"""
        with self.db.get_session() as session:
            agents = (
                session.query(A2AAgent)
                .filter(
                    A2AAgent.is_active == True,
                    A2AAgent.capabilities.contains(f'"{capability}"'),  # JSON contains
                )
                .all()
            )

            return [agent.agent_id for agent in agents]

    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all active agents"""
        cached_agents = self.cache.cache.get("all_active_agents")
        if cached_agents:
            return cached_agents

        agents = {}
        with self.db.get_session() as session:
            db_agents = session.query(A2AAgent).filter(A2AAgent.is_active == True).all()

            for agent in db_agents:
                agents[agent.agent_id] = {
                    "id": agent.agent_id,
                    "type": agent.agent_type,
                    "capabilities": json.loads(agent.capabilities),
                    "config": json.loads(agent.config) if agent.config else {},
                    "status": agent.status,
                    "registered_at": agent.registered_at.isoformat(),
                }

        # Cache for 5 minutes
        self.cache.cache.set("all_active_agents", agents, 300)
        return agents

    def update_agent_status(self, agent_id: str, status: str) -> bool:
        """Update agent status"""
        try:
            with self.db.get_session() as session:
                agent = session.query(A2AAgent).filter(A2AAgent.agent_id == agent_id).first()

                if agent:
                    agent.status = status
                    agent.last_updated = datetime.utcnow()

                    if status in ["active", "busy"]:
                        agent.last_heartbeat = datetime.utcnow()

                    session.commit()

                    # Invalidate caches
                    self.cache.cache.delete(f"agent_{agent_id}")
                    self.cache.cache.delete("all_active_agents")

                    return True

            return False

        except Exception as e:
            logger.error(f"Error updating agent status: {e}")
            return False

    def establish_connection(self, agent1_id: str, agent2_id: str, protocol: str) -> bool:
        """Establish connection between agents"""
        try:
            connection_id = f"{agent1_id}-{agent2_id}"

            with self.db.get_session() as session:
                # Check if connection already exists
                existing = (
                    session.query(A2AConnection)
                    .filter(A2AConnection.connection_id == connection_id)
                    .first()
                )

                if existing:
                    existing.status = "active"
                    existing.last_used = datetime.utcnow()
                else:
                    connection = A2AConnection(
                        connection_id=connection_id,
                        agent1_id=agent1_id,
                        agent2_id=agent2_id,
                        protocol=protocol,
                        status="active",
                    )
                    session.add(connection)

                session.commit()

                # Invalidate connection cache
                self.cache.cache.clear_pattern(f"connections:{agent1_id}:*")
                self.cache.cache.clear_pattern(f"connections:{agent2_id}:*")

                return True

        except Exception as e:
            logger.error(f"Error establishing connection: {e}")
            return False

    @cache_result(ttl=600, key_func=lambda self, agent_id: f"connections_{agent_id}")
    def get_agent_connections(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all connections for an agent"""
        connections = []

        with self.db.get_session() as session:
            db_connections = (
                session.query(A2AConnection)
                .filter(
                    (A2AConnection.agent1_id == agent_id) | (A2AConnection.agent2_id == agent_id),
                    A2AConnection.status == "active",
                )
                .all()
            )

            for conn in db_connections:
                connections.append(
                    {
                        "connection_id": conn.connection_id,
                        "agents": [conn.agent1_id, conn.agent2_id],
                        "protocol": conn.protocol,
                        "established_at": conn.established_at.isoformat(),
                        "last_used": conn.last_used.isoformat() if conn.last_used else None,
                    }
                )

        return connections


class PersistentWorkflowRegistry:
    """Production workflow registry with database persistence"""

    def __init__(self):
        self.db = get_db()
        self.cache = cache_manager

    @cache_result(ttl=3600, key_func=lambda self, workflow_id: f"workflow_{workflow_id}")
    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow definition with caching"""
        with self.db.get_session() as session:
            workflow = (
                session.query(A2AWorkflow)
                .filter(A2AWorkflow.workflow_id == workflow_id, A2AWorkflow.is_active == True)
                .first()
            )

            if workflow:
                return {
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "definition": json.loads(workflow.definition),
                    "version": workflow.version,
                    "created_by": workflow.created_by,
                    "created_at": workflow.created_at.isoformat(),
                    "updated_at": workflow.updated_at.isoformat(),
                }

        return None

    def register_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str,
        definition: Dict[str, Any],
        created_by: str = None,
        current_user=None,
    ) -> bool:
        """Register workflow with database persistence"""
        try:
            with self.db.get_session() as session:
                existing = (
                    session.query(A2AWorkflow)
                    .filter(A2AWorkflow.workflow_id == workflow_id)
                    .first()
                )

                if existing:
                    if existing.is_active:
                        # Update existing workflow (version bump)
                        existing.version += 1
                        existing.definition = json.dumps(definition)
                        existing.description = description
                        existing.updated_at = datetime.utcnow()
                    else:
                        # Reactivate
                        existing.is_active = True
                        existing.definition = json.dumps(definition)
                        existing.updated_at = datetime.utcnow()
                else:
                    workflow = A2AWorkflow(
                        workflow_id=workflow_id,
                        name=name,
                        description=description,
                        definition=json.dumps(definition),
                        created_by=created_by or (current_user.user_id if current_user else None),
                    )
                    session.add(workflow)

                session.commit()

                # Invalidate cache
                self.cache.cache.delete(f"workflow_{workflow_id}")
                self.cache.cache.delete("all_workflows")

                logger.info(f"Workflow {workflow_id} registered successfully")
                return True

        except Exception as e:
            logger.error(f"Error registering workflow {workflow_id}: {e}")
            return False

    def list_workflows(self) -> List[Dict[str, str]]:
        """List all active workflows"""
        cached_list = self.cache.cache.get("all_workflows")
        if cached_list:
            return cached_list

        workflows = []
        with self.db.get_session() as session:
            db_workflows = session.query(A2AWorkflow).filter(A2AWorkflow.is_active == True).all()

            for wf in db_workflows:
                definition = json.loads(wf.definition)
                workflows.append(
                    {
                        "id": wf.workflow_id,
                        "name": wf.name,
                        "description": wf.description,
                        "steps": len(definition.get("steps", [])),
                        "version": wf.version,
                    }
                )

        # Cache for 1 hour
        self.cache.cache.set("all_workflows", workflows, 3600)
        return workflows

    def create_execution(
        self, workflow_id: str, input_data: Dict[str, Any], created_by: str = None
    ) -> str:
        """Create workflow execution record"""
        execution_id = f"exec-{datetime.now().timestamp()}"

        try:
            with self.db.get_session() as session:
                execution = A2AWorkflowExecution(
                    execution_id=execution_id,
                    workflow_id=workflow_id,
                    input_data=json.dumps(input_data),
                    created_by=created_by,
                    status="pending",
                )
                session.add(execution)
                session.commit()

                logger.info(f"Workflow execution {execution_id} created")
                return execution_id

        except Exception as e:
            logger.error(f"Error creating workflow execution: {e}")
            raise

    def update_execution_status(
        self,
        execution_id: str,
        status: str,
        result_data: Dict[str, Any] = None,
        error_message: str = None,
    ) -> bool:
        """Update workflow execution status"""
        try:
            with self.db.get_session() as session:
                execution = (
                    session.query(A2AWorkflowExecution)
                    .filter(A2AWorkflowExecution.execution_id == execution_id)
                    .first()
                )

                if execution:
                    execution.status = status

                    if result_data:
                        execution.result_data = json.dumps(result_data)

                    if error_message:
                        execution.error_message = error_message

                    if status in ["completed", "failed"]:
                        execution.completed_at = datetime.utcnow()

                    session.commit()

                    # Invalidate execution cache
                    self.cache.cache.delete(f"execution_{execution_id}")

                    return True

            return False

        except Exception as e:
            logger.error(f"Error updating execution status: {e}")
            return False

    @cache_result(ttl=300, key_func=lambda self, execution_id: f"execution_{execution_id}")
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status"""
        with self.db.get_session() as session:
            execution = (
                session.query(A2AWorkflowExecution)
                .filter(A2AWorkflowExecution.execution_id == execution_id)
                .first()
            )

            if execution:
                return {
                    "execution_id": execution.execution_id,
                    "workflow_id": execution.workflow_id,
                    "status": execution.status,
                    "input_data": json.loads(execution.input_data)
                    if execution.input_data
                    else None,
                    "result_data": json.loads(execution.result_data)
                    if execution.result_data
                    else None,
                    "error_message": execution.error_message,
                    "started_at": execution.started_at.isoformat(),
                    "completed_at": execution.completed_at.isoformat()
                    if execution.completed_at
                    else None,
                    "created_by": execution.created_by,
                }

        return None


class PersistentMessageLog:
    """A2A message logging with database persistence"""

    def __init__(self):
        self.db = get_db()

    def log_message(
        self,
        message_id: str,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        correlation_id: str = None,
        workflow_context: Dict[str, Any] = None,
    ) -> bool:
        """Log A2A message to database"""
        try:
            with self.db.get_session() as session:
                message = A2AMessage(
                    message_id=message_id,
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    message_type=message_type,
                    payload=json.dumps(payload),
                    priority=priority,
                    correlation_id=correlation_id,
                    workflow_context=json.dumps(workflow_context) if workflow_context else None,
                    status="sent",
                )
                session.add(message)
                session.commit()

                return True

        except Exception as e:
            logger.error(f"Error logging message {message_id}: {e}")
            return False

    def update_message_status(
        self, message_id: str, status: str, error_message: str = None
    ) -> bool:
        """Update message processing status"""
        try:
            with self.db.get_session() as session:
                message = (
                    session.query(A2AMessage).filter(A2AMessage.message_id == message_id).first()
                )

                if message:
                    message.status = status
                    if error_message:
                        message.error_message = error_message
                    if status == "processed":
                        message.processed_at = datetime.utcnow()

                    session.commit()
                    return True

            return False

        except Exception as e:
            logger.error(f"Error updating message status: {e}")
            return False


# Global instances
persistent_agent_registry = PersistentAgentRegistry()
persistent_workflow_registry = PersistentWorkflowRegistry()
persistent_message_log = PersistentMessageLog()
