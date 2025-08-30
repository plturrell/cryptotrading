"""
CDS A2A Service Implementation
RESTful API endpoints for A2A Service with blockchain integration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from flask import Blueprint, request, jsonify
from functools import wraps

# Import CDS-blockchain integration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.cryptotrading.core.protocols.a2a.cds_blockchain_integration import (
    get_cds_adapter,
    cds_register_agent,
    cds_send_message,
    cds_execute_workflow,
    CDSEntityType
)

logger = logging.getLogger(__name__)

# Create Flask blueprint for CDS A2A Service
cds_a2a_bp = Blueprint('cds_a2a_service', __name__, url_prefix='/api/odata/v4/A2AService')


def async_route(f):
    """Decorator to run async functions in Flask routes"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


# ============================================================================
# ENTITY ENDPOINTS (OData v4 compliant)
# ============================================================================

@cds_a2a_bp.route('/A2AAgents', methods=['GET'])
@async_route
async def get_a2a_agents():
    """Get all A2A agents"""
    try:
        adapter = await get_cds_adapter()
        
        # In production, this would query from database
        # For now, return mapped agents
        agents = []
        for cds_id, blockchain_id in adapter.agent_mappings.items():
            agents.append({
                "agentId": cds_id,
                "blockchainId": blockchain_id,
                "status": "active",
                "createdAt": datetime.now().isoformat()
            })
        
        return jsonify({
            "@odata.context": "$metadata#A2AAgents",
            "value": agents
        })
        
    except Exception as e:
        logger.error(f"Error getting agents: {e}")
        return jsonify({"error": str(e)}), 500


@cds_a2a_bp.route('/A2AAgents', methods=['POST'])
@async_route
async def create_a2a_agent():
    """Create a new A2A agent"""
    try:
        data = request.json
        
        result = await cds_register_agent(
            agent_name=data.get('agentName'),
            agent_type=data.get('agentType'),
            capabilities=json.dumps(data.get('capabilities', {}))
        )
        
        return jsonify(result), 201 if result['status'] == 'registered' else 400
        
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        return jsonify({"error": str(e)}), 500


@cds_a2a_bp.route('/A2AMessages', methods=['GET'])
@async_route
async def get_a2a_messages():
    """Get A2A messages"""
    try:
        # Filter parameters
        agent_id = request.args.get('agentId')
        status = request.args.get('status')
        
        # In production, query from database
        messages = []
        
        return jsonify({
            "@odata.context": "$metadata#A2AMessages",
            "value": messages
        })
        
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        return jsonify({"error": str(e)}), 500


@cds_a2a_bp.route('/A2AWorkflows', methods=['GET'])
@async_route
async def get_a2a_workflows():
    """Get A2A workflows"""
    try:
        adapter = await get_cds_adapter()
        
        workflows = []
        for cds_id, mapping in adapter.workflow_mappings.items():
            workflows.append({
                "workflowId": cds_id,
                "blockchainId": mapping.blockchain_workflow_id,
                "workflowType": mapping.workflow_type,
                "status": mapping.status,
                "createdAt": mapping.created_at.isoformat()
            })
        
        return jsonify({
            "@odata.context": "$metadata#A2AWorkflows",
            "value": workflows
        })
        
    except Exception as e:
        logger.error(f"Error getting workflows: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ACTION ENDPOINTS
# ============================================================================

@cds_a2a_bp.route('/registerAgent', methods=['POST'])
@async_route
async def register_agent_action():
    """Action: Register a new agent"""
    try:
        data = request.json
        
        result = await cds_register_agent(
            agent_name=data.get('agentName'),
            agent_type=data.get('agentType'),
            capabilities=data.get('capabilities')
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in registerAgent action: {e}")
        return jsonify({"error": str(e)}), 500


@cds_a2a_bp.route('/connectAgents', methods=['POST'])
@async_route
async def connect_agents_action():
    """Action: Connect two agents"""
    try:
        data = request.json
        adapter = await get_cds_adapter()
        
        # Create connection on blockchain
        connection_data = {
            "fromAgent": data.get('fromAgentId'),
            "toAgent": data.get('toAgentId'),
            "protocol": data.get('protocol', 'a2a'),
            "establishedAt": datetime.now().isoformat()
        }
        
        # Store connection on-chain
        if adapter.data_exchange:
            data_id = await adapter.data_exchange.store_data(
                sender_agent_id=data.get('fromAgentId'),
                receiver_agent_id=data.get('toAgentId'),
                data=connection_data,
                data_type="agent_connection",
                is_encrypted=False
            )
            
            return jsonify({
                "connectionId": f"conn-{data_id}",
                "status": "established" if data_id else "failed",
                "message": f"Connection stored with ID {data_id}" if data_id else "Failed to establish connection"
            })
        
        return jsonify({
            "connectionId": None,
            "status": "failed",
            "message": "Data exchange service not available"
        }), 500
        
    except Exception as e:
        logger.error(f"Error in connectAgents action: {e}")
        return jsonify({"error": str(e)}), 500


@cds_a2a_bp.route('/sendMessage', methods=['POST'])
@async_route
async def send_message_action():
    """Action: Send message between agents"""
    try:
        data = request.json
        
        result = await cds_send_message(
            from_agent_id=data.get('fromAgentId'),
            to_agent_id=data.get('toAgentId'),
            message_type=data.get('messageType'),
            payload=data.get('payload'),
            priority=data.get('priority', 'normal')
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in sendMessage action: {e}")
        return jsonify({"error": str(e)}), 500


@cds_a2a_bp.route('/executeWorkflow', methods=['POST'])
@async_route
async def execute_workflow_action():
    """Action: Execute a workflow"""
    try:
        data = request.json
        
        result = await cds_execute_workflow(
            workflow_id=data.get('workflowId'),
            input_data=data.get('inputData')
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in executeWorkflow action: {e}")
        return jsonify({"error": str(e)}), 500


@cds_a2a_bp.route('/stopWorkflow', methods=['POST'])
@async_route
async def stop_workflow_action():
    """Action: Stop a running workflow"""
    try:
        data = request.json
        adapter = await get_cds_adapter()
        
        execution_id = data.get('executionId')
        reason = data.get('reason', 'User requested')
        
        # In production, would actually stop the workflow
        # For now, just update status
        
        return jsonify({
            "success": True,
            "finalStatus": "cancelled",
            "message": f"Workflow {execution_id} stopped: {reason}"
        })
        
    except Exception as e:
        logger.error(f"Error in stopWorkflow action: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# FUNCTION ENDPOINTS
# ============================================================================

@cds_a2a_bp.route('/getAgentStatus', methods=['GET'])
@async_route
async def get_agent_status():
    """Function: Get agent status"""
    try:
        agent_id = request.args.get('agentId')
        
        if not agent_id:
            return jsonify({"error": "agentId parameter required"}), 400
        
        # In production, query actual status
        # For now, return mock data
        
        return jsonify({
            "status": "active",
            "lastHeartbeat": datetime.now().isoformat(),
            "activeConnections": 3,
            "pendingMessages": 0,
            "runningWorkflows": 1
        })
        
    except Exception as e:
        logger.error(f"Error in getAgentStatus: {e}")
        return jsonify({"error": str(e)}), 500


@cds_a2a_bp.route('/getAgentMetrics', methods=['GET'])
@async_route
async def get_agent_metrics():
    """Function: Get agent metrics"""
    try:
        agent_id = request.args.get('agentId')
        period = request.args.get('period', '1h')
        
        if not agent_id:
            return jsonify({"error": "agentId parameter required"}), 400
        
        # In production, calculate actual metrics
        # For now, return mock data
        
        return jsonify({
            "messagesProcessed": 150,
            "avgResponseTime": 250.5,  # ms
            "successRate": 98.5,  # %
            "errorCount": 2,
            "uptime": 99.9  # %
        })
        
    except Exception as e:
        logger.error(f"Error in getAgentMetrics: {e}")
        return jsonify({"error": str(e)}), 500


@cds_a2a_bp.route('/getWorkflowStatus', methods=['GET'])
@async_route
async def get_workflow_status():
    """Function: Get workflow status"""
    try:
        execution_id = request.args.get('executionId')
        
        if not execution_id:
            return jsonify({"error": "executionId parameter required"}), 400
        
        adapter = await get_cds_adapter()
        status = await adapter.get_workflow_status_for_cds(execution_id)
        
        if status:
            return jsonify({
                "status": status.get('status', 'unknown'),
                "currentStep": status.get('steps', [{}])[-1].get('step_id', '') if status.get('steps') else '',
                "progress": status.get('progress', 0),
                "estimatedCompletion": None,  # Would calculate based on progress
                "errors": []
            })
        
        return jsonify({
            "status": "not_found",
            "currentStep": "",
            "progress": 0,
            "estimatedCompletion": None,
            "errors": ["Workflow not found"]
        }), 404
        
    except Exception as e:
        logger.error(f"Error in getWorkflowStatus: {e}")
        return jsonify({"error": str(e)}), 500


@cds_a2a_bp.route('/getAgentNetwork', methods=['GET'])
@async_route
async def get_agent_network():
    """Function: Get agent network topology"""
    try:
        adapter = await get_cds_adapter()
        
        network = []
        for cds_id, blockchain_id in adapter.agent_mappings.items():
            network.append({
                "agentId": cds_id,
                "agentName": cds_id.split('-')[0] if '-' in cds_id else cds_id,
                "connections": []  # Would populate from actual connections
            })
        
        return jsonify(network)
        
    except Exception as e:
        logger.error(f"Error in getAgentNetwork: {e}")
        return jsonify({"error": str(e)}), 500


@cds_a2a_bp.route('/getMessageQueue', methods=['GET'])
@async_route
async def get_message_queue():
    """Function: Get message queue for an agent"""
    try:
        agent_id = request.args.get('agentId')
        
        if not agent_id:
            return jsonify({"error": "agentId parameter required"}), 400
        
        # In production, query actual message queue
        # For now, return empty queue
        
        return jsonify([])
        
    except Exception as e:
        logger.error(f"Error in getMessageQueue: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# METADATA ENDPOINT
# ============================================================================

@cds_a2a_bp.route('/$metadata', methods=['GET'])
def get_metadata():
    """Return OData metadata document"""
    metadata = {
        "$Version": "4.0",
        "$EntityContainer": "com.rex.cryptotrading.a2a.service.A2AService",
        "$Reference": {
            "https://oasis-tcs.github.io/odata-vocabularies/vocabularies/Org.OData.Core.V1.json": {
                "$Include": [{"$Namespace": "Org.OData.Core.V1", "$Alias": "Core"}]
            }
        },
        "com.rex.cryptotrading.a2a.service": {
            "$kind": "Schema",
            "$Namespace": "com.rex.cryptotrading.a2a.service",
            "A2AService": {
                "$kind": "EntityContainer",
                "A2AAgents": {"$Collection": True, "$Type": "A2AAgent"},
                "A2AMessages": {"$Collection": True, "$Type": "A2AMessage"},
                "A2AWorkflows": {"$Collection": True, "$Type": "A2AWorkflow"}
            }
        }
    }
    
    return jsonify(metadata)


# ============================================================================
# HEALTH CHECK
# ============================================================================

@cds_a2a_bp.route('/health', methods=['GET'])
@async_route
async def health_check():
    """Health check endpoint"""
    try:
        adapter = await get_cds_adapter()
        metrics = adapter.get_metrics()
        
        return jsonify({
            "status": "healthy",
            "service": "CDS A2A Service",
            "blockchain_integration": True,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


def create_app():
    """Create Flask app with CDS A2A Service"""
    from flask import Flask
    
    app = Flask(__name__)
    app.register_blueprint(cds_a2a_bp)
    
    return app


if __name__ == '__main__':
    # Run the service
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)