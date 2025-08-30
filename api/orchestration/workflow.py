"""
Vercel Edge Function for A2A Workflow Orchestration
Handles workflow execution requests
"""

import json
import logging
from typing import Dict, Any

from cryptotrading.core.protocols.a2a.orchestration import (
    OrchestrationService,
    WorkflowTemplates,
    workflow_registry,
)
from cryptotrading.core.protocols.a2a.orchestration.message_queue import workflow_queue

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global orchestration service
orchestration_service = None


async def get_orchestration_service():
    """Get or create orchestration service"""
    global orchestration_service
    if not orchestration_service:
        orchestration_service = OrchestrationService({"max_concurrent": 5})  # Limited for Vercel
        await orchestration_service.start()
    return orchestration_service


async def handler(request):
    """Handle workflow orchestration requests"""
    try:
        # Parse request
        method = request.method
        path = request.url.path

        if method == "GET" and path == "/api/orchestration/workflow":
            # List available workflows
            workflows = workflow_registry.list()
            return {
                "statusCode": 200,
                "body": json.dumps({"workflows": workflows, "count": len(workflows)}),
            }

        elif method == "POST" and path == "/api/orchestration/workflow/execute":
            # Execute workflow
            body = await request.json()
            workflow_id = body.get("workflow_id")
            input_data = body.get("input_data", {})

            if not workflow_id:
                return {"statusCode": 400, "body": json.dumps({"error": "workflow_id required"})}

            # Queue workflow for execution
            message_id = await workflow_queue.enqueue(
                {"type": "workflow_request", "workflow_id": workflow_id, "input_data": input_data}
            )

            return {
                "statusCode": 202,
                "body": json.dumps(
                    {
                        "message": "Workflow queued for execution",
                        "message_id": message_id,
                        "workflow_id": workflow_id,
                    }
                ),
            }

        elif method == "POST" and path == "/api/orchestration/workflow/eth-load":
            # Shortcut for ETH loading workflow
            body = await request.json()
            days_back = body.get("days_back", 30)

            service = await get_orchestration_service()
            execution_id = await WorkflowTemplates.load_eth_data(service, days_back=days_back)

            return {
                "statusCode": 202,
                "body": json.dumps(
                    {
                        "message": "ETH data loading started",
                        "execution_id": execution_id,
                        "days_back": days_back,
                    }
                ),
            }

        elif method == "GET" and path.startswith("/api/orchestration/workflow/status/"):
            # Get workflow execution status
            execution_id = path.split("/")[-1]

            service = await get_orchestration_service()
            status = service.get_execution_status(execution_id)

            if status:
                return {"statusCode": 200, "body": json.dumps(status)}
            else:
                return {"statusCode": 404, "body": json.dumps({"error": "Execution not found"})}

        elif method == "GET" and path == "/api/orchestration/metrics":
            # Get orchestration metrics
            service = await get_orchestration_service()
            metrics = service.get_metrics()

            # Add queue stats
            workflow_stats = await workflow_queue.get_stats()
            metrics["queues"] = {"workflow": workflow_stats}

            return {"statusCode": 200, "body": json.dumps(metrics)}

        else:
            return {"statusCode": 404, "body": json.dumps({"error": "Not found"})}

    except Exception as e:
        logger.error(f"Orchestration error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error", "details": str(e)}),
        }
