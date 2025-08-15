"""
Workflow Registry - Manages and stores workflow definitions
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from .workflow_engine import WorkflowDefinition, WorkflowStep, create_eth_loading_workflow

logger = logging.getLogger(__name__)

class WorkflowRegistry:
    """Registry for workflow definitions with persistence"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.storage_path = storage_path or Path("/Users/apple/projects/cryptotrading/workflows")
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # Load built-in workflows
        self._register_builtin_workflows()
        
        # Load custom workflows from storage
        self._load_workflows_from_storage()
    
    def _register_builtin_workflows(self):
        """Register built-in workflow templates"""
        # ETH Loading Workflow - Only workflow in production system
        self.register(create_eth_loading_workflow())
    
    def register(self, workflow: WorkflowDefinition):
        """Register a workflow definition"""
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.name} ({workflow.workflow_id})")
        
        # Persist to storage
        self._save_workflow(workflow)
    
    def get(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get a workflow definition by ID"""
        return self.workflows.get(workflow_id)
    
    def list(self) -> List[Dict[str, str]]:
        """List all registered workflows"""
        return [
            {
                "id": wf.workflow_id,
                "name": wf.name,
                "description": wf.description,
                "steps": len(wf.steps)
            }
            for wf in self.workflows.values()
        ]
    
    def create_custom_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, any]]
    ) -> WorkflowDefinition:
        """Create a custom workflow from configuration"""
        workflow_id = f"custom-{datetime.now().timestamp()}"
        
        workflow_steps = []
        for step_config in steps:
            step = WorkflowStep(
                step_id=step_config["id"],
                agent_id=step_config["agent"],
                action=step_config["action"],
                input_data=step_config.get("input", {}),
                depends_on=step_config.get("depends_on", []),
                timeout=step_config.get("timeout", 300),
                retry_count=step_config.get("retry", 3)
            )
            workflow_steps.append(step)
        
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description,
            steps=workflow_steps
        )
        
        self.register(workflow)
        return workflow
    
    def _save_workflow(self, workflow: WorkflowDefinition):
        """Save workflow to storage"""
        try:
            filepath = self.storage_path / f"{workflow.workflow_id}.json"
            with open(filepath, 'w') as f:
                json.dump(workflow.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save workflow {workflow.workflow_id}: {e}")
    
    def _load_workflows_from_storage(self):
        """Load workflows from storage"""
        try:
            for filepath in self.storage_path.glob("*.json"):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # Reconstruct workflow from data
                    # This would need proper deserialization
                    logger.info(f"Loaded workflow from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load workflows: {e}")

# Global registry instance
workflow_registry = WorkflowRegistry()