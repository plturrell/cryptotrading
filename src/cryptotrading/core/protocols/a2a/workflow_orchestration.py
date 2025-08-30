"""
Workflow Orchestration with On-Chain Data Exchange
Coordinates cross-agent workflows using blockchain storage
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from .blockchain_data_exchange import (
    BlockchainDataExchangeService,
    WorkflowStatus,
    get_data_exchange_service
)

logger = logging.getLogger(__name__)


class WorkflowStepType(Enum):
    """Types of workflow steps"""
    DATA_COLLECTION = "data_collection"
    DATA_ANALYSIS = "data_analysis"
    ML_PREDICTION = "ml_prediction"
    TECHNICAL_ANALYSIS = "technical_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_EXECUTION = "trade_execution"
    PORTFOLIO_UPDATE = "portfolio_update"
    REPORT_GENERATION = "report_generation"


@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    step_id: str
    step_type: WorkflowStepType
    agent_id: str
    input_data_ids: List[int] = field(default_factory=list)
    output_data_id: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "agent_id": self.agent_id,
            "input_data_ids": self.input_data_ids,
            "output_data_id": self.output_data_id,
            "parameters": self.parameters,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class WorkflowTemplate:
    """Template for reusable workflows"""
    template_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    required_agents: Set[str]
    expected_duration_seconds: int = 300
    
    def instantiate(self, parameters: Dict[str, Any] = None) -> 'WorkflowInstance':
        """Create a workflow instance from template"""
        instance_id = f"{self.template_id}-{uuid4().hex[:8]}"
        steps_copy = [
            WorkflowStep(
                step_id=f"{instance_id}-{step.step_id}",
                step_type=step.step_type,
                agent_id=step.agent_id,
                parameters={**step.parameters, **(parameters or {})}
            )
            for step in self.steps
        ]
        
        return WorkflowInstance(
            workflow_id=instance_id,
            template_id=self.template_id,
            steps=steps_copy,
            participants=list(self.required_agents)
        )


@dataclass
class WorkflowInstance:
    """Active workflow instance"""
    workflow_id: str
    template_id: Optional[str]
    steps: List[WorkflowStep]
    participants: List[str]
    blockchain_workflow_id: Optional[int] = None
    status: WorkflowStatus = WorkflowStatus.INITIATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    final_result: Optional[Any] = None
    
    def get_next_step(self) -> Optional[WorkflowStep]:
        """Get the next pending step"""
        for step in self.steps:
            if step.status == "pending":
                return step
        return None
    
    def is_complete(self) -> bool:
        """Check if all steps are complete"""
        return all(step.status in ["completed", "skipped"] for step in self.steps)
    
    def has_failed(self) -> bool:
        """Check if any step has failed"""
        return any(step.status == "failed" for step in self.steps)


class WorkflowOrchestrator:
    """Orchestrates cross-agent workflows with on-chain data exchange"""
    
    def __init__(self):
        """Initialize workflow orchestrator"""
        self.data_exchange: Optional[BlockchainDataExchangeService] = None
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.active_workflows: Dict[str, WorkflowInstance] = {}
        self.step_handlers: Dict[WorkflowStepType, Callable] = {}
        
        # Initialize built-in templates
        self._init_templates()
        
        logger.info("WorkflowOrchestrator initialized")
    
    async def initialize(self) -> bool:
        """Initialize connection to blockchain data exchange"""
        try:
            self.data_exchange = await get_data_exchange_service()
            logger.info("Connected to blockchain data exchange")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize data exchange: {e}")
            return False
    
    def _init_templates(self):
        """Initialize built-in workflow templates"""
        
        # Market Analysis Workflow
        self.templates["market_analysis"] = WorkflowTemplate(
            template_id="market_analysis_v1",
            name="Market Analysis Workflow",
            description="Comprehensive market analysis using multiple agents",
            steps=[
                WorkflowStep(
                    step_id="collect_data",
                    step_type=WorkflowStepType.DATA_COLLECTION,
                    agent_id="data-collection-agent",
                    parameters={"sources": ["binance", "coinbase", "kraken"]}
                ),
                WorkflowStep(
                    step_id="technical_analysis",
                    step_type=WorkflowStepType.TECHNICAL_ANALYSIS,
                    agent_id="technical-analysis-agent",
                    parameters={"indicators": ["RSI", "MACD", "BB"]}
                ),
                WorkflowStep(
                    step_id="ml_prediction",
                    step_type=WorkflowStepType.ML_PREDICTION,
                    agent_id="ml-agent",
                    parameters={"model": "ensemble", "horizon": "1h"}
                ),
                WorkflowStep(
                    step_id="generate_report",
                    step_type=WorkflowStepType.REPORT_GENERATION,
                    agent_id="report-agent",
                    parameters={"format": "json"}
                )
            ],
            required_agents={
                "data-collection-agent",
                "technical-analysis-agent",
                "ml-agent",
                "report-agent"
            },
            expected_duration_seconds=120
        )
        
        # Trading Signal Workflow
        self.templates["trading_signal"] = WorkflowTemplate(
            template_id="trading_signal_v1",
            name="Trading Signal Generation",
            description="Generate trading signals from multiple data sources",
            steps=[
                WorkflowStep(
                    step_id="analyze_market",
                    step_type=WorkflowStepType.DATA_ANALYSIS,
                    agent_id="data-analysis-agent",
                    parameters={"depth": "deep", "timeframes": ["1m", "5m", "1h"]}
                ),
                WorkflowStep(
                    step_id="assess_risk",
                    step_type=WorkflowStepType.RISK_ASSESSMENT,
                    agent_id="risk-agent",
                    parameters={"risk_level": "medium"}
                ),
                WorkflowStep(
                    step_id="generate_signal",
                    step_type=WorkflowStepType.ML_PREDICTION,
                    agent_id="signal-agent",
                    parameters={"confidence_threshold": 0.7}
                )
            ],
            required_agents={
                "data-analysis-agent",
                "risk-agent",
                "signal-agent"
            },
            expected_duration_seconds=60
        )
        
        # Portfolio Optimization Workflow
        self.templates["portfolio_optimization"] = WorkflowTemplate(
            template_id="portfolio_opt_v1",
            name="Portfolio Optimization",
            description="Optimize portfolio allocation using ML and risk analysis",
            steps=[
                WorkflowStep(
                    step_id="collect_portfolio",
                    step_type=WorkflowStepType.DATA_COLLECTION,
                    agent_id="portfolio-agent",
                    parameters={"include_history": True}
                ),
                WorkflowStep(
                    step_id="analyze_risk",
                    step_type=WorkflowStepType.RISK_ASSESSMENT,
                    agent_id="risk-agent",
                    parameters={"metrics": ["VaR", "CVaR", "Sharpe"]}
                ),
                WorkflowStep(
                    step_id="optimize_allocation",
                    step_type=WorkflowStepType.ML_PREDICTION,
                    agent_id="optimization-agent",
                    parameters={"method": "mean_variance", "constraints": "long_only"}
                ),
                WorkflowStep(
                    step_id="update_portfolio",
                    step_type=WorkflowStepType.PORTFOLIO_UPDATE,
                    agent_id="portfolio-agent",
                    parameters={"execute": False}
                )
            ],
            required_agents={
                "portfolio-agent",
                "risk-agent",
                "optimization-agent"
            },
            expected_duration_seconds=180
        )
    
    async def create_workflow(
        self,
        template_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a new workflow instance from template
        
        Args:
            template_id: ID of workflow template
            parameters: Optional parameters to override template defaults
        
        Returns:
            Workflow instance ID if successful
        """
        try:
            template = self.templates.get(template_id)
            if not template:
                logger.error(f"Template {template_id} not found")
                return None
            
            # Create instance
            instance = template.instantiate(parameters)
            
            # Register on blockchain
            if self.data_exchange:
                blockchain_id = await self.data_exchange.create_workflow(
                    workflow_type=template.name,
                    participants=instance.participants
                )
                
                if blockchain_id:
                    instance.blockchain_workflow_id = blockchain_id
                    logger.info(f"Workflow registered on blockchain: {blockchain_id}")
            
            # Store instance
            self.active_workflows[instance.workflow_id] = instance
            
            logger.info(f"Created workflow instance: {instance.workflow_id}")
            return instance.workflow_id
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            return None
    
    async def execute_workflow(
        self,
        workflow_id: str,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a workflow instance
        
        Args:
            workflow_id: ID of workflow instance
            initial_data: Initial data to pass to first step
        
        Returns:
            Final workflow result if successful
        """
        try:
            instance = self.active_workflows.get(workflow_id)
            if not instance:
                logger.error(f"Workflow {workflow_id} not found")
                return None
            
            instance.status = WorkflowStatus.IN_PROGRESS
            instance.started_at = datetime.now()
            
            # Store initial data on-chain if provided
            initial_data_id = None
            if initial_data and self.data_exchange:
                initial_data_id = await self.data_exchange.store_data(
                    sender_agent_id="orchestrator",
                    receiver_agent_id=instance.steps[0].agent_id,
                    data=initial_data,
                    data_type="workflow_input"
                )
            
            # Execute steps sequentially
            previous_output_id = initial_data_id
            
            for step in instance.steps:
                try:
                    # Update step status
                    step.status = "running"
                    step.started_at = datetime.now()
                    
                    # Prepare input data IDs
                    if previous_output_id:
                        step.input_data_ids = [previous_output_id]
                    
                    # Execute step
                    output_data_id = await self._execute_step(instance, step)
                    
                    if output_data_id:
                        step.output_data_id = output_data_id
                        step.status = "completed"
                        previous_output_id = output_data_id
                        
                        # Add data to blockchain workflow
                        if instance.blockchain_workflow_id and self.data_exchange:
                            await self.data_exchange.add_data_to_workflow(
                                workflow_id=instance.blockchain_workflow_id,
                                data_id=output_data_id,
                                agent_id=step.agent_id
                            )
                    else:
                        step.status = "failed"
                        step.error = "No output data produced"
                        logger.error(f"Step {step.step_id} failed")
                        break
                    
                    step.completed_at = datetime.now()
                    
                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    logger.error(f"Step {step.step_id} failed: {e}")
                    break
            
            # Check workflow completion
            if instance.is_complete():
                instance.status = WorkflowStatus.COMPLETED
                
                # Get final result
                last_step = instance.steps[-1]
                if last_step.output_data_id and self.data_exchange:
                    result_data = await self.data_exchange.retrieve_data(
                        data_id=last_step.output_data_id,
                        agent_id="orchestrator"
                    )
                    instance.final_result = result_data
                    
                    # Complete blockchain workflow
                    if instance.blockchain_workflow_id:
                        await self.data_exchange.complete_workflow(
                            workflow_id=instance.blockchain_workflow_id,
                            result=result_data,
                            agent_id="orchestrator"
                        )
            else:
                instance.status = WorkflowStatus.FAILED
            
            instance.completed_at = datetime.now()
            
            return instance.final_result
            
        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id].status = WorkflowStatus.FAILED
            return None
    
    async def _execute_step(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep
    ) -> Optional[int]:
        """Execute a single workflow step"""
        try:
            # Get agent from registry (simplified for testing)
            # In production, would fetch from A2AAgentRegistry
            agent = None  # Placeholder
            if not agent:
                logger.error(f"Agent {step.agent_id} not found")
                return None
            
            # Prepare input data
            input_data = {}
            if step.input_data_ids and self.data_exchange:
                for data_id in step.input_data_ids:
                    data = await self.data_exchange.retrieve_data(
                        data_id=data_id,
                        agent_id=step.agent_id
                    )
                    if data:
                        input_data[f"input_{data_id}"] = data
            
            # Add step parameters
            input_data.update(step.parameters)
            
            # Execute agent task via MCP
            if hasattr(agent, 'process_mcp_request'):
                result = await agent.process_mcp_request(
                    tool=step.step_type.value,
                    arguments=input_data
                )
                
                # Store result on-chain
                if result and self.data_exchange:
                    output_data_id = await self.data_exchange.store_data(
                        sender_agent_id=step.agent_id,
                        receiver_agent_id="orchestrator",
                        data=result,
                        data_type=f"step_output_{step.step_type.value}"
                    )
                    
                    return output_data_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to execute step {step.step_id}: {e}")
            return None
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        instance = self.active_workflows.get(workflow_id)
        if not instance:
            return None
        
        return {
            "workflow_id": instance.workflow_id,
            "template_id": instance.template_id,
            "status": instance.status.name,
            "created_at": instance.created_at.isoformat(),
            "started_at": instance.started_at.isoformat() if instance.started_at else None,
            "completed_at": instance.completed_at.isoformat() if instance.completed_at else None,
            "steps": [step.to_dict() for step in instance.steps],
            "final_result": instance.final_result
        }
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List available workflow templates"""
        return [
            {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "required_agents": list(template.required_agents),
                "num_steps": len(template.steps),
                "expected_duration": template.expected_duration_seconds
            }
            for template in self.templates.values()
        ]
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List active workflow instances"""
        return [
            {
                "workflow_id": instance.workflow_id,
                "template_id": instance.template_id,
                "status": instance.status.name,
                "created_at": instance.created_at.isoformat(),
                "progress": f"{sum(1 for s in instance.steps if s.status == 'completed')}/{len(instance.steps)}"
            }
            for instance in self.active_workflows.values()
        ]


# Singleton instance
_orchestrator: Optional[WorkflowOrchestrator] = None


async def get_orchestrator() -> WorkflowOrchestrator:
    """Get or create the workflow orchestrator singleton"""
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = WorkflowOrchestrator()
        await _orchestrator.initialize()
    
    return _orchestrator