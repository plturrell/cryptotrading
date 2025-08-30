"""
Enhanced A2A Protocol for Cryptotrading System

Extends the base A2A protocol with specialized message types for
data ingestion, ML workflows, and distributed coordination.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .a2a_protocol import A2AMessage, MessageType


class EnhancedMessageType(Enum):
    """Extended message types for specialized workflows"""

    # Data Ingestion Coordination
    INGESTION_WORKFLOW_START = "ingestion_workflow_start"
    INGESTION_TASK_ASSIGN = "ingestion_task_assign"
    INGESTION_PROGRESS_UPDATE = "ingestion_progress_update"
    INGESTION_VALIDATION_REQUEST = "ingestion_validation_request"
    INGESTION_QUALITY_REPORT = "ingestion_quality_report"

    # Data Source Management
    SOURCE_AVAILABILITY_CHECK = "source_availability_check"
    SOURCE_RATE_LIMIT_STATUS = "source_rate_limit_status"
    SOURCE_FAILOVER_REQUEST = "source_failover_request"

    # ML Pipeline Coordination
    ML_TRAINING_JOB_REQUEST = "ml_training_job_request"
    ML_TRAINING_RESOURCE_ALLOCATION = "ml_training_resource_allocation"
    ML_TRAINING_CHECKPOINT_SYNC = "ml_training_checkpoint_sync"
    ML_TRAINING_METRICS_UPDATE = "ml_training_metrics_update"

    # Model Management
    ML_MODEL_REGISTRY_UPDATE = "ml_model_registry_update"
    ML_MODEL_VERSION_DEPLOY = "ml_model_version_deploy"
    ML_MODEL_PERFORMANCE_MONITOR = "ml_model_performance_monitor"

    # Inference Coordination
    ML_INFERENCE_BATCH_REQUEST = "ml_inference_batch_request"
    ML_INFERENCE_STREAM_START = "ml_inference_stream_start"
    ML_INFERENCE_RESULT_AGGREGATE = "ml_inference_result_aggregate"

    # Streaming Control
    STREAM_SUBSCRIBE = "stream_subscribe"
    STREAM_UNSUBSCRIBE = "stream_unsubscribe"
    STREAM_PAUSE = "stream_pause"
    STREAM_RESUME = "stream_resume"
    STREAM_RATE_ADJUST = "stream_rate_adjust"
    STREAM_HEALTH_CHECK = "stream_health_check"

    # Distributed Computing
    COMPUTE_JOB_SUBMIT = "compute_job_submit"
    COMPUTE_TASK_ASSIGN = "compute_task_assign"
    COMPUTE_TASK_STATUS = "compute_task_status"
    COMPUTE_RESULT_COLLECT = "compute_result_collect"
    COMPUTE_RESOURCE_REQUEST = "compute_resource_request"

    # Transaction Support
    TRANSACTION_BEGIN = "transaction_begin"
    TRANSACTION_PREPARE = "transaction_prepare"
    TRANSACTION_COMMIT = "transaction_commit"
    TRANSACTION_ROLLBACK = "transaction_rollback"


@dataclass
class DataIngestionRequest:
    """Request for coordinated data ingestion"""

    sources: List[str]
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    interval: str = "1h"
    parallel_workers: int = 4
    quality_threshold: float = 0.95
    validation_rules: List[str] = None

    def to_message(self, sender_id: str, receiver_id: str) -> A2AMessage:
        return A2AMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=EnhancedMessageType.INGESTION_WORKFLOW_START,
            payload={
                "sources": self.sources,
                "symbols": self.symbols,
                "date_range": {
                    "start": self.start_date.isoformat(),
                    "end": self.end_date.isoformat(),
                },
                "interval": self.interval,
                "parallel_workers": self.parallel_workers,
                "quality_threshold": self.quality_threshold,
                "validation_rules": self.validation_rules or ["completeness", "price_validity"],
            },
        )


@dataclass
class MLTrainingRequest:
    """Request for distributed ML training"""

    model_type: str
    model_config: Dict[str, Any]
    dataset_id: str
    num_workers: int = 1
    epochs: int = 100
    batch_size: int = 32

    def to_message(self, sender_id: str, receiver_id: str) -> A2AMessage:
        return A2AMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=EnhancedMessageType.ML_TRAINING_JOB_REQUEST,
            payload={
                "model_type": self.model_type,
                "model_config": self.model_config,
                "training_data": {"dataset_id": self.dataset_id},
                "distributed_config": {
                    "num_workers": self.num_workers,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                },
            },
        )


@dataclass
class StreamSubscription:
    """Subscription request for data streaming"""

    stream_id: str
    subscriber_id: str
    filters: Dict[str, Any]
    batch_size: int = 100
    max_rate_per_second: int = 1000
    compression: str = "gzip"
    format: str = "msgpack"

    def to_message(self, sender_id: str, receiver_id: str) -> A2AMessage:
        return A2AMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=EnhancedMessageType.STREAM_SUBSCRIBE,
            payload={
                "stream_id": self.stream_id,
                "subscriber_id": self.subscriber_id,
                "filters": self.filters,
                "batch_size": self.batch_size,
                "max_rate_per_second": self.max_rate_per_second,
                "compression": self.compression,
                "format": self.format,
            },
        )


@dataclass
class DataQualityReport:
    """Data quality validation report"""

    dataset_id: str
    timestamp: datetime
    rules_passed: List[str]
    rules_failed: List[Tuple[str, str]]  # (rule_id, reason)
    quality_score: float
    recommendations: List[str]
    sample_size: int
    total_records: int

    def to_message(self, sender_id: str, receiver_id: str) -> A2AMessage:
        return A2AMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=EnhancedMessageType.INGESTION_QUALITY_REPORT,
            payload={
                "dataset_id": self.dataset_id,
                "timestamp": self.timestamp.isoformat(),
                "rules_passed": self.rules_passed,
                "rules_failed": self.rules_failed,
                "quality_score": self.quality_score,
                "recommendations": self.recommendations,
                "statistics": {
                    "sample_size": self.sample_size,
                    "total_records": self.total_records,
                },
            },
        )


@dataclass
class ComputeTask:
    """Distributed compute task specification"""

    task_id: str
    task_type: str  # "map", "reduce", "aggregate", "transform"
    input_data: Dict[str, Any]
    dependencies: List[str]
    priority: int = 1
    deadline: Optional[datetime] = None
    resource_requirements: Dict[str, Any] = None

    def to_message(self, sender_id: str, receiver_id: str) -> A2AMessage:
        return A2AMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=EnhancedMessageType.COMPUTE_TASK_ASSIGN,
            payload={
                "task_id": self.task_id,
                "task_type": self.task_type,
                "input_data": self.input_data,
                "dependencies": self.dependencies,
                "priority": self.priority,
                "deadline": self.deadline.isoformat() if self.deadline else None,
                "resource_requirements": self.resource_requirements or {},
            },
        )


class EnhancedA2AProtocol:
    """Enhanced A2A protocol with specialized workflows"""

    @staticmethod
    async def coordinate_data_ingestion(
        orchestrator_id: str,
        sources: List[str],
        symbols: List[str],
        date_range: Tuple[datetime, datetime],
        workers: List[str],
    ) -> str:
        """
        Coordinate distributed data ingestion workflow

        Returns:
            workflow_id for tracking progress
        """
        import uuid

        workflow_id = str(uuid.uuid4())

        # Create ingestion request
        request = DataIngestionRequest(
            sources=sources,
            symbols=symbols,
            start_date=date_range[0],
            end_date=date_range[1],
            parallel_workers=len(workers),
        )

        # Send to first available worker
        message = request.to_message(orchestrator_id, workers[0])

        # In real implementation, this would send via transport
        # await transport.send(message)

        return workflow_id

    @staticmethod
    def create_transaction_flow(transaction_id: str, participants: List[str]) -> List[A2AMessage]:
        """
        Create a distributed transaction flow

        Returns:
            List of messages for 2-phase commit
        """
        messages = []

        # Phase 1: Begin transaction
        for participant in participants:
            messages.append(
                A2AMessage(
                    sender_id="coordinator",
                    receiver_id=participant,
                    message_type=EnhancedMessageType.TRANSACTION_BEGIN,
                    payload={"transaction_id": transaction_id},
                )
            )

        # Phase 2: Prepare
        for participant in participants:
            messages.append(
                A2AMessage(
                    sender_id="coordinator",
                    receiver_id=participant,
                    message_type=EnhancedMessageType.TRANSACTION_PREPARE,
                    payload={"transaction_id": transaction_id},
                )
            )

        return messages
