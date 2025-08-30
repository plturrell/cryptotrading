"""
Core agents module - Full Strands Ecosystem
"""
from .base import BaseAgent
from .memory import MemoryAgent

# Import the old StrandsAgent conditionally to avoid circular dependencies
try:
    from .strands import StrandsAgent

    strands_available = True
except ImportError as e:
    # If there are dependency issues, provide a placeholder
    StrandsAgent = None
    strands_available = False

# Enhanced Strands Framework Components
try:
    from .strands_communication import StrandsA2ACommunication
    from .strands_enhanced import EnhancedStrandsAgent
    from .strands_observability import StrandsObservabilitySystem
    from .strands_tools import StrandsToolsAgent
    from .strands_workflows import WorkflowEngine

    strands_enhanced_available = True
except ImportError as e:
    strands_enhanced_available = False
    import logging

    logging.warning(f"Enhanced Strands components not available: {e}")

# Try to import specialized agents
try:
    from .specialized.agent_manager import AgentManagerAgent
    from .specialized.mcts_calculation_agent import (
        MCTSCalculationAgent,
        ProductionMCTSCalculationAgent,
    )
    from .specialized.strands_glean_agent import StrandsGleanAgent

    specialized_available = True
except ImportError as e:
    specialized_available = False
    import logging

    logging.warning(f"Specialized agents not available: {e}")

__all__ = ["BaseAgent", "MemoryAgent"]

if strands_available:
    __all__.append("StrandsAgent")

if strands_enhanced_available:
    __all__.extend(
        [
            "EnhancedStrandsAgent",
            "StrandsToolsAgent",
            "WorkflowEngine",
            "StrandsA2ACommunication",
            "StrandsObservabilitySystem",
        ]
    )

if specialized_available:
    __all__.extend(["ProductionMCTSCalculationAgent", "AgentManagerAgent", "StrandsGleanAgent"])
