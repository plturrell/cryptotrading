"""
A2A Agent Manager - Strand Agent
Enforces (1) Registration (2) MCP Segregation (3) A2A Skill Card Compliance
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ...protocols.a2a.a2a_protocol import (
    A2A_CAPABILITIES,
    A2AAgentRegistry,
    A2AMessage,
    AgentStatus,
    MessageType,
)
from ...protocols.cds import A2AAgentCDSMixin, CDSServiceConfig
try:
    from ...infrastructure.transactions.cds_transactional_client import CDSTransactionalMixin
    from ...infrastructure.transactions.agent_transaction_manager import transactional, TransactionIsolation
    TRANSACTIONS_AVAILABLE = True
except ImportError:
    # Fallback classes for when transactions are not available
    class CDSTransactionalMixin:
        pass
    
    def transactional(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class TransactionIsolation:
        READ_COMMITTED = "read_committed"
    
    TRANSACTIONS_AVAILABLE = False
from ..strands import StrandsAgent

try:
    from ....data.database.client import get_db
    from ....data.database.models import A2AAgent
    from ....infrastructure.analysis.mcp_agent_segregation import (
        AgentContext,
        AgentRole,
        ResourceType,
        get_segregation_manager,
    )

    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    # Mock classes for testing
    INFRASTRUCTURE_AVAILABLE = False

    def get_segregation_manager():
        return None

    def get_db():
        return None

    class AgentContext:
        def __init__(self, **kwargs):
            pass

    class AgentRole:
        ANALYST = "analyst"

    class ResourceType:
        MCP_TOOL = "mcp_tool"

    class A2AAgent:
        def __init__(self, **kwargs):
            pass


logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """A2A Skill Card Compliance Status"""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    SUSPENDED = "suspended"


@dataclass
class SkillCard:
    """A2A Skill Card Definition"""

    skill_id: str
    skill_name: str
    description: str
    required_capabilities: List[str]
    mcp_tools: List[str]
    compliance_rules: Dict[str, Any]
    version: str = "1.0"


@dataclass
class AgentRegistrationRequest:
    """Agent registration request"""

    agent_id: str
    agent_type: str
    capabilities: List[str]
    mcp_tools: List[str] = field(default_factory=list)
    skill_cards: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Agent compliance assessment report"""

    agent_id: str
    status: ComplianceStatus
    skill_card_compliance: Dict[str, bool]
    mcp_segregation_status: bool
    registration_status: bool
    violations: List[str]
    recommendations: List[str]
    last_checked: datetime = field(default_factory=datetime.utcnow)


if TRANSACTIONS_AVAILABLE:
    class AgentManagerAgent(StrandsAgent, A2AAgentCDSMixin, CDSTransactionalMixin):
        pass
else:
    class AgentManagerAgent(StrandsAgent, A2AAgentCDSMixin):
        pass

class AgentManagerAgent(AgentManagerAgent):
    """
    A2A Agent Manager - Strand Agent
    ALL functionality exposed through MCP tools ONLY.
    Enforces registration, MCP segregation, and A2A skill card compliance.
    """

    def __init__(self, agent_id: str = "agent-manager-001", **kwargs):
        super().__init__(
            agent_id=agent_id,
            agent_type="agent_manager",
            capabilities=[
                "agent_registration",
                "mcp_segregation_enforcement",
                "skill_card_compliance",
                "agent_discovery",
                "compliance_monitoring",
            ],
            **kwargs,
        )
        
        # Initialize CDS integration
        self.cds_config = CDSServiceConfig(base_url="http://localhost:4005")
        self.cds_initialized = False
        self.transactional_cds_initialized = False
        self.transactions_available = TRANSACTIONS_AVAILABLE

        self.db = get_db()
        self.segregation_manager = get_segregation_manager()
        self.registered_agents: Dict[str, AgentRegistrationRequest] = {}
        self.compliance_reports: Dict[str, ComplianceReport] = {}
        self.skill_cards: Dict[str, SkillCard] = {}
        self.mcp_segregation_status = {}
        self.system_health = {"status": "initializing", "last_check": datetime.utcnow()}

        # Initialize memory system for agent management
        self._initialize_memory_system()

        # Register with A2A protocol
        capabilities = A2A_CAPABILITIES.get(agent_id, [])
        A2AAgentRegistry.register_agent(agent_id, capabilities, self)

        # Initialize MCP tools - ALL functionality through these
        self.mcp_tools = self._load_mcp_tools()
        self.mcp_handlers = self._initialize_mcp_handlers()

        # Blockchain integration
        self.blockchain_registry = None  # Will be initialized when needed

    async def initialize(self) -> bool:
        """Initialize the Agent Manager with CDS integration"""
        try:
            logger.info(f"Initializing Agent Manager {self.agent_id}")

            # Initialize CDS integration first
            if not await self.initialize_cds(self.agent_id):
                logger.warning("CDS integration failed, falling back to direct DB")
            else:
                self.cds_initialized = True
                logger.info("CDS integration successful")
                
                # Initialize transactional CDS client if available
                if TRANSACTIONS_AVAILABLE and hasattr(self, 'initialize_transactional_cds'):
                    if await self.initialize_transactional_cds(self.agent_id):
                        self.transactional_cds_initialized = True
                        logger.info("Transactional CDS client initialized")
                    else:
                        logger.warning("Transactional CDS client initialization failed")
                else:
                    logger.info("Transactional CDS not available - using standard CDS integration")

            # Verify database connection (fallback)
            if not self.db and not self.cds_initialized:
                logger.error("Neither CDS nor database connection available")
                return False

            # Initialize segregation manager
            if not self.segregation_manager:
                logger.warning("MCP segregation manager not available")

            # Register with CDS if available
            if self.cds_initialized:
                try:
                    await self.register_with_cds([
                        "agent_registration",
                        "mcp_segregation_enforcement", 
                        "skill_card_compliance",
                        "agent_discovery",
                        "compliance_monitoring"
                    ])
                except Exception as e:
                    logger.warning(f"CDS registration failed: {e}")

            # Update system health
            self.system_health = {
                "status": "running",
                "last_check": datetime.utcnow(),
                "db_connected": bool(self.db),
                "cds_connected": self.cds_initialized,
                "segregation_enabled": bool(self.segregation_manager),
            }

            logger.info(f"Agent Manager {self.agent_id} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Agent Manager {self.agent_id}: {e}")
            self.system_health = {
                "status": "error",
                "last_check": datetime.utcnow(),
                "error": str(e),
            }
            return False

    async def start(self) -> bool:
        """Start the Agent Manager"""
        try:
            logger.info(f"Starting Agent Manager {self.agent_id}")

            # Start compliance monitoring
            await self._start_compliance_monitoring()

            # Update system health
            self.system_health["status"] = "active"
            self.system_health["started_at"] = datetime.utcnow()

            logger.info(f"Agent Manager {self.agent_id} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start Agent Manager {self.agent_id}: {e}")
            self.system_health["status"] = "error"
            self.system_health["error"] = str(e)
            return False

    async def _start_compliance_monitoring(self):
        """Start background compliance monitoring"""
        try:
            # This could be expanded to include periodic compliance checks
            logger.info("Compliance monitoring started")
        except Exception as e:
            logger.warning(f"Compliance monitoring startup failed: {e}")

    async def _initialize_memory_system(self):
        """Initialize memory system for agent management and compliance tracking"""
        try:
            # Store agent manager configuration
            await self.store_memory(
                "agent_manager_config",
                {
                    "agent_id": self.agent_id,
                    "compliance_enabled": True,
                    "mcp_segregation_enabled": True,
                    "initialized_at": datetime.utcnow().isoformat(),
                },
                {"type": "configuration", "persistent": True},
            )

            # Initialize agent registry cache
            await self.store_memory(
                "agent_registry_cache", {}, {"type": "registry", "persistent": True}
            )

            # Initialize compliance history
            await self.store_memory(
                "compliance_history", [], {"type": "compliance_log", "persistent": True}
            )

            # Initialize system health tracking
            await self.store_memory(
                "system_health_history", [], {"type": "health_monitoring", "persistent": True}
            )

            logger.info(f"Memory system initialized for Agent Manager {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to initialize agent manager memory system: {e}")

    def _initialize_skill_cards(self):
        """Initialize core A2A skill cards"""
        self.skill_cards = {
            "historical_data_loading": SkillCard(
                skill_id="historical_data_loading",
                skill_name="Historical Data Loading",
                description="Load and manage historical market data",
                required_capabilities=["data_loading", "historical_data", "technical_indicators"],
                mcp_tools=[
                    "get_market_data",
                    "get_historical_prices",
                    "calculate_technical_indicators",
                ],
                compliance_rules={
                    "max_data_retention_days": 365,
                    "required_data_sources": ["yahoo_finance", "fred"],
                    "mandatory_indicators": ["sma", "ema", "rsi"],
                },
            ),
            "portfolio_management": SkillCard(
                skill_id="portfolio_management",
                skill_name="Portfolio Management",
                description="Manage crypto portfolios and wallets",
                required_capabilities=["portfolio_management", "wallet_operations"],
                mcp_tools=["get_portfolio", "get_wallet_balance", "execute_trade"],
                compliance_rules={
                    "max_position_size": 0.1,
                    "required_risk_checks": ["position_limit", "exposure_limit"],
                    "mandatory_reporting": ["daily_pnl", "risk_metrics"],
                },
            ),
            "market_analysis": SkillCard(
                skill_id="market_analysis",
                skill_name="Market Analysis",
                description="Analyze market conditions and sentiment",
                required_capabilities=["market_analysis", "sentiment_analysis"],
                mcp_tools=["analyze_market_sentiment", "get_price_alerts"],
                compliance_rules={
                    "analysis_frequency": "hourly",
                    "required_sources": ["news", "social", "technical"],
                    "confidence_threshold": 0.7,
                },
            ),
            "code_analysis": SkillCard(
                skill_id="code_analysis",
                skill_name="Code Analysis",
                description="Analyze code using CLRS+Tree algorithms",
                required_capabilities=["code_analysis", "dependency_analysis"],
                mcp_tools=["analyze_clrs_tree", "generate_dependency_graph"],
                compliance_rules={
                    "max_analysis_depth": 10,
                    "required_metrics": ["complexity", "maintainability"],
                    "security_scan_required": True,
                },
            ),
            "monte_carlo_calculation": SkillCard(
                skill_id="monte_carlo_calculation",
                skill_name="Monte Carlo Calculations",
                description="Perform MCTS calculations and optimization",
                required_capabilities=["calculation", "optimization", "monte_carlo_search"],
                mcp_tools=["mcts_calculate", "evaluate_strategy", "optimize_parameters"],
                compliance_rules={
                    "min_iterations": 1000,
                    "max_computation_time": 300,
                    "required_validation": "backtesting",
                },
            ),
        }

    def _load_mcp_tools(self) -> Dict[str, Any]:
        """Load MCP tools configuration."""
        import json
        from pathlib import Path

        tools_path = Path(__file__).parent.parent / "mcp_tools" / "agent_manager_tools.json"
        with open(tools_path, "r") as f:
            return json.load(f)

    def _initialize_mcp_handlers(self) -> Dict[str, Any]:
        """Initialize MCP tool handlers - ALL functionality through these."""
        return {
            # Core agent management
            "register_agent": self._mcp_register_agent,
            "validate_compliance": self._mcp_validate_compliance,
            "audit_segregation": self._mcp_audit_segregation,
            "manage_lifecycle": self._mcp_manage_lifecycle,
            "discover_agents": self._mcp_discover_agents,
            "health_check": self._mcp_health_check,
            "blockchain_register": self._mcp_blockchain_register,
            "blockchain_update": self._mcp_blockchain_update,
            "skill_card_validate": self._mcp_skill_card_validate,
            "skill_card_issue": self._mcp_skill_card_issue,
            "enforce_policies": self._mcp_enforce_policies,
            "generate_report": self._mcp_generate_report,
            # Alert & notification management
            "send_alert": self._mcp_send_alert,
            "configure_alerts": self._mcp_configure_alerts,
            # Blockchain operations
            "check_wallet_balance": self._mcp_check_wallet_balance,
            "execute_blockchain_tx": self._mcp_execute_blockchain_tx,
            "manage_gas": self._mcp_manage_gas,
            "track_contracts": self._mcp_track_contracts,
            # DeFi monitoring
            "monitor_defi": self._mcp_monitor_defi,
        }

    async def process_mcp_request(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main entry point for ALL agent manager functionality.
        All requests MUST come through MCP tool invocations.
        """
        logger.info(f"Processing MCP tool request: {tool_name}")

        if tool_name not in self.mcp_handlers:
            raise ValueError(f"Unknown MCP tool: {tool_name}")

        handler = self.mcp_handlers[tool_name]
        result = await handler(**parameters)

        return result

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process agent management requests.
        ALL functionality routed through MCP tools.
        """
        # Check if this is an MCP tool request
        if "mcp_tool" in message:
            return await self.process_mcp_request(
                message["mcp_tool"], message.get("parameters", {})
            )

        # Legacy support - map old message types to MCP tools
        msg_type = message.get("type")
        tool_mapping = {
            "register_agent": "register_agent",
            "check_compliance": "validate_compliance",
            "audit_segregation": "audit_segregation",
            "lifecycle": "manage_lifecycle",
            "discover": "discover_agents",
            "health": "health_check",
        }

        if msg_type in tool_mapping:
            return await self.process_mcp_request(tool_mapping[msg_type], message.get("data", {}))

        # Deprecated direct method calls
        if msg_type == "register_agent":
            return await self.process_mcp_request("register_agent", message)
        elif msg_type == "check_compliance":
            return await self._handle_compliance_check(message)
        elif msg_type == "enforce_segregation":
            return await self._handle_segregation_enforcement(message)
        elif msg_type == "discover_agents":
            return await self._handle_agent_discovery(message)
        elif msg_type == "validate_skill_card":
            return await self._handle_skill_card_validation(message)
        else:
            return {"error": f"Unknown message type: {msg_type}"}

    async def register_agent(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new agent with A2A compliance validation using CDS or fallback to DB"""
        try:
            agent_id = agent_data.get("agent_id")
            agent_name = agent_data.get("agent_name", agent_id)
            agent_type = agent_data.get("agent_type", "generic")
            capabilities = agent_data.get("capabilities", [])
            
            if not agent_id:
                return {"success": False, "error": "Agent ID is required"}

            # Use CDS registration if available
            if self.cds_initialized:
                try:
                    async with self.cds_client.transaction() as tx:
                        # Register via CDS service
                        result = await self.cds_client.call_action('registerAgent', {
                            'agentName': agent_name,
                            'agentType': agent_type,
                            'capabilities': ' '.join(capabilities) if isinstance(capabilities, list) else capabilities
                        })
                        
                        if result.get('status') == 'SUCCESS':
                            # Store in local cache for fast access
                            registration_data = {
                                "agent_id": result.get('agentId', agent_id),
                                "agent_name": agent_name,
                                "agent_type": agent_type,
                                "capabilities": capabilities,
                                "status": "ACTIVE",
                                "registered_at": datetime.utcnow(),
                                "cds_registered": True,
                            }
                            
                            self.registered_agents[agent_id] = registration_data
                            
                            # Update registry cache in memory
                            registry_cache = await self.retrieve_memory("agent_registry_cache") or {}
                            registry_cache[agent_id] = registration_data
                            await self.store_memory("agent_registry_cache", registry_cache, {"type": "registry"})
                            
                            # Log successful registration
                            await self._log_compliance_event(
                                agent_id,
                                "cds_registration_success",
                                {"agent_name": agent_name, "capabilities_count": len(capabilities)},
                            )
                            
                            logger.info(f"Agent {agent_id} registered successfully via CDS")
                            
                            return {
                                "success": True,
                                "agent_id": result.get('agentId', agent_id),
                                "message": result.get('message', 'Registered via CDS'),
                                "method": "CDS",
                                "registered_at": registration_data["registered_at"].isoformat(),
                            }
                        else:
                            logger.warning(f"CDS registration failed: {result}")
                            
                except Exception as cds_error:
                    logger.warning(f"CDS registration failed for {agent_id}: {cds_error}")

            # Fallback to original registration logic
            logger.info(f"Using fallback registration for {agent_id}")
            
            # Check if agent is already registered in memory
            registry_cache = await self.retrieve_memory("agent_registry_cache") or {}
            if agent_id in registry_cache:
                logger.info(f"Agent {agent_id} already registered, updating registration")

            # Validate agent capabilities against A2A standards
            compliance_result = await self._validate_a2a_compliance(agent_id, capabilities)

            if not compliance_result["compliant"]:
                # Store compliance failure in memory
                await self._log_compliance_event(agent_id, "registration_failed", compliance_result)
                return {
                    "success": False,
                    "error": f"A2A compliance validation failed: {compliance_result['issues']}",
                }

            # Register agent
            registration_data = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_type": agent_type,
                "capabilities": capabilities,
                "status": AgentStatus.ACTIVE,
                "registered_at": datetime.utcnow(),
                "compliance_score": compliance_result["score"],
                "last_heartbeat": datetime.utcnow(),
                "cds_registered": False,
            }

            self.registered_agents[agent_id] = registration_data

            # Update registry cache in memory
            registry_cache[agent_id] = registration_data
            await self.store_memory("agent_registry_cache", registry_cache, {"type": "registry"})

            # Generate skill card
            skill_card = await self._generate_skill_card(agent_id, capabilities)
            self.skill_cards[agent_id] = skill_card

            # Log successful registration
            await self._log_compliance_event(
                agent_id,
                "fallback_registration_success",
                {
                    "compliance_score": compliance_result["score"],
                    "capabilities_count": len(capabilities),
                },
            )

            logger.info(f"Agent {agent_id} registered successfully via fallback method")

            return {
                "success": True,
                "agent_id": agent_id,
                "compliance_score": compliance_result["score"],
                "skill_card": skill_card,
                "method": "Fallback",
                "registered_at": registration_data["registered_at"].isoformat(),
            }

        except Exception as e:
            # Store error in memory for learning
            await self.store_memory(
                f"registration_error_{datetime.utcnow().timestamp()}",
                {"error": str(e), "agent_id": agent_id, "timestamp": datetime.utcnow().isoformat()},
                {"type": "error_log"},
            )
            logger.error(f"Agent registration failed: {e}")
            return {"success": False, "error": str(e)}

    if TRANSACTIONS_AVAILABLE:
        @transactional(
            transaction_type="AGENT_REGISTRATION",
            isolation_level=TransactionIsolation.READ_COMMITTED,
            timeout_seconds=60
        )
        async def register_agent_transactional(self, agent_data: Dict[str, Any], transaction=None) -> Dict[str, Any]:
            pass
    
    async def register_agent_transactional(self, agent_data: Dict[str, Any], transaction=None) -> Dict[str, Any]:
        """Register agent with full transaction boundaries and rollback support"""
        try:
            agent_id = agent_data.get("agent_id")
            agent_name = agent_data.get("agent_name", agent_id)
            agent_type = agent_data.get("agent_type", "generic")
            capabilities = agent_data.get("capabilities", [])
            
            if not agent_id:
                return {"success": False, "error": "Agent ID is required"}

            logger.info(f"Starting transactional registration for agent {agent_id}")

            # Use transactional CDS client if available
            if self.transactional_cds_initialized:
                try:
                    # This will create its own transaction context with proper monitoring
                    result = await self.register_agent_with_transaction(capabilities)
                    
                    if result.get('status') == 'SUCCESS':
                        # Also store in local cache within the transaction
                        registration_data = {
                            "agent_id": result.get('agentId', agent_id),
                            "agent_name": agent_name,
                            "agent_type": agent_type,
                            "capabilities": capabilities,
                            "status": "ACTIVE",
                            "registered_at": datetime.utcnow(),
                            "cds_registered": True,
                            "transaction_id": transaction.transaction_id if transaction else None
                        }
                        
                        self.registered_agents[agent_id] = registration_data
                        
                        # Create checkpoint in the transaction
                        if transaction:
                            transaction.create_checkpoint(
                                "agent_cached",
                                {"agent_id": agent_id, "cache_updated": True}
                            )
                        
                        logger.info(f"Agent {agent_id} registered successfully via transactional CDS")
                        
                        return {
                            "success": True,
                            "agent_id": result.get('agentId', agent_id),
                            "message": result.get('message', 'Registered via transactional CDS'),
                            "method": "Transactional_CDS",
                            "transaction_id": transaction.transaction_id if transaction else None,
                            "registered_at": registration_data["registered_at"].isoformat()
                        }
                    else:
                        logger.warning(f"Transactional CDS registration failed: {result}")
                        
                except Exception as tx_error:
                    logger.warning(f"Transactional CDS registration failed: {tx_error}")

            # Fallback to local registration within transaction
            logger.info(f"Using local registration within transaction for {agent_id}")
            
            # Validate agent capabilities
            compliance_result = await self._validate_a2a_compliance(agent_id, capabilities)
            
            if not compliance_result["compliant"]:
                return {
                    "success": False,
                    "error": f"A2A compliance validation failed: {compliance_result['issues']}",
                    "transaction_id": transaction.transaction_id if transaction else None
                }

            # Register agent locally
            registration_data = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_type": agent_type,
                "capabilities": capabilities,
                "status": AgentStatus.ACTIVE,
                "registered_at": datetime.utcnow(),
                "compliance_score": compliance_result["score"],
                "last_heartbeat": datetime.utcnow(),
                "cds_registered": False,
                "transaction_id": transaction.transaction_id if transaction else None
            }

            self.registered_agents[agent_id] = registration_data

            # Generate skill card
            skill_card = await self._generate_skill_card(agent_id, capabilities)
            self.skill_cards[agent_id] = skill_card

            if transaction:
                transaction.create_checkpoint(
                    "local_registration_completed",
                    {
                        "agent_id": agent_id,
                        "method": "Local",
                        "compliance_score": compliance_result["score"]
                    }
                )

            logger.info(f"Agent {agent_id} registered successfully via transactional local method")

            return {
                "success": True,
                "agent_id": agent_id,
                "compliance_score": compliance_result["score"],
                "skill_card": skill_card,
                "method": "Local_Transactional",
                "transaction_id": transaction.transaction_id if transaction else None,
                "registered_at": registration_data["registered_at"].isoformat()
            }

        except Exception as e:
            logger.error(f"Transactional agent registration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": agent_data.get("agent_id"),
                "transaction_id": transaction.transaction_id if transaction else None
            }

    async def cleanup(self) -> bool:
        """Cleanup CDS connections and resources"""
        try:
            # Disconnect from transactional CDS client if connected
            if (TRANSACTIONS_AVAILABLE and hasattr(self, '_transactional_cds_client') 
                and self._transactional_cds_client and hasattr(self, 'cleanup_transactional_cds')):
                await self.cleanup_transactional_cds()
                logger.info("Transactional CDS client disconnected")
            
            # Disconnect from regular CDS if connected
            if hasattr(self, 'cds_client') and self.cds_client:
                await self.cds_client.disconnect()
                logger.info("CDS client disconnected")
            
            # Update system health
            self.system_health["status"] = "shutdown"
            self.system_health["shutdown_at"] = datetime.utcnow()
            
            return True
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

    async def _validate_a2a_compliance(self, agent_id: str, capabilities: List[str]) -> Dict[str, Any]:
        """Validate agent A2A compliance"""
        # Mock compliance validation
        return {
            "compliant": True,
            "score": 85.0,
            "issues": []
        }

    async def _generate_skill_card(self, agent_id: str, capabilities: List[str]) -> Dict[str, Any]:
        """Generate skill card for agent"""
        return {
            "skill_card_id": f"skill_{agent_id}",
            "capabilities": capabilities,
            "issued_at": datetime.utcnow().isoformat(),
            "valid_until": (datetime.utcnow() + timedelta(days=365)).isoformat()
        }

    async def _log_compliance_event(
        self, agent_id: str, event_type: str, event_data: Dict[str, Any]
    ):
        """Log compliance events to memory for tracking and analysis"""
        try:
            compliance_history = await self.retrieve_memory("compliance_history") or []

            event = {
                "agent_id": agent_id,
                "event_type": event_type,
                "event_data": event_data,
                "timestamp": datetime.utcnow().isoformat(),
            }

            compliance_history.append(event)

            # Keep only last 1000 events to prevent memory bloat
            if len(compliance_history) > 1000:
                compliance_history = compliance_history[-1000:]

            await self.store_memory(
                "compliance_history", compliance_history, {"type": "compliance_log"}
            )

        except Exception as e:
            logger.error(f"Failed to log compliance event: {e}")

    async def _handle_agent_registration(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent registration with compliance checks"""
        try:
            request_data = message.get("payload", {})
            request = AgentRegistrationRequest(**request_data)

            # Step 1: Validate registration request
            validation_result = await self._validate_registration_request(request)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": "Registration validation failed",
                    "details": validation_result["errors"],
                }

            # Step 2: Check MCP tool segregation
            segregation_result = await self._check_mcp_segregation(request)
            if not segregation_result["compliant"]:
                return {
                    "success": False,
                    "error": "MCP segregation violation",
                    "details": segregation_result["violations"],
                }

            # Step 3: Validate skill card compliance
            skill_compliance = await self._validate_skill_card_compliance(request)
            if not skill_compliance["compliant"]:
                return {
                    "success": False,
                    "error": "Skill card compliance violation",
                    "details": skill_compliance["violations"],
                }

            # Step 4: Register agent in database
            registration_result = await self._register_agent_in_db(request)
            if not registration_result["success"]:
                return registration_result

            # Step 5: Configure MCP segregation
            await self._configure_agent_segregation(request)

            # Step 6: Store registration
            self.registered_agents[request.agent_id] = request

            # Step 7: Generate compliance report
            compliance_report = ComplianceReport(
                agent_id=request.agent_id,
                status=ComplianceStatus.COMPLIANT,
                skill_card_compliance=skill_compliance["skill_status"],
                mcp_segregation_status=True,
                registration_status=True,
                violations=[],
                recommendations=[],
            )
            self.compliance_reports[request.agent_id] = compliance_report

            logger.info(f"Agent {request.agent_id} registered successfully")

            return {
                "success": True,
                "agent_id": request.agent_id,
                "compliance_status": compliance_report.status.value,
                "assigned_mcp_tools": request.mcp_tools,
                "skill_cards": request.skill_cards,
                "registration_timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return {"success": False, "error": str(e)}

    async def _validate_registration_request(
        self, request: AgentRegistrationRequest
    ) -> Dict[str, Any]:
        """Validate agent registration request"""
        errors = []

        # Check agent ID format
        if not request.agent_id or len(request.agent_id) < 3:
            errors.append("Agent ID must be at least 3 characters")

        # Check for duplicate registration
        if request.agent_id in self.registered_agents:
            errors.append(f"Agent {request.agent_id} already registered")

        # Validate capabilities
        if not request.capabilities:
            errors.append("Agent must declare at least one capability")

        # Check A2A protocol compliance
        if request.agent_id in A2A_CAPABILITIES:
            expected_caps = set(A2A_CAPABILITIES[request.agent_id])
            declared_caps = set(request.capabilities)
            if not expected_caps.issubset(declared_caps):
                missing = expected_caps - declared_caps
                errors.append(f"Missing A2A capabilities: {list(missing)}")

        return {"valid": len(errors) == 0, "errors": errors}

    async def _check_mcp_segregation(self, request: AgentRegistrationRequest) -> Dict[str, Any]:
        """Check MCP tool segregation compliance"""
        violations = []

        # Check if MCP tools are already assigned to other agents
        for tool_name in request.mcp_tools:
            for existing_agent_id, existing_request in self.registered_agents.items():
                if tool_name in existing_request.mcp_tools:
                    violations.append(
                        f"MCP tool '{tool_name}' already assigned to agent '{existing_agent_id}'"
                    )

        # Validate tool ownership boundaries
        agent_type = request.agent_type
        allowed_tools = self._get_allowed_mcp_tools_for_agent_type(agent_type)

        for tool_name in request.mcp_tools:
            if tool_name not in allowed_tools:
                violations.append(
                    f"MCP tool '{tool_name}' not allowed for agent type '{agent_type}'"
                )

        return {"compliant": len(violations) == 0, "violations": violations}

    def _get_allowed_mcp_tools_for_agent_type(self, agent_type: str) -> List[str]:
        """Get allowed MCP tools for agent type"""
        tool_mappings = {
            "historical_loader": [
                "get_market_data",
                "get_historical_prices",
                "calculate_technical_indicators",
            ],
            "database_manager": [
                "get_portfolio",
                "get_wallet_balance",
                "store_transaction_history",
            ],
            "data_management": ["validate_data_schema", "discover_data_sources"],
            "strands_agent": ["analyze_clrs_tree", "generate_dependency_graph", "optimize_code"],
            "mcts_calculation": ["mcts_calculate", "evaluate_strategy", "optimize_parameters"],
        }

        return tool_mappings.get(agent_type, [])

    async def _validate_skill_card_compliance(
        self, request: AgentRegistrationRequest
    ) -> Dict[str, Any]:
        """Validate skill card compliance"""
        violations = []
        skill_status = {}

        for skill_card_id in request.skill_cards:
            if skill_card_id not in self.skill_cards:
                violations.append(f"Unknown skill card: {skill_card_id}")
                skill_status[skill_card_id] = False
                continue

            skill_card = self.skill_cards[skill_card_id]

            # Check required capabilities
            required_caps = set(skill_card.required_capabilities)
            agent_caps = set(request.capabilities)

            if not required_caps.issubset(agent_caps):
                missing = required_caps - agent_caps
                violations.append(
                    f"Skill card '{skill_card_id}' requires capabilities: {list(missing)}"
                )
                skill_status[skill_card_id] = False
            else:
                skill_status[skill_card_id] = True

            # Check required MCP tools
            required_tools = set(skill_card.mcp_tools)
            agent_tools = set(request.mcp_tools)

            if not required_tools.issubset(agent_tools):
                missing_tools = required_tools - agent_tools
                violations.append(
                    f"Skill card '{skill_card_id}' requires MCP tools: {list(missing_tools)}"
                )
                skill_status[skill_card_id] = False

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "skill_status": skill_status,
        }

    async def _register_agent_in_db(self, request: AgentRegistrationRequest) -> Dict[str, Any]:
        """Register agent in database"""
        try:
            with self.db.get_session() as session:
                # Check if agent already exists
                existing = (
                    session.query(A2AAgent).filter(A2AAgent.agent_id == request.agent_id).first()
                )

                if existing:
                    return {"success": False, "error": "Agent already exists in database"}

                # Create new agent record
                agent = A2AAgent(
                    agent_id=request.agent_id,
                    agent_type=request.agent_type,
                    capabilities=json.dumps(request.capabilities),
                    metadata=json.dumps(
                        {
                            "mcp_tools": request.mcp_tools,
                            "skill_cards": request.skill_cards,
                            **request.metadata,
                        }
                    ),
                    status="active",
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                )

                session.add(agent)
                session.commit()

                return {"success": True, "database_id": agent.id}

        except Exception as e:
            logger.error(f"Database registration failed: {e}")
            return {"success": False, "error": str(e)}

    async def _configure_agent_segregation(self, request: AgentRegistrationRequest):
        """Configure MCP segregation for the agent"""
        try:
            # Create agent context for segregation
            agent_context = AgentContext(
                agent_id=request.agent_id,
                tenant_id=f"tenant_{request.agent_id}",
                role=AgentRole.ANALYST,  # Default role
                session_token="",  # Will be generated during authentication
                permissions=set(),
                resource_quotas={},
            )

            # Register agent with segregation manager
            await self.segregation_manager.register_agent_context(agent_context)

            # Assign MCP tools to agent
            for tool_name in request.mcp_tools:
                await self.segregation_manager.assign_tool_to_agent(
                    agent_id=request.agent_id,
                    tool_name=tool_name,
                    resource_type=ResourceType.MCP_TOOL,
                )

            logger.info(f"Configured MCP segregation for agent {request.agent_id}")

        except Exception as e:
            logger.error(f"Segregation configuration failed: {e}")
            raise

    async def _handle_compliance_check(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compliance check request"""
        agent_id = message.get("payload", {}).get("agent_id")

        if not agent_id:
            return {"success": False, "error": "Agent ID required"}

        if agent_id not in self.compliance_reports:
            return {"success": False, "error": "Agent not found"}

        report = self.compliance_reports[agent_id]

        return {"success": True, "compliance_report": report.__dict__}

    async def _handle_segregation_enforcement(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle segregation enforcement request"""
        payload = message.get("payload", {})
        action = payload.get("action")
        agent_id = payload.get("agent_id")

        if action == "audit_segregation":
            return await self._audit_mcp_segregation()
        elif action == "enforce_boundaries":
            return await self._enforce_agent_boundaries(agent_id)
        else:
            return {"success": False, "error": f"Unknown enforcement action: {action}"}

    async def _audit_mcp_segregation(self) -> Dict[str, Any]:
        """Audit MCP tool segregation across all agents"""
        violations = []
        tool_assignments = {}

        # Check for duplicate tool assignments
        for agent_id, request in self.registered_agents.items():
            for tool_name in request.mcp_tools:
                if tool_name in tool_assignments:
                    violations.append(
                        {
                            "type": "duplicate_assignment",
                            "tool": tool_name,
                            "agents": [tool_assignments[tool_name], agent_id],
                        }
                    )
                else:
                    tool_assignments[tool_name] = agent_id

        return {
            "success": True,
            "audit_results": {
                "total_agents": len(self.registered_agents),
                "total_tools": len(tool_assignments),
                "violations": violations,
                "segregation_healthy": len(violations) == 0,
            },
        }

    async def _handle_agent_discovery(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent discovery request"""
        filters = message.get("payload", {}).get("filters", {})

        agents = []
        for agent_id, request in self.registered_agents.items():
            agent_info = {
                "agent_id": agent_id,
                "agent_type": request.agent_type,
                "capabilities": request.capabilities,
                "mcp_tools": request.mcp_tools,
                "skill_cards": request.skill_cards,
                "compliance_status": self.compliance_reports.get(agent_id, {}).status.value
                if agent_id in self.compliance_reports
                else "unknown",
            }

            # Apply filters
            if self._matches_filters(agent_info, filters):
                agents.append(agent_info)

        return {"success": True, "agents": agents, "total_count": len(agents)}

    def _matches_filters(self, agent_info: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if agent matches discovery filters"""
        if "agent_type" in filters and agent_info["agent_type"] != filters["agent_type"]:
            return False

        if "capability" in filters and filters["capability"] not in agent_info["capabilities"]:
            return False

        if "mcp_tool" in filters and filters["mcp_tool"] not in agent_info["mcp_tools"]:
            return False

        return True

    async def _handle_skill_card_validation(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle skill card validation request"""
        skill_card_id = message.get("payload", {}).get("skill_card_id")
        agent_id = message.get("payload", {}).get("agent_id")

        if not skill_card_id or not agent_id:
            return {"success": False, "error": "Skill card ID and agent ID required"}

        if agent_id not in self.registered_agents:
            return {"success": False, "error": "Agent not found"}

        if skill_card_id not in self.skill_cards:
            return {"success": False, "error": "Skill card not found"}

        request = self.registered_agents[agent_id]
        compliance = await self._validate_skill_card_compliance(request)

        return {
            "success": True,
            "validation_result": {
                "skill_card_id": skill_card_id,
                "agent_id": agent_id,
                "compliant": compliance["skill_status"].get(skill_card_id, False),
                "violations": [v for v in compliance["violations"] if skill_card_id in v],
            },
        }

    def _initialize_mcp_tools(self) -> Dict[str, Any]:
        """Initialize MCP tools for Agent Manager."""
        return {
            "register_agent": self._mcp_register_agent,
            "validate_compliance": self._mcp_validate_compliance,
            "enforce_mcp_segregation": self._mcp_enforce_mcp_segregation,
            "generate_skill_card": self._mcp_generate_skill_card,
            "blockchain_register": self._mcp_blockchain_register,
            "manage_lifecycle": self._mcp_manage_lifecycle,
            "query_registry": self._mcp_query_registry,
            "audit_compliance": self._mcp_audit_compliance,
            "monitor_health": self._mcp_monitor_health,
        }

    async def _mcp_register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        mcp_tools: List[str],
        skill_card: Dict[str, Any],
        blockchain_register: bool = True,
    ) -> Dict[str, Any]:
        """MCP tool: Register agent with full A2A compliance."""
        try:
            registration_data = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "capabilities": capabilities,
                "mcp_tools": mcp_tools,
                "skill_card": skill_card,
            }

            result = await self.register_agent(registration_data)

            if result["success"] and blockchain_register:
                blockchain_result = await self._mcp_blockchain_register(
                    agent_id=agent_id, skill_card=skill_card
                )
                result["blockchain_tx"] = blockchain_result.get("transaction_hash")

            return result

        except Exception as e:
            logger.error(f"MCP register_agent failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _mcp_blockchain_register(
        self, agent_id: str, skill_card: Dict[str, Any], registry_contract: Optional[str] = None
    ) -> Dict[str, Any]:
        """MCP tool: Register agent on blockchain."""
        try:
            # In production, this would submit to blockchain
            # For now, simulate the response
            tx_hash = f"0x{agent_id[:8]}...{datetime.utcnow().timestamp():.0f}"

            logger.info(f"Agent {agent_id} registered on blockchain: {tx_hash}")

            return {
                "transaction_hash": tx_hash,
                "block_number": 12345678,
                "registry_address": registry_contract or "0x0000...default",
                "gas_used": 150000,
            }

        except Exception as e:
            logger.error(f"Blockchain registration failed: {e}")
            return {"status": "error", "error": str(e)}

    async def process_mcp_tool_invocation(
        self, message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process MCP tool invocation messages."""
        tool_name = message.get("tool")
        parameters = message.get("parameters", {})

        if tool_name in self.mcp_tools:
            result = await self.mcp_tools[tool_name](**parameters)

            return {
                "type": "MCP_TOOL_RESPONSE",
                "sender": self.agent_id,
                "receiver": message.get("sender"),
                "payload": result,
            }

        return None

    # ============= MCP Tool Handlers =============
    # ALL agent manager functionality exposed ONLY through these handlers

    async def _mcp_register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        mcp_tools: Optional[List[str]] = None,
        skill_cards: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """MCP handler for agent registration."""
        request = AgentRegistrationRequest(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities,
            mcp_tools=mcp_tools or [],
            skill_cards=skill_cards or [],
            metadata=metadata or {},
        )

        # Validate registration request
        validation_result = await self._validate_registration_request(request)
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": "Registration validation failed",
                "details": validation_result["errors"],
            }

        # Check MCP tool segregation
        segregation_result = await self._check_mcp_segregation(request)
        if not segregation_result["compliant"]:
            return {
                "success": False,
                "error": "MCP segregation violation",
                "details": segregation_result["violations"],
            }

        # Register the agent
        self.registered_agents[agent_id] = request

        # Log registration event
        await self._log_compliance_event(
            agent_id, "registration", {"type": agent_type, "capabilities": capabilities}
        )

        return {
            "success": True,
            "agent_id": agent_id,
            "status": "registered",
            "compliance_status": "compliant",
        }

    async def _mcp_validate_compliance(self, agent_id: str) -> Dict[str, Any]:
        """MCP handler for compliance validation."""
        if agent_id not in self.registered_agents:
            return {"success": False, "error": "Agent not registered"}

        agent = self.registered_agents[agent_id]

        # Check skill card compliance
        skill_compliance = await self._validate_skill_card_compliance(agent)

        # Check MCP segregation
        segregation_check = await self._check_mcp_segregation(agent)

        # Generate compliance report
        report = ComplianceReport(
            agent_id=agent_id,
            status=ComplianceStatus.COMPLIANT
            if skill_compliance["compliant"] and segregation_check["compliant"]
            else ComplianceStatus.NON_COMPLIANT,
            skill_card_compliance=skill_compliance.get("details", {}),
            mcp_segregation_status=segregation_check["compliant"],
            registration_status=True,
            violations=skill_compliance.get("violations", [])
            + segregation_check.get("violations", []),
            recommendations=skill_compliance.get("recommendations", []),
        )

        self.compliance_reports[agent_id] = report

        return {
            "success": True,
            "agent_id": agent_id,
            "compliance_status": report.status.value,
            "violations": report.violations,
            "recommendations": report.recommendations,
        }

    async def _mcp_audit_segregation(self, agent_id: str) -> Dict[str, Any]:
        """MCP handler for MCP segregation audit."""
        if agent_id not in self.registered_agents:
            return {"success": False, "error": "Agent not registered"}

        agent = self.registered_agents[agent_id]
        segregation_result = await self._check_mcp_segregation(agent)

        # Update segregation status
        self.mcp_segregation_status[agent_id] = segregation_result

        return {
            "success": True,
            "agent_id": agent_id,
            "compliant": segregation_result["compliant"],
            "violations": segregation_result.get("violations", []),
            "mcp_tools": agent.mcp_tools,
            "segregation_status": "enforced" if segregation_result["compliant"] else "violated",
        }

    async def _mcp_manage_lifecycle(
        self, agent_id: str, action: str, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """MCP handler for agent lifecycle management."""
        valid_actions = ["suspend", "resume", "terminate", "restart"]

        if action not in valid_actions:
            return {"success": False, "error": f"Invalid action. Must be one of {valid_actions}"}

        if agent_id not in self.registered_agents:
            return {"success": False, "error": "Agent not registered"}

        # Perform lifecycle action
        result = {"success": True, "agent_id": agent_id, "action": action}

        if action == "suspend":
            if agent_id in self.compliance_reports:
                self.compliance_reports[agent_id].status = ComplianceStatus.SUSPENDED
            result["status"] = "suspended"
        elif action == "resume":
            if agent_id in self.compliance_reports:
                self.compliance_reports[agent_id].status = ComplianceStatus.PENDING_REVIEW
            result["status"] = "resumed"
        elif action == "terminate":
            del self.registered_agents[agent_id]
            if agent_id in self.compliance_reports:
                del self.compliance_reports[agent_id]
            result["status"] = "terminated"
        elif action == "restart":
            result["status"] = "restarted"

        # Log lifecycle event
        await self._log_compliance_event(agent_id, f"lifecycle_{action}", {"reason": reason})

        return result

    async def _mcp_discover_agents(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """MCP handler for agent discovery."""
        agents = []

        for agent_id, agent in self.registered_agents.items():
            # Apply filters if provided
            if filters:
                if "agent_type" in filters and agent.agent_type != filters["agent_type"]:
                    continue
                if "capabilities" in filters:
                    required_caps = set(filters["capabilities"])
                    if not required_caps.issubset(set(agent.capabilities)):
                        continue

            # Get compliance status
            compliance_status = "unknown"
            if agent_id in self.compliance_reports:
                compliance_status = self.compliance_reports[agent_id].status.value

            agents.append(
                {
                    "agent_id": agent_id,
                    "agent_type": agent.agent_type,
                    "capabilities": agent.capabilities,
                    "mcp_tools": agent.mcp_tools,
                    "compliance_status": compliance_status,
                }
            )

        return {"success": True, "agents": agents, "total": len(agents)}

    async def _mcp_health_check(self) -> Dict[str, Any]:
        """MCP handler for system health check."""
        # Update health status
        self.system_health["last_check"] = datetime.utcnow()

        # Check database connection
        db_healthy = bool(self.db)

        # Check segregation manager
        segregation_healthy = bool(self.segregation_manager)

        # Calculate overall health
        all_healthy = db_healthy and segregation_healthy
        self.system_health["status"] = "healthy" if all_healthy else "degraded"

        return {
            "success": True,
            "status": self.system_health["status"],
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "segregation_manager": "healthy" if segregation_healthy else "unhealthy",
            },
            "registered_agents": len(self.registered_agents),
            "compliance_reports": len(self.compliance_reports),
            "last_check": self.system_health["last_check"].isoformat(),
        }

    async def _mcp_blockchain_register(
        self, agent_id: str, contract_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """MCP handler for blockchain registration."""
        if agent_id not in self.registered_agents:
            return {"success": False, "error": "Agent not registered"}

        # Mock blockchain registration (would interact with smart contract in production)
        tx_hash = f"0x{uuid.uuid4().hex}"

        return {
            "success": True,
            "agent_id": agent_id,
            "transaction_hash": tx_hash,
            "contract_address": contract_address or "0x0000000000000000000000000000000000000000",
            "status": "blockchain_registered",
        }

    async def _mcp_blockchain_update(
        self, agent_id: str, update_type: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """MCP handler for blockchain updates."""
        valid_updates = ["status", "capabilities", "metadata"]

        if update_type not in valid_updates:
            return {
                "success": False,
                "error": f"Invalid update type. Must be one of {valid_updates}",
            }

        # Mock blockchain update
        tx_hash = f"0x{uuid.uuid4().hex}"

        return {
            "success": True,
            "agent_id": agent_id,
            "update_type": update_type,
            "transaction_hash": tx_hash,
            "status": "blockchain_updated",
        }

    async def _mcp_skill_card_validate(self, skill_card_id: str, agent_id: str) -> Dict[str, Any]:
        """MCP handler for skill card validation."""
        if skill_card_id not in self.skill_cards:
            return {"success": False, "error": "Skill card not found"}

        if agent_id not in self.registered_agents:
            return {"success": False, "error": "Agent not registered"}

        skill_card = self.skill_cards[skill_card_id]
        agent = self.registered_agents[agent_id]

        # Check if agent has required capabilities
        missing_caps = set(skill_card.required_capabilities) - set(agent.capabilities)

        # Check if agent has required MCP tools
        missing_tools = set(skill_card.mcp_tools) - set(agent.mcp_tools)

        compliant = len(missing_caps) == 0 and len(missing_tools) == 0

        return {
            "success": True,
            "skill_card_id": skill_card_id,
            "agent_id": agent_id,
            "compliant": compliant,
            "missing_capabilities": list(missing_caps),
            "missing_tools": list(missing_tools),
        }

    async def _mcp_skill_card_issue(self, agent_id: str, skill_card_id: str) -> Dict[str, Any]:
        """MCP handler for issuing skill cards."""
        # First validate
        validation = await self._mcp_skill_card_validate(skill_card_id, agent_id)

        if not validation["success"]:
            return validation

        if not validation["compliant"]:
            return {
                "success": False,
                "error": "Agent not compliant with skill card requirements",
                "details": validation,
            }

        # Issue the skill card
        if agent_id not in self.registered_agents:
            return {"success": False, "error": "Agent not registered"}

        agent = self.registered_agents[agent_id]
        if skill_card_id not in agent.skill_cards:
            agent.skill_cards.append(skill_card_id)

        return {
            "success": True,
            "agent_id": agent_id,
            "skill_card_id": skill_card_id,
            "status": "issued",
            "issued_at": datetime.utcnow().isoformat(),
        }

    async def _mcp_enforce_policies(
        self, policy_type: str, targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """MCP handler for policy enforcement."""
        valid_policies = ["mcp_segregation", "skill_card_compliance", "registration_required"]

        if policy_type not in valid_policies:
            return {
                "success": False,
                "error": f"Invalid policy type. Must be one of {valid_policies}",
            }

        # If no targets specified, apply to all registered agents
        if not targets:
            targets = list(self.registered_agents.keys())

        enforcement_results = []

        for agent_id in targets:
            if agent_id not in self.registered_agents:
                enforcement_results.append({"agent_id": agent_id, "status": "not_found"})
                continue

            # Enforce policy
            if policy_type == "mcp_segregation":
                result = await self._mcp_audit_segregation(agent_id)
            elif policy_type == "skill_card_compliance":
                result = await self._mcp_validate_compliance(agent_id)
            else:  # registration_required
                result = {"success": True, "status": "registered"}

            enforcement_results.append(
                {
                    "agent_id": agent_id,
                    "status": "enforced" if result["success"] else "failed",
                    "details": result,
                }
            )

        return {
            "success": True,
            "policy_type": policy_type,
            "targets": targets,
            "results": enforcement_results,
        }

    async def _mcp_generate_report(
        self, report_type: str, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """MCP handler for report generation."""
        valid_reports = ["compliance", "registration", "health", "audit"]

        if report_type not in valid_reports:
            return {
                "success": False,
                "error": f"Invalid report type. Must be one of {valid_reports}",
            }

        report_data = {}

        if report_type == "compliance":
            report_data = {
                "total_agents": len(self.registered_agents),
                "compliant": sum(
                    1
                    for r in self.compliance_reports.values()
                    if r.status == ComplianceStatus.COMPLIANT
                ),
                "non_compliant": sum(
                    1
                    for r in self.compliance_reports.values()
                    if r.status == ComplianceStatus.NON_COMPLIANT
                ),
                "suspended": sum(
                    1
                    for r in self.compliance_reports.values()
                    if r.status == ComplianceStatus.SUSPENDED
                ),
                "details": [
                    {
                        "agent_id": agent_id,
                        "status": report.status.value,
                        "violations": report.violations,
                    }
                    for agent_id, report in self.compliance_reports.items()
                ],
            }
        elif report_type == "registration":
            report_data = {
                "total_registered": len(self.registered_agents),
                "by_type": {},
                "agents": [
                    {
                        "agent_id": agent_id,
                        "agent_type": agent.agent_type,
                        "capabilities": agent.capabilities,
                    }
                    for agent_id, agent in self.registered_agents.items()
                ],
            }
            # Count by type
            for agent in self.registered_agents.values():
                report_data["by_type"][agent.agent_type] = (
                    report_data["by_type"].get(agent.agent_type, 0) + 1
                )
        elif report_type == "health":
            report_data = await self._mcp_health_check()
        elif report_type == "audit":
            report_data = {
                "segregation_status": self.mcp_segregation_status,
                "compliance_reports": len(self.compliance_reports),
                "skill_cards_defined": len(self.skill_cards),
                "timestamp": datetime.utcnow().isoformat(),
            }

        return {
            "success": True,
            "report_type": report_type,
            "generated_at": datetime.utcnow().isoformat(),
            "data": report_data,
        }

    # ============= Alert & Notification MCP Handlers =============

    async def _mcp_send_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        agent_id: Optional[str] = None,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """MCP handler for sending alerts through configured channels."""
        # Import alert system if available
        try:
            from ....infrastructure.monitoring.real_alert_system import Alert, RealAlertSystem

            alert_system = RealAlertSystem()
        except ImportError:
            # Fallback to logging
            logger.warning(f"Alert System not available. Logging alert: {alert_type} - {message}")
            return {"success": False, "error": "Alert system not configured", "fallback": "logged"}

        # Create alert
        alert = Alert(
            id=f"agent_manager_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            title=f"Agent Manager Alert: {alert_type}",
            message=message,
            severity=severity,
            category="agent_management",
            source="agent_manager",
            context={"agent_id": agent_id} if agent_id else {},
            channels=channels,
        )

        # Send alert
        await alert_system.send_alert(alert)

        # Log compliance event if agent-specific
        if agent_id:
            await self._log_compliance_event(
                agent_id, f"alert_{alert_type}", {"severity": severity, "message": message}
            )

        return {
            "success": True,
            "alert_id": alert.id,
            "channels": channels or ["all"],
            "severity": severity,
            "status": "sent",
        }

    async def _mcp_configure_alerts(
        self, channel_type: str, config: Dict[str, Any], severity_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """MCP handler for configuring alert channels."""
        # Store alert configuration
        if not hasattr(self, "alert_configs"):
            self.alert_configs = {}

        self.alert_configs[channel_type] = {
            "config": config,
            "severity_filter": severity_filter or ["low", "medium", "high", "critical"],
            "enabled": True,
            "configured_at": datetime.now().isoformat(),
        }

        return {
            "success": True,
            "channel_type": channel_type,
            "status": "configured",
            "severity_filter": severity_filter,
        }

    # ============= Blockchain Operations MCP Handlers =============

    async def _mcp_check_wallet_balance(
        self, agent_id: str, network: str = "local_anvil", include_tokens: bool = False
    ) -> Dict[str, Any]:
        """MCP handler for checking agent wallet balance."""
        # Get agent's wallet address
        if agent_id not in self.registered_agents:
            return {"success": False, "error": "Agent not registered"}

        # Import Web3 service if available
        try:
            from ....infrastructure.blockchain.web3_service import Web3Service

            web3_service = Web3Service()
        except ImportError:
            # Mock response for local testing
            return {
                "success": True,
                "agent_id": agent_id,
                "network": network,
                "balance": "1.0",  # Mock 1 ETH
                "balance_usd": 2000.0,
                "tokens": []
                if not include_tokens
                else [
                    {"symbol": "USDT", "balance": "1000.0"},
                    {"symbol": "LINK", "balance": "50.0"},
                ],
                "mock_data": True,
            }

        # Get wallet address for agent (mock for now)
        wallet_address = f"0x{'0' * 39}{agent_id[-1]}"

        # Check balance
        balance = await web3_service.get_wallet_balance(wallet_address, network)

        result = {
            "success": True,
            "agent_id": agent_id,
            "wallet_address": wallet_address,
            "network": network,
            "balance": str(balance.balance),
            "balance_usd": float(balance.balance_usd) if balance.balance_usd else None,
        }

        if include_tokens:
            tokens = await web3_service.get_token_balances(wallet_address, network)
            result["tokens"] = [
                {"symbol": t.symbol, "balance": str(t.balance), "contract": t.contract_address}
                for t in tokens
            ]

        return result

    async def _mcp_execute_blockchain_tx(
        self, agent_id: str, tx_type: str, data: Dict[str, Any], network: str = "local_anvil"
    ) -> Dict[str, Any]:
        """MCP handler for executing blockchain transactions."""
        if agent_id not in self.registered_agents:
            return {"success": False, "error": "Agent not registered"}

        # Validate transaction type
        valid_tx_types = ["register", "update_status", "send_message", "update_capabilities"]
        if tx_type not in valid_tx_types:
            return {
                "success": False,
                "error": f"Invalid transaction type. Must be one of {valid_tx_types}",
            }

        # Mock transaction for local Anvil
        if network == "local_anvil":
            tx_hash = f"0x{uuid.uuid4().hex}"

            # Log blockchain event
            await self._log_compliance_event(
                agent_id,
                f"blockchain_{tx_type}",
                {"network": network, "tx_hash": tx_hash, "data": data},
            )

            return {
                "success": True,
                "agent_id": agent_id,
                "tx_type": tx_type,
                "transaction_hash": tx_hash,
                "network": network,
                "block_number": 12345,  # Mock
                "gas_used": 21000,  # Mock
                "status": "confirmed",
            }

        # For real networks, would use Web3Service
        return {"success": False, "error": f"Network {network} not yet supported"}

    async def _mcp_manage_gas(
        self, network: str, operation: str, max_gas_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """MCP handler for gas optimization."""
        # Analyze gas costs for operation
        gas_estimates = {
            "register": 150000,
            "update_status": 50000,
            "send_message": 75000,
            "update_capabilities": 100000,
        }

        estimated_gas = gas_estimates.get(operation, 21000)

        # Get current gas prices (mock)
        gas_prices = {"slow": 10, "standard": 20, "fast": 30}  # gwei

        # Calculate costs
        costs = {}
        for speed, price in gas_prices.items():
            if max_gas_price and price > max_gas_price:
                continue
            costs[speed] = {
                "gas_price_gwei": price,
                "estimated_gas": estimated_gas,
                "estimated_cost_eth": (price * estimated_gas) / 1e9,
                "estimated_cost_usd": (price * estimated_gas) / 1e9 * 2000,  # Mock ETH price
            }

        return {
            "success": True,
            "network": network,
            "operation": operation,
            "gas_estimates": costs,
            "recommended": "standard" if "standard" in costs else "slow",
            "max_gas_price": max_gas_price,
        }

    async def _mcp_track_contracts(
        self, contracts: List[Dict[str, Any]], events: List[str], real_time: bool = True
    ) -> Dict[str, Any]:
        """MCP handler for tracking smart contracts."""
        # Store contract tracking configuration
        if not hasattr(self, "tracked_contracts"):
            self.tracked_contracts = []

        for contract in contracts:
            tracking_config = {
                "address": contract.get("address"),
                "abi": contract.get("abi"),
                "events": events,
                "real_time": real_time,
                "started_at": datetime.now().isoformat(),
            }
            self.tracked_contracts.append(tracking_config)

        return {
            "success": True,
            "tracking": len(contracts),
            "events": events,
            "real_time": real_time,
            "status": "tracking_started",
        }

    # ============= DeFi Monitoring MCP Handlers =============

    async def _mcp_monitor_defi(
        self,
        protocols: List[str],
        metrics: List[str],
        alert_thresholds: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """MCP handler for monitoring DeFi protocols."""
        # Import DEX service if available
        try:
            from ....infrastructure.defi.dex_service import DEXService

            dex_service = DEXService()
        except ImportError:
            # Mock DeFi data
            mock_data = {}
            for protocol in protocols:
                mock_data[protocol] = {
                    "tvl": 1000000000,  # $1B
                    "apy": 0.05,  # 5%
                    "volume": 50000000,  # $50M
                    "fees": 150000,  # $150k
                }

            return {
                "success": True,
                "protocols": protocols,
                "metrics": metrics,
                "data": mock_data,
                "mock_data": True,
                "timestamp": datetime.now().isoformat(),
            }

        # Monitor real DeFi data
        defi_data = {}
        alerts = []

        for protocol in protocols:
            protocol_data = await dex_service.get_trending_pools(protocol)

            # Extract requested metrics
            defi_data[protocol] = {}
            for metric in metrics:
                if metric == "tvl":
                    defi_data[protocol]["tvl"] = protocol_data.get("total_liquidity_usd", 0)
                elif metric == "apy":
                    defi_data[protocol]["apy"] = protocol_data.get("apy", 0)
                elif metric == "volume":
                    defi_data[protocol]["volume"] = protocol_data.get("volume_24h", 0)
                elif metric == "fees":
                    defi_data[protocol]["fees"] = protocol_data.get("fees_24h", 0)

            # Check alert thresholds
            if alert_thresholds:
                for metric, threshold in alert_thresholds.items():
                    if metric in defi_data[protocol]:
                        value = defi_data[protocol][metric]
                        if value > threshold.get("max", float("inf")) or value < threshold.get(
                            "min", 0
                        ):
                            alerts.append(
                                {
                                    "protocol": protocol,
                                    "metric": metric,
                                    "value": value,
                                    "threshold": threshold,
                                }
                            )

        result = {
            "success": True,
            "protocols": protocols,
            "metrics": metrics,
            "data": defi_data,
            "timestamp": datetime.now().isoformat(),
        }

        if alerts:
            result["alerts"] = alerts
            # Send alerts for threshold violations
            for alert in alerts:
                await self._mcp_send_alert(
                    alert_type="defi_threshold",
                    severity="high",
                    message=f"DeFi alert: {alert['protocol']} {alert['metric']} = {alert['value']}",
                )

        return result


# Factory function
async def create_agent_manager() -> AgentManagerAgent:
    """Create and initialize Agent Manager"""
    manager = AgentManagerAgent()
    await manager.initialize()
    return manager
