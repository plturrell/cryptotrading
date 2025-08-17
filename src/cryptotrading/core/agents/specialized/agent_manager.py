"""
A2A Agent Manager - Strand Agent
Enforces (1) Registration (2) MCP Segregation (3) A2A Skill Card Compliance
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid

from ..strands import StrandsAgent
from ...protocols.a2a.a2a_protocol import A2A_CAPABILITIES, MessageType, AgentStatus
try:
    from ....infrastructure.analysis.mcp_agent_segregation import (
        get_segregation_manager,
        AgentContext,
        AgentRole,
        ResourceType
    )
    from ....data.database.client import get_db
    from ....data.database.models import A2AAgent
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

class AgentManagerAgent(StrandsAgent):
    """
    A2A Agent Manager - Strand Agent
    Enforces registration, MCP segregation, and A2A skill card compliance
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
                "compliance_monitoring"
            ],
            **kwargs
        )
        
        self.db = get_db()
        self.segregation_manager = get_segregation_manager()
        self.registered_agents: Dict[str, AgentRegistrationRequest] = {}
        self.skill_cards: Dict[str, SkillCard] = {}
        self.compliance_reports: Dict[str, ComplianceReport] = {}
        
        # Initialize core skill cards
        self._initialize_skill_cards()
        
        logger.info(f"Agent Manager {agent_id} initialized")
    
    def _initialize_skill_cards(self):
        """Initialize core A2A skill cards"""
        self.skill_cards = {
            "historical_data_loading": SkillCard(
                skill_id="historical_data_loading",
                skill_name="Historical Data Loading",
                description="Load and manage historical market data",
                required_capabilities=["data_loading", "historical_data", "technical_indicators"],
                mcp_tools=["get_market_data", "get_historical_prices", "calculate_technical_indicators"],
                compliance_rules={
                    "max_data_retention_days": 365,
                    "required_data_sources": ["yahoo_finance", "fred"],
                    "mandatory_indicators": ["sma", "ema", "rsi"]
                }
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
                    "mandatory_reporting": ["daily_pnl", "risk_metrics"]
                }
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
                    "confidence_threshold": 0.7
                }
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
                    "security_scan_required": True
                }
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
                    "required_validation": "backtesting"
                }
            )
        }
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent management requests"""
        msg_type = message.get('type')
        
        if msg_type == 'register_agent':
            return await self._handle_agent_registration(message)
        elif msg_type == 'check_compliance':
            return await self._handle_compliance_check(message)
        elif msg_type == 'enforce_segregation':
            return await self._handle_segregation_enforcement(message)
        elif msg_type == 'discover_agents':
            return await self._handle_agent_discovery(message)
        elif msg_type == 'validate_skill_card':
            return await self._handle_skill_card_validation(message)
        else:
            return {'error': f'Unknown message type: {msg_type}'}
    
    async def _handle_agent_registration(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent registration with compliance checks"""
        try:
            request_data = message.get('payload', {})
            request = AgentRegistrationRequest(**request_data)
            
            # Step 1: Validate registration request
            validation_result = await self._validate_registration_request(request)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': 'Registration validation failed',
                    'details': validation_result['errors']
                }
            
            # Step 2: Check MCP tool segregation
            segregation_result = await self._check_mcp_segregation(request)
            if not segregation_result['compliant']:
                return {
                    'success': False,
                    'error': 'MCP segregation violation',
                    'details': segregation_result['violations']
                }
            
            # Step 3: Validate skill card compliance
            skill_compliance = await self._validate_skill_card_compliance(request)
            if not skill_compliance['compliant']:
                return {
                    'success': False,
                    'error': 'Skill card compliance violation',
                    'details': skill_compliance['violations']
                }
            
            # Step 4: Register agent in database
            registration_result = await self._register_agent_in_db(request)
            if not registration_result['success']:
                return registration_result
            
            # Step 5: Configure MCP segregation
            await self._configure_agent_segregation(request)
            
            # Step 6: Store registration
            self.registered_agents[request.agent_id] = request
            
            # Step 7: Generate compliance report
            compliance_report = ComplianceReport(
                agent_id=request.agent_id,
                status=ComplianceStatus.COMPLIANT,
                skill_card_compliance=skill_compliance['skill_status'],
                mcp_segregation_status=True,
                registration_status=True,
                violations=[],
                recommendations=[]
            )
            self.compliance_reports[request.agent_id] = compliance_report
            
            logger.info(f"Agent {request.agent_id} registered successfully")
            
            return {
                'success': True,
                'agent_id': request.agent_id,
                'compliance_status': compliance_report.status.value,
                'assigned_mcp_tools': request.mcp_tools,
                'skill_cards': request.skill_cards,
                'registration_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _validate_registration_request(self, request: AgentRegistrationRequest) -> Dict[str, Any]:
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
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _check_mcp_segregation(self, request: AgentRegistrationRequest) -> Dict[str, Any]:
        """Check MCP tool segregation compliance"""
        violations = []
        
        # Check if MCP tools are already assigned to other agents
        for tool_name in request.mcp_tools:
            for existing_agent_id, existing_request in self.registered_agents.items():
                if tool_name in existing_request.mcp_tools:
                    violations.append(f"MCP tool '{tool_name}' already assigned to agent '{existing_agent_id}'")
        
        # Validate tool ownership boundaries
        agent_type = request.agent_type
        allowed_tools = self._get_allowed_mcp_tools_for_agent_type(agent_type)
        
        for tool_name in request.mcp_tools:
            if tool_name not in allowed_tools:
                violations.append(f"MCP tool '{tool_name}' not allowed for agent type '{agent_type}'")
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
    
    def _get_allowed_mcp_tools_for_agent_type(self, agent_type: str) -> List[str]:
        """Get allowed MCP tools for agent type"""
        tool_mappings = {
            "historical_loader": [
                "get_market_data", "get_historical_prices", "calculate_technical_indicators"
            ],
            "database_manager": [
                "get_portfolio", "get_wallet_balance", "store_transaction_history"
            ],
            "data_management": [
                "validate_data_schema", "discover_data_sources"
            ],
            "strands_agent": [
                "analyze_clrs_tree", "generate_dependency_graph", "optimize_code"
            ],
            "mcts_calculation": [
                "mcts_calculate", "evaluate_strategy", "optimize_parameters"
            ]
        }
        
        return tool_mappings.get(agent_type, [])
    
    async def _validate_skill_card_compliance(self, request: AgentRegistrationRequest) -> Dict[str, Any]:
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
                violations.append(f"Skill card '{skill_card_id}' requires capabilities: {list(missing)}")
                skill_status[skill_card_id] = False
            else:
                skill_status[skill_card_id] = True
            
            # Check required MCP tools
            required_tools = set(skill_card.mcp_tools)
            agent_tools = set(request.mcp_tools)
            
            if not required_tools.issubset(agent_tools):
                missing_tools = required_tools - agent_tools
                violations.append(f"Skill card '{skill_card_id}' requires MCP tools: {list(missing_tools)}")
                skill_status[skill_card_id] = False
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'skill_status': skill_status
        }
    
    async def _register_agent_in_db(self, request: AgentRegistrationRequest) -> Dict[str, Any]:
        """Register agent in database"""
        try:
            with self.db.get_session() as session:
                # Check if agent already exists
                existing = session.query(A2AAgent).filter(
                    A2AAgent.agent_id == request.agent_id
                ).first()
                
                if existing:
                    return {'success': False, 'error': 'Agent already exists in database'}
                
                # Create new agent record
                agent = A2AAgent(
                    agent_id=request.agent_id,
                    agent_type=request.agent_type,
                    capabilities=json.dumps(request.capabilities),
                    metadata=json.dumps({
                        'mcp_tools': request.mcp_tools,
                        'skill_cards': request.skill_cards,
                        **request.metadata
                    }),
                    status='active',
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow()
                )
                
                session.add(agent)
                session.commit()
                
                return {'success': True, 'database_id': agent.id}
                
        except Exception as e:
            logger.error(f"Database registration failed: {e}")
            return {'success': False, 'error': str(e)}
    
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
                resource_quotas={}
            )
            
            # Register agent with segregation manager
            await self.segregation_manager.register_agent_context(agent_context)
            
            # Assign MCP tools to agent
            for tool_name in request.mcp_tools:
                await self.segregation_manager.assign_tool_to_agent(
                    agent_id=request.agent_id,
                    tool_name=tool_name,
                    resource_type=ResourceType.MCP_TOOL
                )
            
            logger.info(f"Configured MCP segregation for agent {request.agent_id}")
            
        except Exception as e:
            logger.error(f"Segregation configuration failed: {e}")
            raise
    
    async def _handle_compliance_check(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compliance check request"""
        agent_id = message.get('payload', {}).get('agent_id')
        
        if not agent_id:
            return {'success': False, 'error': 'Agent ID required'}
        
        if agent_id not in self.compliance_reports:
            return {'success': False, 'error': 'Agent not found'}
        
        report = self.compliance_reports[agent_id]
        
        return {
            'success': True,
            'compliance_report': report.__dict__
        }
    
    async def _handle_segregation_enforcement(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle segregation enforcement request"""
        payload = message.get('payload', {})
        action = payload.get('action')
        agent_id = payload.get('agent_id')
        
        if action == 'audit_segregation':
            return await self._audit_mcp_segregation()
        elif action == 'enforce_boundaries':
            return await self._enforce_agent_boundaries(agent_id)
        else:
            return {'success': False, 'error': f'Unknown enforcement action: {action}'}
    
    async def _audit_mcp_segregation(self) -> Dict[str, Any]:
        """Audit MCP tool segregation across all agents"""
        violations = []
        tool_assignments = {}
        
        # Check for duplicate tool assignments
        for agent_id, request in self.registered_agents.items():
            for tool_name in request.mcp_tools:
                if tool_name in tool_assignments:
                    violations.append({
                        'type': 'duplicate_assignment',
                        'tool': tool_name,
                        'agents': [tool_assignments[tool_name], agent_id]
                    })
                else:
                    tool_assignments[tool_name] = agent_id
        
        return {
            'success': True,
            'audit_results': {
                'total_agents': len(self.registered_agents),
                'total_tools': len(tool_assignments),
                'violations': violations,
                'segregation_healthy': len(violations) == 0
            }
        }
    
    async def _handle_agent_discovery(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent discovery request"""
        filters = message.get('payload', {}).get('filters', {})
        
        agents = []
        for agent_id, request in self.registered_agents.items():
            agent_info = {
                'agent_id': agent_id,
                'agent_type': request.agent_type,
                'capabilities': request.capabilities,
                'mcp_tools': request.mcp_tools,
                'skill_cards': request.skill_cards,
                'compliance_status': self.compliance_reports.get(agent_id, {}).status.value if agent_id in self.compliance_reports else 'unknown'
            }
            
            # Apply filters
            if self._matches_filters(agent_info, filters):
                agents.append(agent_info)
        
        return {
            'success': True,
            'agents': agents,
            'total_count': len(agents)
        }
    
    def _matches_filters(self, agent_info: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if agent matches discovery filters"""
        if 'agent_type' in filters and agent_info['agent_type'] != filters['agent_type']:
            return False
        
        if 'capability' in filters and filters['capability'] not in agent_info['capabilities']:
            return False
        
        if 'mcp_tool' in filters and filters['mcp_tool'] not in agent_info['mcp_tools']:
            return False
        
        return True
    
    async def _handle_skill_card_validation(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle skill card validation request"""
        skill_card_id = message.get('payload', {}).get('skill_card_id')
        agent_id = message.get('payload', {}).get('agent_id')
        
        if not skill_card_id or not agent_id:
            return {'success': False, 'error': 'Skill card ID and agent ID required'}
        
        if agent_id not in self.registered_agents:
            return {'success': False, 'error': 'Agent not found'}
        
        if skill_card_id not in self.skill_cards:
            return {'success': False, 'error': 'Skill card not found'}
        
        request = self.registered_agents[agent_id]
        compliance = await self._validate_skill_card_compliance(request)
        
        return {
            'success': True,
            'validation_result': {
                'skill_card_id': skill_card_id,
                'agent_id': agent_id,
                'compliant': compliance['skill_status'].get(skill_card_id, False),
                'violations': [v for v in compliance['violations'] if skill_card_id in v]
            }
        }

# Factory function
async def create_agent_manager() -> AgentManagerAgent:
    """Create and initialize Agent Manager"""
    manager = AgentManagerAgent()
    await manager.initialize()
    return manager
