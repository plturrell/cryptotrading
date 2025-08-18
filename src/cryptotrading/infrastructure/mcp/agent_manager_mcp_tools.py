"""
MCP Tools for Agent Manager
Exposes agent registration, compliance, and orchestration capabilities via Model Context Protocol
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
import json
from datetime import datetime, timedelta
import uuid

# Import agent manager components
from ...core.agents.specialized.agent_manager import (
    AgentManagerAgent,
    AgentRegistrationRequest,
    ComplianceReport,
    ComplianceStatus,
    SkillCard
)

logger = logging.getLogger(__name__)

class AgentManagerMCPTools:
    """MCP tools for Agent Manager operations"""
    
    def __init__(self):
        self.agent_manager = AgentManagerAgent()
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions"""
        return [
            {
                "name": "register_agent",
                "description": "Register a new agent in the A2A system",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Unique agent identifier"
                        },
                        "agent_type": {
                            "type": "string",
                            "description": "Type of agent (technical_analysis, mcts_calculation, etc.)"
                        },
                        "capabilities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of agent capabilities"
                        },
                        "mcp_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of MCP tools provided by agent"
                        },
                        "skill_cards": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of skill card IDs"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional agent metadata"
                        }
                    },
                    "required": ["agent_id", "agent_type", "capabilities"]
                }
            },
            {
                "name": "get_agent_status",
                "description": "Get status and compliance information for an agent",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Agent identifier"
                        },
                        "include_compliance": {
                            "type": "boolean",
                            "description": "Include compliance assessment",
                            "default": True
                        },
                        "include_mcp_status": {
                            "type": "boolean",
                            "description": "Include MCP segregation status",
                            "default": True
                        }
                    },
                    "required": ["agent_id"]
                }
            },
            {
                "name": "list_registered_agents",
                "description": "List all registered agents with filtering options",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_type": {
                            "type": "string",
                            "description": "Filter by agent type"
                        },
                        "status": {
                            "type": "string",
                            "description": "Filter by agent status",
                            "enum": ["active", "inactive", "suspended", "all"]
                        },
                        "has_mcp_tools": {
                            "type": "boolean",
                            "description": "Filter agents with MCP tools"
                        },
                        "compliance_status": {
                            "type": "string",
                            "description": "Filter by compliance status",
                            "enum": ["compliant", "non_compliant", "pending_review", "suspended"]
                        }
                    }
                }
            },
            {
                "name": "assess_agent_compliance",
                "description": "Perform comprehensive compliance assessment for an agent",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Agent identifier"
                        },
                        "check_skill_cards": {
                            "type": "boolean",
                            "description": "Check skill card compliance",
                            "default": True
                        },
                        "check_mcp_segregation": {
                            "type": "boolean",
                            "description": "Check MCP segregation compliance",
                            "default": True
                        },
                        "check_registration": {
                            "type": "boolean",
                            "description": "Check registration compliance",
                            "default": True
                        }
                    },
                    "required": ["agent_id"]
                }
            },
            {
                "name": "create_skill_card",
                "description": "Create a new A2A skill card definition",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {
                            "type": "string",
                            "description": "Unique skill identifier"
                        },
                        "skill_name": {
                            "type": "string",
                            "description": "Human-readable skill name"
                        },
                        "description": {
                            "type": "string",
                            "description": "Skill description"
                        },
                        "required_capabilities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Required capabilities for this skill"
                        },
                        "mcp_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "MCP tools associated with this skill"
                        },
                        "compliance_rules": {
                            "type": "object",
                            "description": "Compliance rules for this skill"
                        },
                        "version": {
                            "type": "string",
                            "description": "Skill card version",
                            "default": "1.0"
                        }
                    },
                    "required": ["skill_id", "skill_name", "description", "required_capabilities"]
                }
            },
            {
                "name": "get_skill_cards",
                "description": "Get skill card definitions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {
                            "type": "string",
                            "description": "Specific skill ID to retrieve"
                        },
                        "agent_type": {
                            "type": "string",
                            "description": "Filter by agent type"
                        },
                        "include_compliance_rules": {
                            "type": "boolean",
                            "description": "Include compliance rules",
                            "default": True
                        }
                    }
                }
            },
            {
                "name": "enforce_mcp_segregation",
                "description": "Enforce MCP segregation policies for agents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Agent to enforce segregation for"
                        },
                        "segregation_level": {
                            "type": "string",
                            "description": "Level of segregation to enforce",
                            "enum": ["strict", "moderate", "basic"],
                            "default": "strict"
                        },
                        "resource_limits": {
                            "type": "object",
                            "properties": {
                                "max_requests_per_hour": {"type": "integer"},
                                "max_memory_mb": {"type": "integer"},
                                "max_cpu_time_seconds": {"type": "integer"}
                            },
                            "description": "Resource limits for the agent"
                        }
                    },
                    "required": ["agent_id"]
                }
            },
            {
                "name": "get_system_health",
                "description": "Get overall A2A system health and statistics",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_agent_stats": {
                            "type": "boolean",
                            "description": "Include agent statistics",
                            "default": True
                        },
                        "include_compliance_summary": {
                            "type": "boolean",
                            "description": "Include compliance summary",
                            "default": True
                        },
                        "include_mcp_status": {
                            "type": "boolean",
                            "description": "Include MCP system status",
                            "default": True
                        }
                    }
                }
            }
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        try:
            if tool_name == "register_agent":
                return await self._register_agent(arguments)
            elif tool_name == "get_agent_status":
                return await self._get_agent_status(arguments)
            elif tool_name == "list_registered_agents":
                return await self._list_registered_agents(arguments)
            elif tool_name == "assess_agent_compliance":
                return await self._assess_agent_compliance(arguments)
            elif tool_name == "create_skill_card":
                return await self._create_skill_card(arguments)
            elif tool_name == "get_skill_cards":
                return await self._get_skill_cards(arguments)
            elif tool_name == "enforce_mcp_segregation":
                return await self._enforce_mcp_segregation(arguments)
            elif tool_name == "get_system_health":
                return await self._get_system_health(arguments)
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }
        except Exception as e:
            logger.error(f"Error in {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    async def _register_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new agent"""
        try:
            request = AgentRegistrationRequest(
                agent_id=args["agent_id"],
                agent_type=args["agent_type"],
                capabilities=args["capabilities"],
                mcp_tools=args.get("mcp_tools", []),
                skill_cards=args.get("skill_cards", []),
                metadata=args.get("metadata", {})
            )
            
            # Use agent manager to register
            result = await self.agent_manager.register_agent(request)
            
            return {
                "success": True,
                "agent_id": args["agent_id"],
                "registration_status": result.get("status", "registered"),
                "assigned_role": result.get("role", "basic_user"),
                "mcp_tools_registered": len(args.get("mcp_tools", [])),
                "skill_cards_assigned": len(args.get("skill_cards", [])),
                "registration_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to register agent: {str(e)}"
            }
    
    async def _get_agent_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get agent status information"""
        agent_id = args["agent_id"]
        include_compliance = args.get("include_compliance", True)
        include_mcp_status = args.get("include_mcp_status", True)
        
        try:
            # Get basic agent info
            agent_info = await self.agent_manager.get_agent_info(agent_id)
            
            status_data = {
                "agent_id": agent_id,
                "status": agent_info.get("status", "unknown"),
                "agent_type": agent_info.get("agent_type"),
                "capabilities": agent_info.get("capabilities", []),
                "mcp_tools": agent_info.get("mcp_tools", []),
                "last_seen": agent_info.get("last_seen"),
                "registration_date": agent_info.get("registration_date")
            }
            
            # Include compliance assessment if requested
            if include_compliance:
                compliance = await self.agent_manager.assess_compliance(agent_id)
                status_data["compliance"] = {
                    "status": compliance.status.value if compliance else "unknown",
                    "skill_card_compliance": compliance.skill_card_compliance if compliance else {},
                    "violations": compliance.violations if compliance else [],
                    "last_checked": compliance.last_checked.isoformat() if compliance else None
                }
            
            # Include MCP status if requested
            if include_mcp_status:
                mcp_status = await self.agent_manager.get_mcp_segregation_status(agent_id)
                status_data["mcp_segregation"] = {
                    "enabled": mcp_status.get("enabled", False),
                    "isolation_level": mcp_status.get("isolation_level"),
                    "resource_usage": mcp_status.get("resource_usage", {}),
                    "quota_limits": mcp_status.get("quota_limits", {})
                }
            
            return {
                "success": True,
                "agent_status": status_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get agent status: {str(e)}"
            }
    
    async def _list_registered_agents(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List registered agents with filtering"""
        try:
            # Apply filters
            filters = {}
            if args.get("agent_type"):
                filters["agent_type"] = args["agent_type"]
            if args.get("status"):
                filters["status"] = args["status"]
            if args.get("has_mcp_tools") is not None:
                filters["has_mcp_tools"] = args["has_mcp_tools"]
            if args.get("compliance_status"):
                filters["compliance_status"] = args["compliance_status"]
            
            # Get agents list
            agents = await self.agent_manager.list_agents(filters)
            
            # Process results
            agent_summary = {
                "total_agents": len(agents),
                "by_type": {},
                "by_status": {},
                "with_mcp_tools": 0,
                "compliant_agents": 0
            }
            
            for agent in agents:
                # Count by type
                agent_type = agent.get("agent_type", "unknown")
                agent_summary["by_type"][agent_type] = agent_summary["by_type"].get(agent_type, 0) + 1
                
                # Count by status
                status = agent.get("status", "unknown")
                agent_summary["by_status"][status] = agent_summary["by_status"].get(status, 0) + 1
                
                # Count MCP tools
                if agent.get("mcp_tools"):
                    agent_summary["with_mcp_tools"] += 1
                
                # Count compliant
                if agent.get("compliance_status") == "compliant":
                    agent_summary["compliant_agents"] += 1
            
            return {
                "success": True,
                "agents": agents,
                "summary": agent_summary,
                "filters_applied": filters
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to list agents: {str(e)}"
            }
    
    async def _assess_agent_compliance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Assess agent compliance"""
        agent_id = args["agent_id"]
        
        try:
            # Perform compliance assessment
            compliance_report = await self.agent_manager.assess_compliance(
                agent_id,
                check_skill_cards=args.get("check_skill_cards", True),
                check_mcp_segregation=args.get("check_mcp_segregation", True),
                check_registration=args.get("check_registration", True)
            )
            
            if compliance_report:
                assessment_result = {
                    "agent_id": agent_id,
                    "overall_status": compliance_report.status.value,
                    "registration_compliant": compliance_report.registration_status,
                    "mcp_segregation_compliant": compliance_report.mcp_segregation_status,
                    "skill_card_compliance": compliance_report.skill_card_compliance,
                    "violations": compliance_report.violations,
                    "recommendations": compliance_report.recommendations,
                    "assessment_timestamp": compliance_report.last_checked.isoformat(),
                    "compliance_score": self._calculate_compliance_score(compliance_report)
                }
            else:
                assessment_result = {
                    "agent_id": agent_id,
                    "overall_status": "unknown",
                    "error": "Could not perform compliance assessment"
                }
            
            return {
                "success": True,
                "compliance_assessment": assessment_result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to assess compliance: {str(e)}"
            }
    
    def _calculate_compliance_score(self, report: ComplianceReport) -> float:
        """Calculate overall compliance score"""
        scores = []
        
        if report.registration_status:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        if report.mcp_segregation_status:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Skill card compliance score
        if report.skill_card_compliance:
            skill_scores = list(report.skill_card_compliance.values())
            if skill_scores:
                avg_skill_score = sum(1.0 for s in skill_scores if s) / len(skill_scores)
                scores.append(avg_skill_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def _create_skill_card(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new skill card"""
        try:
            skill_card = SkillCard(
                skill_id=args["skill_id"],
                skill_name=args["skill_name"],
                description=args["description"],
                required_capabilities=args["required_capabilities"],
                mcp_tools=args["mcp_tools"],
                compliance_rules=args.get("compliance_rules", {}),
                version=args.get("version", "1.0")
            )
            
            # Store skill card
            result = await self.agent_manager.create_skill_card(skill_card)
            
            return {
                "success": True,
                "skill_card": {
                    "skill_id": skill_card.skill_id,
                    "skill_name": skill_card.skill_name,
                    "version": skill_card.version,
                    "required_capabilities": skill_card.required_capabilities,
                    "mcp_tools": skill_card.mcp_tools
                },
                "creation_status": result.get("status", "created"),
                "creation_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create skill card: {str(e)}"
            }
    
    async def _get_skill_cards(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get skill card definitions"""
        try:
            filters = {}
            if args.get("skill_id"):
                filters["skill_id"] = args["skill_id"]
            if args.get("agent_type"):
                filters["agent_type"] = args["agent_type"]
            
            skill_cards = await self.agent_manager.get_skill_cards(filters)
            
            # Process skill cards
            cards_data = []
            for card in skill_cards:
                card_data = {
                    "skill_id": card.skill_id,
                    "skill_name": card.skill_name,
                    "description": card.description,
                    "version": card.version,
                    "required_capabilities": card.required_capabilities,
                    "mcp_tools": card.mcp_tools
                }
                
                if args.get("include_compliance_rules", True):
                    card_data["compliance_rules"] = card.compliance_rules
                
                cards_data.append(card_data)
            
            return {
                "success": True,
                "skill_cards": cards_data,
                "total_cards": len(cards_data)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get skill cards: {str(e)}"
            }
    
    async def _enforce_mcp_segregation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce MCP segregation for an agent"""
        agent_id = args["agent_id"]
        segregation_level = args.get("segregation_level", "strict")
        resource_limits = args.get("resource_limits", {})
        
        try:
            # Apply segregation policies
            result = await self.agent_manager.enforce_mcp_segregation(
                agent_id, 
                segregation_level,
                resource_limits
            )
            
            return {
                "success": True,
                "agent_id": agent_id,
                "segregation_level": segregation_level,
                "resource_limits": resource_limits,
                "enforcement_status": result.get("status", "applied"),
                "isolation_context": result.get("isolation_context"),
                "enforcement_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to enforce MCP segregation: {str(e)}"
            }
    
    async def _get_system_health(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get overall system health"""
        try:
            health_data = {
                "system_status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "agent_manager_version": "1.0.0"
            }
            
            if args.get("include_agent_stats", True):
                agent_stats = await self.agent_manager.get_agent_statistics()
                health_data["agent_statistics"] = agent_stats
            
            if args.get("include_compliance_summary", True):
                compliance_summary = await self.agent_manager.get_compliance_summary()
                health_data["compliance_summary"] = compliance_summary
            
            if args.get("include_mcp_status", True):
                mcp_status = await self.agent_manager.get_mcp_system_status()
                health_data["mcp_system_status"] = mcp_status
            
            return {
                "success": True,
                "system_health": health_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get system health: {str(e)}",
                "system_status": "unhealthy"
            }

# Export for MCP server registration
agent_manager_mcp_tools = AgentManagerMCPTools()
