"""
MCP-Strand Framework Integration
Bridges MCP tools and resources with Strand agents for seamless workflow integration
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
import json

# MCP imports
from .server import MCPServer
from .client import MCPClient
from .tools import MCPTool, ToolResult
from .auth import AuthContext
from .cache import mcp_cache
from .metrics import mcp_metrics
from .events import EventType, MCPEvent, create_event_publisher

# Strand imports - use the actual project paths
try:
    from ...agents.strands import StrandsAgent
    STRAND_AVAILABLE = True
except ImportError:
    STRAND_AVAILABLE = False
    logging.warning("Strand framework not available - MCP integration will be limited")
    # Import base agent as fallback
    from ...agents.base import BaseAgent as StrandsAgent

# Optional external strands framework
try:
    from strands import Agent, tool
    EXTERNAL_STRANDS_AVAILABLE = True
except ImportError:
    EXTERNAL_STRANDS_AVAILABLE = False
    def tool(func):
        return func

logger = logging.getLogger(__name__)


@dataclass
class MCPStrandConfig:
    """Configuration for MCP-Strand integration"""
    enable_tool_bridging: bool = True
    enable_resource_sharing: bool = True
    enable_workflow_orchestration: bool = True
    cache_tool_results: bool = True
    auto_register_mcp_tools: bool = True
    event_streaming: bool = True


class MCPStrandBridge:
    """Bridge between MCP and Strand frameworks"""
    
    def __init__(self, config: MCPStrandConfig = None):
        self.config = config or MCPStrandConfig()
        self.mcp_server: Optional[MCPServer] = None
        self.mcp_client: Optional[MCPClient] = None
        self.strand_agents: Dict[str, StrandsAgent] = {}
        self.mcp_tools_registry: Dict[str, MCPTool] = {}
        self.event_publisher = create_event_publisher()
        
        logger.info("MCP-Strand bridge initialized")
    
    def set_mcp_server(self, server: MCPServer):
        """Set MCP server for the bridge"""
        self.mcp_server = server
        if self.config.auto_register_mcp_tools:
            self._register_mcp_tools_as_strand_tools()
    
    def set_mcp_client(self, client: MCPClient):
        """Set MCP client for the bridge"""
        self.mcp_client = client
    
    def register_strand_agent(self, agent: StrandsAgent):
        """Register a Strand agent with the bridge"""
        self.strand_agents[agent.agent_id] = agent
        logger.info(f"Registered Strand agent: {agent.agent_id}")
        
        # Add MCP tools to the agent if enabled
        if self.config.enable_tool_bridging and STRAND_AVAILABLE:
            self._add_mcp_tools_to_agent(agent)
    
    def _register_mcp_tools_as_strand_tools(self):
        """Register MCP tools as Strand tools"""
        if not self.mcp_server or not STRAND_AVAILABLE:
            return
        
        for tool_name, mcp_tool in self.mcp_server.tools.items():
            self.mcp_tools_registry[tool_name] = mcp_tool
            logger.debug(f"Registered MCP tool for Strand use: {tool_name}")
    
    def _add_mcp_tools_to_agent(self, agent: StrandsAgent):
        """Add MCP tools to a Strand agent"""
        if not STRAND_AVAILABLE:
            return
        
        # Create Strand tool wrappers for MCP tools
        for tool_name, mcp_tool in self.mcp_tools_registry.items():
            strand_tool = self._create_strand_tool_wrapper(mcp_tool, agent)
            
            # Add to agent's tools (if the agent supports dynamic tool addition)
            if hasattr(agent, 'add_tool'):
                agent.add_tool(strand_tool)
            elif hasattr(agent.agent, 'tools'):
                agent.agent.tools.append(strand_tool)
    
    def _create_strand_tool_wrapper(self, mcp_tool: MCPTool, agent: StrandsAgent):
        """Create a Strand tool wrapper for an MCP tool"""
        if not STRAND_AVAILABLE:
            return None
        
        @tool
        def mcp_tool_wrapper(**kwargs) -> str:
            """Dynamically created Strand tool wrapper for MCP tool"""
            try:
                # Get auth context from agent if available
                auth_context = getattr(agent, 'auth_context', None)
                
                # Execute MCP tool
                result = asyncio.run(self._execute_mcp_tool_async(
                    mcp_tool.name, kwargs, auth_context
                ))
                
                # Cache result if enabled
                if self.config.cache_tool_results:
                    cache_key = f"mcp_tool:{mcp_tool.name}:{hash(str(kwargs))}"
                    mcp_cache.set(cache_key, result, ttl=300)
                
                # Publish event
                if self.config.event_streaming:
                    asyncio.run(self.event_publisher.publish_tool_execution(
                        mcp_tool.name, kwargs, result.to_dict(), result.is_success
                    ))
                
                return result.content if result.is_success else f"Error: {result.content}"
                
            except Exception as e:
                logger.error(f"Error executing MCP tool {mcp_tool.name}: {e}")
                return f"Error executing {mcp_tool.name}: {str(e)}"
        
        # Set tool metadata
        mcp_tool_wrapper.__name__ = f"mcp_{mcp_tool.name}"
        mcp_tool_wrapper.__doc__ = mcp_tool.description
        
        return mcp_tool_wrapper
    
    async def _execute_mcp_tool_async(self, tool_name: str, arguments: Dict[str, Any],
                                     auth_context: AuthContext = None) -> ToolResult:
        """Execute MCP tool asynchronously"""
        if not self.mcp_server:
            return ToolResult.error_result("MCP server not available")
        
        # Get tool from server
        mcp_tool = self.mcp_server.tools.get(tool_name)
        if not mcp_tool:
            return ToolResult.error_result(f"MCP tool '{tool_name}' not found")
        
        # Execute tool
        start_time = mcp_metrics.tool_execution_start(
            tool_name, auth_context.user_id if auth_context else None
        )
        
        try:
            result = await mcp_tool.execute(arguments)
            mcp_metrics.tool_execution_end(tool_name, start_time, result.is_success)
            return result
        except Exception as e:
            mcp_metrics.tool_execution_end(tool_name, start_time, False)
            return ToolResult.error_result(str(e))
    
    async def orchestrate_workflow(self, workflow_config: Dict[str, Any],
                                  auth_context: AuthContext = None) -> Dict[str, Any]:
        """Orchestrate a workflow using both MCP and Strand capabilities"""
        workflow_id = workflow_config.get('workflow_id', 'unknown')
        steps = workflow_config.get('steps', [])
        
        logger.info(f"Starting MCP-Strand workflow: {workflow_id}")
        
        results = {
            'workflow_id': workflow_id,
            'steps': [],
            'success': True,
            'error': None
        }
        
        try:
            for i, step in enumerate(steps):
                step_type = step.get('type')
                step_config = step.get('config', {})
                
                logger.info(f"Executing workflow step {i+1}: {step_type}")
                
                if step_type == 'mcp_tool':
                    step_result = await self._execute_mcp_workflow_step(step_config, auth_context)
                elif step_type == 'strand_agent':
                    step_result = await self._execute_strand_workflow_step(step_config, auth_context)
                elif step_type == 'data_pipeline':
                    step_result = await self._execute_data_pipeline_step(step_config, auth_context)
                else:
                    step_result = {
                        'success': False,
                        'error': f"Unknown step type: {step_type}"
                    }
                
                results['steps'].append({
                    'step_number': i + 1,
                    'step_type': step_type,
                    'result': step_result
                })
                
                # Stop on first failure if not configured to continue
                if not step_result.get('success', False) and not step_config.get('continue_on_error', False):
                    results['success'] = False
                    results['error'] = step_result.get('error', 'Step failed')
                    break
            
            # Publish workflow completion event
            if self.config.event_streaming:
                await self.event_publisher.publish_system_status(
                    'workflow_orchestrator',
                    'completed' if results['success'] else 'failed',
                    {'workflow_id': workflow_id, 'steps_completed': len(results['steps'])}
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results
    
    async def _execute_mcp_workflow_step(self, config: Dict[str, Any],
                                        auth_context: AuthContext = None) -> Dict[str, Any]:
        """Execute an MCP tool workflow step"""
        tool_name = config.get('tool_name')
        arguments = config.get('arguments', {})
        
        if not tool_name:
            return {'success': False, 'error': 'Missing tool_name in MCP step config'}
        
        try:
            result = await self._execute_mcp_tool_async(tool_name, arguments, auth_context)
            return {
                'success': result.is_success,
                'data': result.to_dict(),
                'error': None if result.is_success else result.content
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _execute_strand_workflow_step(self, config: Dict[str, Any],
                                           auth_context: AuthContext = None) -> Dict[str, Any]:
        """Execute a Strand agent workflow step"""
        agent_id = config.get('agent_id')
        request = config.get('request')
        
        if not agent_id or not request:
            return {'success': False, 'error': 'Missing agent_id or request in Strand step config'}
        
        agent = self.strand_agents.get(agent_id)
        if not agent:
            return {'success': False, 'error': f'Strand agent {agent_id} not found'}
        
        try:
            if STRAND_AVAILABLE:
                response = await agent.process_request(request)
                return {
                    'success': True,
                    'data': {'response': response},
                    'error': None
                }
            else:
                return {'success': False, 'error': 'Strand framework not available'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _execute_data_pipeline_step(self, config: Dict[str, Any],
                                         auth_context: AuthContext = None) -> Dict[str, Any]:
        """Execute a data pipeline step combining MCP resources and Strand agents"""
        pipeline_type = config.get('pipeline_type')
        
        if pipeline_type == 'historical_data_load':
            return await self._execute_historical_data_pipeline(config, auth_context)
        elif pipeline_type == 'market_analysis':
            return await self._execute_market_analysis_pipeline(config, auth_context)
        elif pipeline_type == 'portfolio_sync':
            return await self._execute_portfolio_sync_pipeline(config, auth_context)
        else:
            return {'success': False, 'error': f'Unknown pipeline type: {pipeline_type}'}
    
    async def _execute_historical_data_pipeline(self, config: Dict[str, Any],
                                               auth_context: AuthContext = None) -> Dict[str, Any]:
        """Execute historical data loading pipeline"""
        try:
            # Step 1: Use MCP tool to load historical data
            symbols = config.get('symbols', ['BTC', 'ETH'])
            start_date = config.get('start_date', '2024-01-01')
            
            data_result = await self._execute_mcp_tool_async(
                'load_historical_data',
                {'symbols': symbols, 'start_date': start_date},
                auth_context
            )
            
            if not data_result.is_success:
                return {'success': False, 'error': f'Data loading failed: {data_result.content}'}
            
            # Step 2: Use Strand agent for analysis if available
            analysis_agent_id = config.get('analysis_agent_id')
            if analysis_agent_id and analysis_agent_id in self.strand_agents:
                agent = self.strand_agents[analysis_agent_id]
                analysis_request = f"Analyze historical data for {symbols} from {start_date}"
                
                if STRAND_AVAILABLE:
                    analysis_response = await agent.process_request(analysis_request)
                    
                    return {
                        'success': True,
                        'data': {
                            'historical_data': data_result.to_dict(),
                            'analysis': analysis_response
                        },
                        'error': None
                    }
            
            return {
                'success': True,
                'data': {'historical_data': data_result.to_dict()},
                'error': None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _execute_market_analysis_pipeline(self, config: Dict[str, Any],
                                               auth_context: AuthContext = None) -> Dict[str, Any]:
        """Execute market analysis pipeline"""
        try:
            # Step 1: Get market data via MCP
            symbol = config.get('symbol', 'BTC')
            market_data_result = await self._execute_mcp_tool_async(
                'get_market_data',
                {'symbol': symbol},
                auth_context
            )
            
            if not market_data_result.is_success:
                return {'success': False, 'error': f'Market data failed: {market_data_result.content}'}
            
            # Step 2: Technical analysis via MCP plugin
            tech_analysis_result = await self._execute_mcp_tool_async(
                'technical_analysis',
                {'symbol': symbol, 'timeframe': '1h'},
                auth_context
            )
            
            # Step 3: Sentiment analysis via Strand agent
            sentiment_agent_id = config.get('sentiment_agent_id')
            sentiment_analysis = None
            
            if sentiment_agent_id and sentiment_agent_id in self.strand_agents and STRAND_AVAILABLE:
                agent = self.strand_agents[sentiment_agent_id]
                sentiment_request = f"Analyze market sentiment for {symbol}"
                sentiment_analysis = await agent.process_request(sentiment_request)
            
            return {
                'success': True,
                'data': {
                    'market_data': market_data_result.to_dict(),
                    'technical_analysis': tech_analysis_result.to_dict() if tech_analysis_result.is_success else None,
                    'sentiment_analysis': sentiment_analysis
                },
                'error': None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _execute_portfolio_sync_pipeline(self, config: Dict[str, Any],
                                              auth_context: AuthContext = None) -> Dict[str, Any]:
        """Execute portfolio synchronization pipeline"""
        try:
            # Step 1: Get portfolio data via MCP
            portfolio_result = await self._execute_mcp_tool_async(
                'get_portfolio',
                {'user_id': auth_context.user_id if auth_context else 'default'},
                auth_context
            )
            
            if not portfolio_result.is_success:
                return {'success': False, 'error': f'Portfolio sync failed: {portfolio_result.content}'}
            
            # Step 2: Update portfolio via Strand agent if configured
            portfolio_agent_id = config.get('portfolio_agent_id')
            if portfolio_agent_id and portfolio_agent_id in self.strand_agents and STRAND_AVAILABLE:
                agent = self.strand_agents[portfolio_agent_id]
                update_request = f"Update portfolio with latest data: {portfolio_result.content}"
                update_response = await agent.process_request(update_request)
                
                return {
                    'success': True,
                    'data': {
                        'portfolio_data': portfolio_result.to_dict(),
                        'update_response': update_response
                    },
                    'error': None
                }
            
            return {
                'success': True,
                'data': {'portfolio_data': portfolio_result.to_dict()},
                'error': None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get MCP-Strand integration statistics"""
        return {
            'mcp_server_connected': self.mcp_server is not None,
            'mcp_client_connected': self.mcp_client is not None,
            'registered_strand_agents': len(self.strand_agents),
            'mcp_tools_available': len(self.mcp_tools_registry),
            'config': {
                'tool_bridging': self.config.enable_tool_bridging,
                'resource_sharing': self.config.enable_resource_sharing,
                'workflow_orchestration': self.config.enable_workflow_orchestration,
                'event_streaming': self.config.event_streaming
            },
            'strand_framework_available': STRAND_AVAILABLE
        }


# Global bridge instance
global_mcp_strand_bridge = MCPStrandBridge()


def get_mcp_strand_bridge() -> MCPStrandBridge:
    """Get global MCP-Strand bridge"""
    return global_mcp_strand_bridge


def setup_mcp_strand_integration(mcp_server: MCPServer = None,
                                mcp_client: MCPClient = None,
                                config: MCPStrandConfig = None) -> MCPStrandBridge:
    """Setup MCP-Strand integration"""
    bridge = get_mcp_strand_bridge()
    
    if config:
        bridge.config = config
    
    if mcp_server:
        bridge.set_mcp_server(mcp_server)
    
    if mcp_client:
        bridge.set_mcp_client(mcp_client)
    
    logger.info("MCP-Strand integration setup completed")
    return bridge


def register_strand_agent_with_mcp(agent: StrandsAgent) -> bool:
    """Register a Strand agent with MCP bridge"""
    try:
        bridge = get_mcp_strand_bridge()
        bridge.register_strand_agent(agent)
        return True
    except Exception as e:
        logger.error(f"Failed to register Strand agent with MCP: {e}")
        return False


async def execute_mcp_strand_workflow(workflow_config: Dict[str, Any],
                                     auth_context: AuthContext = None) -> Dict[str, Any]:
    """Execute a workflow using MCP-Strand integration"""
    bridge = get_mcp_strand_bridge()
    return await bridge.orchestrate_workflow(workflow_config, auth_context)
