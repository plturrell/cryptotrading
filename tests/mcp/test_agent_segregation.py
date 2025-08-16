"""
Comprehensive Test Suite for MCP Agent Segregation

Tests multi-tenant isolation, authentication, authorization, resource quotas,
context isolation, and audit logging for MCP tools.
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import os

# Import segregation components
from src.cryptotrading.infrastructure.analysis.mcp_agent_segregation import (
    AgentContext,
    AgentRole,
    ResourceType,
    MCPAgentSegregationManager,
    SecureToolWrapper,
    get_segregation_manager
)
from src.cryptotrading.infrastructure.analysis.mcp_auth_middleware import (
    MCPAuthenticationMiddleware,
    AuthenticationRequest,
    ContextIsolationManager,
    get_auth_middleware
)
from src.cryptotrading.infrastructure.analysis.mcp_monitoring_audit import (
    MCPAuditLogger,
    MCPPerformanceMonitor,
    SecurityEvent,
    get_audit_logger
)
from src.cryptotrading.infrastructure.analysis.segregated_mcp_tools import (
    CLRSAnalysisTool,
    DependencyGraphTool,
    CodeSimilarityTool,
    create_segregated_tools
)

class TestMCPAgentSegregation:
    """Test agent segregation and multi-tenancy"""
    
    @pytest.fixture
    def segregation_manager(self):
        """Create test segregation manager"""
        return MCPAgentSegregationManager()
    
    @pytest.fixture
    def test_agents(self, segregation_manager):
        """Create test agents for different tenants"""
        agents = {}
        
        # Tenant A agents
        agents['admin_a'] = segregation_manager.create_agent_context(
            agent_id="agent_admin_tenant_a",
            tenant_id="tenant_a",
            role=AgentRole.ADMIN
        )
        
        agents['analyst_a'] = segregation_manager.create_agent_context(
            agent_id="agent_analyst_tenant_a", 
            tenant_id="tenant_a",
            role=AgentRole.ANALYST
        )
        
        # Tenant B agents
        agents['admin_b'] = segregation_manager.create_agent_context(
            agent_id="agent_admin_tenant_b",
            tenant_id="tenant_b", 
            role=AgentRole.ADMIN
        )
        
        agents['basic_b'] = segregation_manager.create_agent_context(
            agent_id="agent_basic_tenant_b",
            tenant_id="tenant_b",
            role=AgentRole.BASIC_USER
        )
        
        return agents
    
    def test_agent_context_creation(self, segregation_manager):
        """Test agent context creation and validation"""
        agent_context = segregation_manager.create_agent_context(
            agent_id="test_agent_001",
            tenant_id="test_tenant_001", 
            role=AgentRole.ANALYST
        )
        
        assert agent_context.agent_id == "test_agent_001"
        assert agent_context.tenant_id == "test_tenant_001"
        assert agent_context.role == AgentRole.ANALYST
        assert ResourceType.CODE_ANALYSIS in agent_context.permissions
        assert agent_context.resource_quotas["requests_per_hour"] == 1000
    
    def test_permission_checking(self, segregation_manager, test_agents):
        """Test permission checking for different roles"""
        # Admin should have all permissions
        assert segregation_manager.check_permission(test_agents['admin_a'], ResourceType.CLRS_ALGORITHMS)
        assert segregation_manager.check_permission(test_agents['admin_a'], ResourceType.CONFIGURATION)
        
        # Analyst should have analysis permissions but not configuration
        assert segregation_manager.check_permission(test_agents['analyst_a'], ResourceType.CODE_ANALYSIS)
        assert not segregation_manager.check_permission(test_agents['analyst_a'], ResourceType.CONFIGURATION)
        
        # Basic user should have limited permissions
        assert segregation_manager.check_permission(test_agents['basic_b'], ResourceType.CODE_ANALYSIS)
        assert not segregation_manager.check_permission(test_agents['basic_b'], ResourceType.CLRS_ALGORITHMS)
    
    def test_tenant_isolation(self, segregation_manager, test_agents):
        """Test that agents cannot access other tenants' resources"""
        # Create tenant-specific data
        tenant_a_data = {"tenant": "tenant_a", "secret": "tenant_a_secret"}
        tenant_b_data = {"tenant": "tenant_b", "secret": "tenant_b_secret"}
        
        # Tenant A agent should only access tenant A data
        assert segregation_manager.is_tenant_isolated(test_agents['admin_a'], tenant_a_data)
        assert not segregation_manager.is_tenant_isolated(test_agents['admin_a'], tenant_b_data)
        
        # Tenant B agent should only access tenant B data
        assert segregation_manager.is_tenant_isolated(test_agents['admin_b'], tenant_b_data)
        assert not segregation_manager.is_tenant_isolated(test_agents['admin_b'], tenant_a_data)
    
    def test_resource_quota_enforcement(self, segregation_manager, test_agents):
        """Test resource quota enforcement"""
        agent = test_agents['basic_b']
        
        # Should be able to consume resources within quota
        assert segregation_manager.check_resource_quota(agent, "requests_per_hour")
        segregation_manager.consume_resource(agent, "requests_per_hour")
        
        # Exhaust quota
        for _ in range(agent.resource_quotas["requests_per_hour"] - 1):
            segregation_manager.consume_resource(agent, "requests_per_hour")
        
        # Should now be at quota limit
        assert not segregation_manager.check_resource_quota(agent, "requests_per_hour")
    
    def test_session_token_management(self, segregation_manager, test_agents):
        """Test session token generation and validation"""
        agent = test_agents['analyst_a']
        
        # Generate session token
        token = segregation_manager.generate_session_token(agent)
        assert token is not None
        assert len(token) > 0
        
        # Validate token
        validated_agent = segregation_manager.validate_session_token(token)
        assert validated_agent is not None
        assert validated_agent.agent_id == agent.agent_id
        assert validated_agent.tenant_id == agent.tenant_id

class TestMCPAuthentication:
    """Test MCP authentication middleware"""
    
    @pytest.fixture
    def auth_middleware(self):
        """Create test authentication middleware"""
        return MCPAuthenticationMiddleware()
    
    @pytest.mark.asyncio
    async def test_agent_authentication(self, auth_middleware):
        """Test agent authentication flow"""
        auth_request = AuthenticationRequest(
            agent_id="agent_test_001",
            tenant_id="tenant_test_001"
        )
        
        response = await auth_middleware.authenticate_agent(auth_request)
        
        assert response.success
        assert response.agent_context is not None
        assert response.session_token is not None
        assert response.agent_context.agent_id == "agent_test_001"
        assert response.agent_context.tenant_id == "tenant_test_001"
    
    @pytest.mark.asyncio
    async def test_session_token_validation(self, auth_middleware):
        """Test session token validation"""
        # First authenticate to get token
        auth_request = AuthenticationRequest(
            agent_id="agent_test_002",
            tenant_id="tenant_test_002"
        )
        
        response = await auth_middleware.authenticate_agent(auth_request)
        assert response.success
        
        # Use token for subsequent authentication
        token_request = AuthenticationRequest(
            agent_id="agent_test_002",
            tenant_id="tenant_test_002",
            session_token=response.session_token
        )
        
        token_response = await auth_middleware.authenticate_agent(token_request)
        assert token_response.success
        assert token_response.agent_context.agent_id == "agent_test_002"
    
    @pytest.mark.asyncio
    async def test_cross_tenant_authentication_denial(self, auth_middleware):
        """Test that cross-tenant authentication is denied"""
        # Authenticate with one tenant
        auth_request = AuthenticationRequest(
            agent_id="agent_test_003",
            tenant_id="tenant_test_003"
        )
        
        response = await auth_middleware.authenticate_agent(auth_request)
        assert response.success
        
        # Try to use token with different tenant
        cross_tenant_request = AuthenticationRequest(
            agent_id="agent_test_003",
            tenant_id="tenant_different",
            session_token=response.session_token
        )
        
        cross_response = await auth_middleware.authenticate_agent(cross_tenant_request)
        assert not cross_response.success
        assert cross_response.error_code == "TENANT_MISMATCH"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, auth_middleware):
        """Test authentication rate limiting"""
        # Make multiple failed attempts
        for i in range(6):  # Exceed the 5 attempt limit
            auth_request = AuthenticationRequest(
                agent_id="agent_invalid",
                tenant_id="tenant_invalid"
            )
            
            response = await auth_middleware.authenticate_agent(auth_request)
            
            if i < 5:
                assert response.error_code == "AUTHENTICATION_FAILED"
            else:
                assert response.error_code == "RATE_LIMITED"

class TestContextIsolation:
    """Test context isolation for agent execution"""
    
    @pytest.fixture
    def isolation_manager(self):
        """Create test context isolation manager"""
        return ContextIsolationManager()
    
    @pytest.fixture
    def test_agent_context(self):
        """Create test agent context"""
        segregation_manager = get_segregation_manager()
        return segregation_manager.create_agent_context(
            agent_id="agent_isolation_test",
            tenant_id="tenant_isolation_test",
            role=AgentRole.ANALYST
        )
    
    @pytest.mark.asyncio
    async def test_isolated_context_creation(self, isolation_manager, test_agent_context):
        """Test creation of isolated execution context"""
        context_id = await isolation_manager.create_isolated_execution_context(
            test_agent_context, "test_tool"
        )
        
        assert context_id is not None
        assert "tenant_isolation_test" in context_id
        assert "agent_isolation_test" in context_id
        assert "test_tool" in context_id
        
        # Verify context exists
        assert context_id in isolation_manager.active_contexts
        
        context = isolation_manager.active_contexts[context_id]
        assert context["tenant_id"] == "tenant_isolation_test"
        assert context["agent_id"] == "agent_isolation_test"
        assert context["tool_name"] == "test_tool"
    
    @pytest.mark.asyncio
    async def test_isolated_execution(self, isolation_manager, test_agent_context):
        """Test execution within isolated context"""
        context_id = await isolation_manager.create_isolated_execution_context(
            test_agent_context, "test_tool"
        )
        
        async def test_operation(value: int) -> int:
            # Verify environment isolation
            assert os.environ.get("TENANT_ID") == "tenant_isolation_test"
            assert os.environ.get("AGENT_ID") == "agent_isolation_test"
            return value * 2
        
        result = await isolation_manager.execute_in_context(context_id, test_operation, 21)
        assert result == 42
        
        # Clean up
        await isolation_manager.cleanup_context(context_id)

class TestSegregatedTools:
    """Test segregated MCP tools"""
    
    @pytest.fixture
    def segregated_tools(self):
        """Create segregated tools"""
        return create_segregated_tools()
    
    @pytest.fixture
    def test_agent_contexts(self):
        """Create test agent contexts"""
        segregation_manager = get_segregation_manager()
        return {
            'admin': segregation_manager.create_agent_context(
                agent_id="agent_admin_tools_test",
                tenant_id="tenant_tools_test",
                role=AgentRole.ADMIN
            ),
            'analyst': segregation_manager.create_agent_context(
                agent_id="agent_analyst_tools_test", 
                tenant_id="tenant_tools_test",
                role=AgentRole.ANALYST
            ),
            'basic': segregation_manager.create_agent_context(
                agent_id="agent_basic_tools_test",
                tenant_id="tenant_tools_test", 
                role=AgentRole.BASIC_USER
            )
        }
    
    @pytest.mark.asyncio
    async def test_clrs_analysis_tool_authorization(self, segregated_tools, test_agent_contexts):
        """Test CLRS analysis tool with different authorization levels"""
        clrs_tool = segregated_tools["clrs_analysis"]
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test_function():\n    return 42\n")
            test_file = f.name
        
        try:
            # Admin should have access
            admin_result = await clrs_tool.execute({
                "file_path": test_file,
                "algorithm": "complexity",
                "tenant_id": "tenant_tools_test"
            }, test_agent_contexts['admin'])
            
            assert admin_result["success"]
            assert admin_result["agent_id"] == "agent_admin_tools_test"
            assert admin_result["tenant_id"] == "tenant_tools_test"
            
            # Analyst should have access
            analyst_result = await clrs_tool.execute({
                "file_path": test_file,
                "algorithm": "complexity", 
                "tenant_id": "tenant_tools_test"
            }, test_agent_contexts['analyst'])
            
            assert analyst_result["success"]
            
            # Basic user should be denied
            basic_result = await clrs_tool.execute({
                "file_path": test_file,
                "algorithm": "complexity",
                "tenant_id": "tenant_tools_test"
            }, test_agent_contexts['basic'])
            
            assert not basic_result.get("success", True)
            assert basic_result.get("code") == "PERMISSION_DENIED"
            
        finally:
            os.unlink(test_file)
    
    @pytest.mark.asyncio
    async def test_cross_tenant_access_denial(self, segregated_tools, test_agent_contexts):
        """Test that tools deny cross-tenant access"""
        clrs_tool = segregated_tools["clrs_analysis"]
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test_function():\n    return 42\n")
            test_file = f.name
        
        try:
            # Try to access with different tenant_id
            result = await clrs_tool.execute({
                "file_path": test_file,
                "algorithm": "complexity",
                "tenant_id": "different_tenant"
            }, test_agent_contexts['admin'])
            
            assert not result.get("success", True)
            assert result.get("code") == "CROSS_TENANT_ACCESS"
            
        finally:
            os.unlink(test_file)
    
    @pytest.mark.asyncio
    async def test_file_size_quota_enforcement(self, segregated_tools, test_agent_contexts):
        """Test file size quota enforcement"""
        clrs_tool = segregated_tools["clrs_analysis"]
        
        # Create large test file (exceeds quota)
        large_content = "# Large file\n" + "x = 1\n" * 100000  # Large file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            large_file = f.name
        
        try:
            result = await clrs_tool.execute({
                "file_path": large_file,
                "algorithm": "complexity",
                "tenant_id": "tenant_tools_test"
            }, test_agent_contexts['basic'])  # Basic user has lower quotas
            
            assert not result.get("success", True)
            assert result.get("code") == "FILE_TOO_LARGE"
            
        finally:
            os.unlink(large_file)

class TestAuditingAndMonitoring:
    """Test audit logging and monitoring"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def audit_logger(self, temp_db_path):
        """Create test audit logger"""
        return MCPAuditLogger(temp_db_path)
    
    @pytest.fixture
    def performance_monitor(self, audit_logger):
        """Create test performance monitor"""
        return MCPPerformanceMonitor(audit_logger)
    
    @pytest.mark.asyncio
    async def test_access_logging(self, audit_logger):
        """Test access attempt logging"""
        from src.cryptotrading.infrastructure.analysis.mcp_agent_segregation import AccessLog
        
        access_log = AccessLog(
            agent_id="agent_audit_test",
            tenant_id="tenant_audit_test",
            resource_type=ResourceType.CODE_ANALYSIS,
            action="EXECUTE_TOOL",
            success=True,
            reason="AUTHORIZED",
            timestamp=datetime.utcnow()
        )
        
        await audit_logger.log_access(access_log)
        await audit_logger._flush_queue()
        
        # Verify log was stored
        summary = await audit_logger.get_tenant_access_summary("tenant_audit_test", 1)
        assert summary["statistics"]["total_requests"] == 1
        assert summary["statistics"]["successful_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_security_event_logging(self, audit_logger):
        """Test security event logging"""
        security_event = SecurityEvent(
            event_id="test_security_001",
            event_type="UNAUTHORIZED_ACCESS",
            severity="HIGH",
            agent_id="agent_security_test",
            tenant_id="tenant_security_test",
            resource_type=ResourceType.CONFIGURATION,
            description="Attempted unauthorized configuration access",
            metadata={"attempted_action": "modify_config"},
            timestamp=datetime.utcnow()
        )
        
        await audit_logger.log_security_event(security_event)
        await audit_logger._flush_queue()
        
        # Verify event was stored
        events = await audit_logger.get_security_events("tenant_security_test", "HIGH", 1)
        assert len(events) == 1
        assert events[0]["event_type"] == "UNAUTHORIZED_ACCESS"
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, performance_monitor):
        """Test performance metric recording"""
        segregation_manager = get_segregation_manager()
        agent_context = segregation_manager.create_agent_context(
            agent_id="agent_perf_test",
            tenant_id="tenant_perf_test",
            role=AgentRole.ANALYST
        )
        
        await performance_monitor.record_operation_metrics(
            agent_context=agent_context,
            tool_name="test_tool",
            execution_time=2.5,  # 2.5 seconds
            memory_used=256.0,   # 256 MB
            success=True
        )
        
        await performance_monitor.audit_logger._flush_queue()
        
        # Verify metrics were recorded
        summary = await performance_monitor.get_tenant_performance_summary("tenant_perf_test", 1)
        assert len(summary["response_times"]) > 0

# Integration test
class TestFullIntegration:
    """Test full integration of agent segregation system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_segregation(self):
        """Test complete end-to-end agent segregation flow"""
        # 1. Authentication
        auth_middleware = get_auth_middleware()
        auth_request = AuthenticationRequest(
            agent_id="agent_integration_test",
            tenant_id="tenant_integration_test"
        )
        
        auth_response = await auth_middleware.authenticate_agent(auth_request)
        assert auth_response.success
        
        # 2. Tool execution with segregation
        segregated_tools = create_segregated_tools()
        clrs_tool = segregated_tools["clrs_analysis"]
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def integration_test():\n    return 'success'\n")
            test_file = f.name
        
        try:
            # 3. Execute tool with proper context
            result = await clrs_tool.execute({
                "file_path": test_file,
                "algorithm": "all",
                "tenant_id": "tenant_integration_test"
            }, auth_response.agent_context)
            
            assert result["success"]
            assert result["tenant_id"] == "tenant_integration_test"
            assert result["agent_id"] == "agent_integration_test"
            
            # 4. Verify audit trail
            audit_logger = get_audit_logger()
            await audit_logger._flush_queue()
            
            summary = await audit_logger.get_tenant_access_summary("tenant_integration_test", 1)
            assert summary["statistics"]["total_requests"] >= 1
            
        finally:
            os.unlink(test_file)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
