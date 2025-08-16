"""
Test Agent Manager - Registration, MCP Segregation, and A2A Skill Card Compliance
"""

import asyncio
import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cryptotrading.core.agents.specialized.agent_manager import (
    AgentManagerAgent,
    AgentRegistrationRequest,
    ComplianceStatus,
    create_agent_manager
)
from cryptotrading.infrastructure.registry.enhanced_a2a_registry import (
    get_enhanced_a2a_registry
)

class TestAgentManager:
    """Test Agent Manager functionality"""
    
    @pytest.fixture
    async def agent_manager(self):
        """Create agent manager for testing"""
        with patch('cryptotrading.core.agents.specialized.agent_manager.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
            mock_db.return_value.get_session.return_value.__exit__.return_value = None
            
            manager = AgentManagerAgent()
            await manager.initialize()
            return manager, mock_session
    
    @pytest.mark.asyncio
    async def test_agent_registration_success(self, agent_manager):
        """Test successful agent registration"""
        manager, mock_session = agent_manager
        
        # Mock database query to return no existing agent
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        registration_request = {
            'type': 'register_agent',
            'payload': {
                'agent_id': 'test-agent-001',
                'agent_type': 'historical_loader',
                'capabilities': ['data_loading', 'historical_data', 'technical_indicators'],
                'mcp_tools': ['get_market_data', 'get_historical_prices'],
                'skill_cards': ['historical_data_loading'],
                'metadata': {'test': True}
            }
        }
        
        result = await manager.process_message(registration_request)
        
        assert result['success'] == True
        assert result['agent_id'] == 'test-agent-001'
        assert result['compliance_status'] == 'compliant'
        assert 'get_market_data' in result['assigned_mcp_tools']
        assert 'historical_data_loading' in result['skill_cards']
    
    @pytest.mark.asyncio
    async def test_agent_registration_mcp_conflict(self, agent_manager):
        """Test agent registration with MCP tool conflict"""
        manager, mock_session = agent_manager
        
        # Register first agent
        first_request = AgentRegistrationRequest(
            agent_id='first-agent-001',
            agent_type='historical_loader',
            capabilities=['data_loading'],
            mcp_tools=['get_market_data']
        )
        manager.registered_agents['first-agent-001'] = first_request
        
        # Try to register second agent with same MCP tool
        conflict_request = {
            'type': 'register_agent',
            'payload': {
                'agent_id': 'second-agent-001',
                'agent_type': 'database_manager',
                'capabilities': ['data_storage'],
                'mcp_tools': ['get_market_data'],  # Conflict!
                'skill_cards': []
            }
        }
        
        result = await manager.process_message(conflict_request)
        
        assert result['success'] == False
        assert 'MCP segregation violation' in result['error']
        assert 'already assigned' in result['details'][0]
    
    @pytest.mark.asyncio
    async def test_skill_card_compliance_validation(self, agent_manager):
        """Test skill card compliance validation"""
        manager, mock_session = agent_manager
        
        # Test agent missing required capabilities for skill card
        invalid_request = {
            'type': 'register_agent',
            'payload': {
                'agent_id': 'invalid-agent-001',
                'agent_type': 'historical_loader',
                'capabilities': ['basic_capability'],  # Missing required capabilities
                'mcp_tools': ['get_market_data'],
                'skill_cards': ['historical_data_loading']  # Requires specific capabilities
            }
        }
        
        result = await manager.process_message(invalid_request)
        
        assert result['success'] == False
        assert 'Skill card compliance violation' in result['error']
        assert any('requires capabilities' in violation for violation in result['details'])
    
    @pytest.mark.asyncio
    async def test_compliance_check(self, agent_manager):
        """Test compliance checking"""
        manager, mock_session = agent_manager
        
        # Register a compliant agent first
        request = AgentRegistrationRequest(
            agent_id='compliant-agent-001',
            agent_type='historical_loader',
            capabilities=['data_loading', 'historical_data', 'technical_indicators'],
            mcp_tools=['get_market_data'],
            skill_cards=['historical_data_loading']
        )
        manager.registered_agents['compliant-agent-001'] = request
        
        # Add compliance report
        from cryptotrading.core.agents.specialized.agent_manager import ComplianceReport
        manager.compliance_reports['compliant-agent-001'] = ComplianceReport(
            agent_id='compliant-agent-001',
            status=ComplianceStatus.COMPLIANT,
            skill_card_compliance={'historical_data_loading': True},
            mcp_segregation_status=True,
            registration_status=True,
            violations=[],
            recommendations=[]
        )
        
        compliance_request = {
            'type': 'check_compliance',
            'payload': {
                'agent_id': 'compliant-agent-001'
            }
        }
        
        result = await manager.process_message(compliance_request)
        
        assert result['success'] == True
        assert result['compliance_report']['status'] == 'compliant'
        assert result['compliance_report']['mcp_segregation_status'] == True
    
    @pytest.mark.asyncio
    async def test_agent_discovery(self, agent_manager):
        """Test agent discovery functionality"""
        manager, mock_session = agent_manager
        
        # Register multiple agents
        agents = [
            AgentRegistrationRequest(
                agent_id='loader-001',
                agent_type='historical_loader',
                capabilities=['data_loading'],
                mcp_tools=['get_market_data']
            ),
            AgentRegistrationRequest(
                agent_id='database-001',
                agent_type='database_manager',
                capabilities=['data_storage'],
                mcp_tools=['get_portfolio']
            )
        ]
        
        for agent in agents:
            manager.registered_agents[agent.agent_id] = agent
            manager.compliance_reports[agent.agent_id] = ComplianceReport(
                agent_id=agent.agent_id,
                status=ComplianceStatus.COMPLIANT,
                skill_card_compliance={},
                mcp_segregation_status=True,
                registration_status=True,
                violations=[],
                recommendations=[]
            )
        
        # Test discovery without filters
        discovery_request = {
            'type': 'discover_agents',
            'payload': {}
        }
        
        result = await manager.process_message(discovery_request)
        
        assert result['success'] == True
        assert result['total_count'] == 2
        assert len(result['agents']) == 2
        
        # Test discovery with filters
        filtered_request = {
            'type': 'discover_agents',
            'payload': {
                'filters': {
                    'agent_type': 'historical_loader'
                }
            }
        }
        
        result = await manager.process_message(filtered_request)
        
        assert result['success'] == True
        assert result['total_count'] == 1
        assert result['agents'][0]['agent_id'] == 'loader-001'
    
    @pytest.mark.asyncio
    async def test_segregation_audit(self, agent_manager):
        """Test MCP segregation audit"""
        manager, mock_session = agent_manager
        
        # Register agents with proper segregation
        good_agents = [
            AgentRegistrationRequest(
                agent_id='agent-001',
                agent_type='historical_loader',
                capabilities=['data_loading'],
                mcp_tools=['tool_1']
            ),
            AgentRegistrationRequest(
                agent_id='agent-002',
                agent_type='database_manager',
                capabilities=['data_storage'],
                mcp_tools=['tool_2']
            )
        ]
        
        for agent in good_agents:
            manager.registered_agents[agent.agent_id] = agent
        
        audit_request = {
            'type': 'enforce_segregation',
            'payload': {
                'action': 'audit_segregation'
            }
        }
        
        result = await manager.process_message(audit_request)
        
        assert result['success'] == True
        assert result['audit_results']['segregation_healthy'] == True
        assert result['audit_results']['total_agents'] == 2
        assert len(result['audit_results']['violations']) == 0

@pytest.mark.asyncio
async def test_enhanced_registry_integration():
    """Test integration with enhanced A2A registry"""
    with patch('cryptotrading.infrastructure.registry.enhanced_a2a_registry.get_db') as mock_db:
        mock_session = Mock()
        mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
        mock_db.return_value.get_session.return_value.__exit__.return_value = None
        
        # Mock database queries
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_session.query.return_value.filter.return_value.all.return_value = []
        mock_session.query.return_value.count.return_value = 0
        
        registry = get_enhanced_a2a_registry()
        
        # Test agent discovery
        result = await registry.discover_agents()
        assert result['success'] == True
        
        # Test boundary validation
        validation = await registry.validate_agent_boundaries()
        assert validation['success'] == True
        
        # Test statistics
        stats = await registry.get_registry_statistics()
        assert 'total_agents' in stats

if __name__ == "__main__":
    # Run basic test
    async def run_basic_test():
        print("🧪 Testing Agent Manager...")
        
        # Create mock agent manager
        with patch('cryptotrading.core.agents.specialized.agent_manager.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
            mock_db.return_value.get_session.return_value.__exit__.return_value = None
            mock_session.query.return_value.filter.return_value.first.return_value = None
            
            manager = AgentManagerAgent()
            
            # Test registration
            registration_request = {
                'type': 'register_agent',
                'payload': {
                    'agent_id': 'test-historical-loader-001',
                    'agent_type': 'historical_loader',
                    'capabilities': ['data_loading', 'historical_data', 'technical_indicators'],
                    'mcp_tools': ['get_market_data', 'get_historical_prices'],
                    'skill_cards': ['historical_data_loading']
                }
            }
            
            result = await manager.process_message(registration_request)
            
            if result['success']:
                print("✅ Agent registration successful")
                print(f"   Agent ID: {result['agent_id']}")
                print(f"   Compliance: {result['compliance_status']}")
                print(f"   MCP Tools: {result['assigned_mcp_tools']}")
            else:
                print(f"❌ Registration failed: {result['error']}")
            
            # Test discovery
            discovery_request = {
                'type': 'discover_agents',
                'payload': {}
            }
            
            discovery_result = await manager.process_message(discovery_request)
            print(f"✅ Agent discovery: {discovery_result['total_count']} agents found")
            
            print("\n🎉 Agent Manager tests completed!")
    
    asyncio.run(run_basic_test())
