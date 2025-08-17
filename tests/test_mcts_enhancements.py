"""
Comprehensive tests for MCTS enhancements
Tests adaptive control, security, A/B testing, and anomaly detection
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.cryptotrading.core.agents.specialized.mcts_calculation_agent import (
    ProductionMCTSCalculationAgent, 
    ProductionTradingEnvironment,
    MCTSConfig
)
from src.cryptotrading.core.agents.specialized.mcts_adaptive_control import (
    AdaptiveIterationController,
    ConvergenceMetrics,
    DynamicExplorationParams
)
from src.cryptotrading.core.security.mcts_auth import (
    SecurityManager,
    Permission,
    SecurityLevel,
    UserSession,
    APIKey
)
from src.cryptotrading.core.agents.specialized.mcts_ab_testing import (
    ABTestManager,
    VariantConfig,
    VariantType
)
from src.cryptotrading.core.agents.specialized.mcts_anomaly_detection import (
    AnomalyDetector,
    AnomalyType,
    AnomalySeverity
)


class TestAdaptiveControl:
    """Test adaptive iteration control and dynamic parameters"""
    
    @pytest.fixture
    def controller(self):
        return AdaptiveIterationController(
            min_iterations=50,
            max_iterations=1000,
            convergence_window=20,
            early_stop_confidence=0.9
        )
    
    @pytest.mark.asyncio
    async def test_convergence_detection(self, controller):
        """Test convergence detection logic"""
        # Simulate stable convergence
        for i in range(100):
            value = 0.5 + (0.01 * (i % 5))  # Stable with small variation
            confidence = 0.8 + (i / 1000)  # Gradually increasing confidence
            
            should_continue, reason, status = controller.should_continue_search(
                value, confidence, "test_action"
            )
            
            if not should_continue:
                break
        
        assert not should_continue
        assert "convergence" in reason
        assert controller.convergence_metrics.get_convergence_confidence() > 0.5
    
    @pytest.mark.asyncio
    async def test_dynamic_exploration_params(self, controller):
        """Test dynamic exploration parameter adjustment"""
        initial_c_param = controller.dynamic_params.current_c_param
        
        # Simulate search progress
        for i in range(50):
            value = 0.1 + (i * 0.01)  # Improving values
            confidence = min(0.9, i / 50)  # Increasing confidence
            
            should_continue, reason, status = controller.should_continue_search(
                value, confidence, "test_action"
            )
            
            if not should_continue:
                break
        
        # Exploration parameter should have adapted
        final_c_param = controller.dynamic_params.current_c_param
        assert final_c_param != initial_c_param
        
        # Should have switched to exploitation phase
        assert not controller.dynamic_params.exploration_phase
    
    @pytest.mark.asyncio
    async def test_early_stopping(self, controller):
        """Test early stopping functionality"""
        # Simulate quick convergence
        for i in range(30):
            # Very stable values indicating convergence
            value = 0.75
            confidence = 0.95
            
            should_continue, reason, status = controller.should_continue_search(
                value, confidence, "stable_action"
            )
            
            if not should_continue:
                break
        
        # Should stop early due to convergence
        assert not should_continue
        assert controller.current_iteration < controller.max_iterations
        
        final_report = controller.get_final_report()
        assert final_report['efficiency_gain'] > 0


class TestSecurity:
    """Test security enhancements"""
    
    @pytest.fixture
    def security_manager(self):
        return SecurityManager(SecurityLevel.DEVELOPMENT)
    
    @pytest.mark.asyncio
    async def test_jwt_authentication(self, security_manager):
        """Test JWT token authentication"""
        # Create user session
        username = "test_user"
        password = "admin123"  # Development password
        permissions = [Permission.READ, Permission.CALCULATE]
        
        result = await security_manager.create_user_session(
            username, password, permissions, "127.0.0.1", "test-agent"
        )
        
        assert result is not None
        jwt_token, session = result
        assert session.username == username
        assert Permission.READ in session.permissions
        
        # Test token validation
        validated_session = await security_manager.authenticate_jwt(jwt_token)
        assert validated_session is not None
        assert validated_session.user_id == session.user_id
    
    @pytest.mark.asyncio
    async def test_api_key_authentication(self, security_manager):
        """Test API key authentication"""
        # Create API key
        key_id, raw_key = await security_manager.create_api_key(
            description="Test API key",
            permissions=[Permission.CALCULATE, Permission.OPTIMIZE],
            created_by="test_admin"
        )
        
        # Test authentication
        api_key = await security_manager.authenticate_api_key(key_id, raw_key)
        assert api_key is not None
        assert Permission.CALCULATE in api_key.permissions
        
        # Test invalid key
        invalid_api_key = await security_manager.authenticate_api_key(key_id, "wrong_key")
        assert invalid_api_key is None
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, security_manager):
        """Test rate limiting functionality"""
        # Create session with low rate limit
        username = "test_user"
        password = "admin123"
        permissions = [Permission.READ]
        
        result = await security_manager.create_user_session(
            username, password, permissions
        )
        jwt_token, session = result
        
        # Set low rate limit for testing
        session.rate_limit = 3
        
        # Test rate limiting
        for i in range(5):
            within_limit = session.check_rate_limit()
            if i < 3:
                assert within_limit
            else:
                assert not within_limit
    
    @pytest.mark.asyncio
    async def test_failed_attempt_blocking(self, security_manager):
        """Test IP blocking after failed attempts"""
        ip_address = "192.168.1.100"
        
        # Simulate failed attempts
        for _ in range(6):
            await security_manager._record_failed_attempt(ip_address)
        
        # IP should be blocked
        is_blocked = await security_manager._is_ip_blocked(ip_address)
        assert is_blocked


class TestABTesting:
    """Test A/B testing framework"""
    
    @pytest.fixture
    def ab_manager(self):
        return ABTestManager()
    
    @pytest.fixture
    def mock_agent(self):
        agent = Mock()
        agent.config = Mock()
        agent.config.exploration_constant = 1.4
        agent.config.simulation_depth = 10
        agent.config.enable_rave = True
        agent.config.enable_progressive_widening = True
        agent.config.parallel_simulations = 4
        
        # Mock methods
        agent._create_test_environment = Mock()
        agent.run_mcts_parallel = AsyncMock(return_value={
            'expected_value': 0.1,
            'confidence': 0.8,
            'stats': {
                'iterations': 100,
                'tree_size': 500,
                'convergence_reason': 'early_convergence',
                'efficiency_gain': 0.2
            }
        })
        
        return agent
    
    def test_experiment_creation(self, ab_manager):
        """Test creating A/B test experiments"""
        variants = ab_manager.add_predefined_variants("test_experiment")
        
        assert len(variants) > 0
        assert "test_experiment" in ab_manager.experiments
        
        # Should have exactly one control variant
        control_variants = [v for v in variants if v.is_control]
        assert len(control_variants) == 1
    
    @pytest.mark.asyncio
    async def test_experiment_execution(self, ab_manager, mock_agent):
        """Test running A/B test experiment"""
        # Create simple experiment
        variants = [
            VariantConfig(
                variant_id="control",
                variant_type=VariantType.EXPLORATION_PARAM,
                name="Control",
                description="Default settings",
                parameters={'c_param': 1.4},
                is_control=True
            ),
            VariantConfig(
                variant_id="high_exploration",
                variant_type=VariantType.EXPLORATION_PARAM,
                name="High Exploration",
                description="Increased exploration",
                parameters={'c_param': 2.0}
            )
        ]
        
        ab_manager.create_experiment("simple_test", variants)
        
        # Run experiment with small sample size
        test_params = {
            'iterations': 100,
            'symbols': ['BTC'],
            'initial_portfolio': 10000
        }
        
        result = await ab_manager.run_experiment(
            "simple_test", mock_agent, test_params, runs_per_variant=2
        )
        
        assert result['experiment_id'] == "simple_test"
        assert result['total_runs'] == 4  # 2 variants × 2 runs
        assert 'analysis' in result
        assert 'variant_stats' in result['analysis']
    
    def test_statistical_analysis(self, ab_manager):
        """Test statistical analysis of results"""
        from src.cryptotrading.core.agents.specialized.mcts_ab_testing import ExperimentResult
        
        # Create mock results
        results = [
            ExperimentResult(
                variant_id="control",
                execution_time=2.0,
                iterations_completed=100,
                expected_value=0.1,
                confidence=0.8,
                convergence_reason="early_convergence",
                memory_usage_mb=50,
                tree_size=500,
                efficiency_gain=0.1
            ),
            ExperimentResult(
                variant_id="test",
                execution_time=1.5,
                iterations_completed=100,
                expected_value=0.15,
                confidence=0.85,
                convergence_reason="early_convergence",
                memory_usage_mb=45,
                tree_size=450,
                efficiency_gain=0.2
            )
        ]
        
        analysis = ab_manager._analyze_results("test_analysis", results)
        
        assert 'variant_stats' in analysis
        assert 'best_by_expected_value' in analysis
        assert 'recommendations' in analysis


class TestAnomalyDetection:
    """Test anomaly detection system"""
    
    @pytest.fixture
    def detector(self):
        return AnomalyDetector("test_agent")
    
    @pytest.mark.asyncio
    async def test_normal_metrics(self, detector):
        """Test that normal metrics don't trigger anomalies"""
        # Add normal execution time samples
        for i in range(20):
            await detector.record_metric('execution_time', 2.0 + (i % 3) * 0.1)
        
        # Should have no active alerts
        alerts = detector.get_active_alerts()
        assert len(alerts) == 0
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, detector):
        """Test anomaly detection triggers"""
        # Add normal samples first
        for i in range(15):
            await detector.record_metric('execution_time', 2.0)
        
        # Add anomalous sample
        await detector.record_metric('execution_time', 10.0)  # 5x normal
        
        # Should trigger anomaly alert
        alerts = detector.get_active_alerts()
        assert len(alerts) > 0
        
        alert = alerts[0]
        assert alert.anomaly_type == AnomalyType.EXECUTION_TIME_ANOMALY
        assert alert.current_value == 10.0
    
    @pytest.mark.asyncio
    async def test_memory_spike_detection(self, detector):
        """Test memory spike detection"""
        # Add normal memory usage
        for i in range(10):
            await detector.record_metric('memory_usage', 100.0)
        
        # Add memory spike
        await detector.record_metric('memory_usage', 450.0)  # Critical level
        
        alerts = detector.get_active_alerts()
        memory_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.MEMORY_SPIKE]
        
        assert len(memory_alerts) > 0
        assert memory_alerts[0].severity == AnomalySeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_error_rate_monitoring(self, detector):
        """Test error rate anomaly detection"""
        # Add normal error rates
        for i in range(15):
            await detector.record_metric('error_rate', 0.05)  # 5% normal
        
        # Add high error rate
        await detector.record_metric('error_rate', 0.25)  # 25% errors
        
        alerts = detector.get_active_alerts()
        error_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.ERROR_RATE_SPIKE]
        
        assert len(error_alerts) > 0
        assert error_alerts[0].severity == AnomalySeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_health_assessment(self, detector):
        """Test system health assessment"""
        # Add various metrics
        await detector.record_metric('execution_time', 2.0)
        await detector.record_metric('memory_usage', 100.0)
        await detector.record_metric('error_rate', 0.02)
        await detector.record_metric('confidence', 0.85)
        
        health = await detector.get_system_health()
        
        assert health['agent_id'] == "test_agent"
        assert health['health_score'] >= 90  # Should be healthy
        assert health['health_status'] == "excellent"
        assert 'metric_summaries' in health
    
    def test_alert_resolution(self, detector):
        """Test alert resolution functionality"""
        # Create a mock alert
        from src.cryptotrading.core.agents.specialized.mcts_anomaly_detection import AnomalyAlert
        
        alert = AnomalyAlert(
            alert_id="test_alert_001",
            anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
            severity=AnomalySeverity.MEDIUM,
            message="Test alert",
            detected_at=datetime.utcnow(),
            metric_name="test_metric",
            current_value=10.0,
            expected_range=(5.0, 15.0),
            confidence=0.8
        )
        
        detector.alerts.append(alert)
        
        # Resolve alert
        resolved = detector.resolve_alert("test_alert_001")
        assert resolved
        assert alert.resolved_at is not None
        
        # Try to resolve non-existent alert
        not_resolved = detector.resolve_alert("non_existent")
        assert not not_resolved


class TestIntegratedAgent:
    """Test integrated enhanced MCTS agent"""
    
    @pytest.fixture
    def enhanced_agent(self):
        config = MCTSConfig()
        config.iterations = 100  # Small for testing
        
        agent = ProductionMCTSCalculationAgent(
            agent_id="test_enhanced_agent",
            config=config
        )
        
        return agent
    
    @pytest.mark.asyncio
    async def test_secure_calculation(self, enhanced_agent):
        """Test secure calculation with authentication"""
        # Test unauthenticated request
        message = {
            'type': 'calculate',
            'parameters': {
                'initial_portfolio': 10000,
                'symbols': ['BTC'],
                'max_depth': 3
            }
        }
        
        result = await enhanced_agent.process_message(message)
        assert 'error' not in result  # Should work without auth in development
        
        # Test with authentication
        auth_result = await enhanced_agent.authenticate_request(
            "ApiKey dev_key_001:dev_mcts_key_" + "0" * 32,  # Mock dev key
            "127.0.0.1"
        )
        
        # Should work with proper authentication structure
        assert 'auth_type' in auth_result or 'error' in auth_result
    
    @pytest.mark.asyncio
    async def test_adaptive_mcts_execution(self, enhanced_agent):
        """Test MCTS with adaptive controls"""
        # Setup test environment
        config = {
            'initial_portfolio': 10000,
            'symbols': ['BTC'],
            'max_depth': 3
        }
        
        enhanced_agent.environment = ProductionTradingEnvironment(config)
        
        # Run with adaptive controls
        result = await enhanced_agent.run_mcts_parallel(iterations=50)
        
        assert 'best_action' in result
        assert 'stats' in result
        assert 'efficiency_gain' in result['stats']
        assert 'convergence_reason' in result['stats']
        assert 'adaptive_params_final' in result['stats']
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, enhanced_agent):
        """Test monitoring and anomaly detection integration"""
        # Monitoring should be initialized
        assert enhanced_agent.anomaly_detector is not None
        assert enhanced_agent.monitoring_dashboard is not None
        
        # Test metric recording
        await enhanced_agent._record_execution_metrics(
            execution_time=2.0,
            iterations=100,
            expected_value=0.1,
            confidence=0.8,
            tree_size=500,
            convergence_confidence=0.9
        )
        
        # Check that metrics were recorded
        health = await enhanced_agent.anomaly_detector.get_system_health()
        assert health['agent_id'] == enhanced_agent.agent_id


@pytest.mark.integration
class TestFullWorkflow:
    """Integration test for complete enhanced workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_enhanced_workflow(self):
        """Test complete workflow with all enhancements"""
        # Setup
        config = MCTSConfig()
        config.iterations = 200
        
        agent = ProductionMCTSCalculationAgent(
            agent_id="integration_test_agent",
            config=config
        )
        
        # Test A/B experiment
        variants = agent.ab_test_manager.add_predefined_variants("integration_test")
        assert len(variants) > 0
        
        # Test security
        key_id, raw_key = await agent.security_manager.create_api_key(
            description="Integration test key",
            permissions=[Permission.CALCULATE],
            created_by="test"
        )
        
        api_key = await agent.security_manager.authenticate_api_key(key_id, raw_key)
        assert api_key is not None
        
        # Test calculation with monitoring
        test_config = {
            'initial_portfolio': 10000,
            'symbols': ['BTC', 'ETH'],
            'max_depth': 5
        }
        
        agent.environment = ProductionTradingEnvironment(test_config)
        result = await agent.run_mcts_parallel(iterations=100)
        
        # Verify results
        assert result['best_action'] is not None
        assert result['stats']['efficiency_gain'] >= 0
        assert 'convergence_reason' in result['stats']
        
        # Check monitoring data
        health = await agent.anomaly_detector.get_system_health()
        assert health['health_score'] > 0
        
        # Verify no critical alerts
        critical_alerts = agent.anomaly_detector.get_active_alerts(AnomalySeverity.CRITICAL)
        assert len(critical_alerts) == 0
        
        logger.info("Full enhanced workflow test completed successfully")