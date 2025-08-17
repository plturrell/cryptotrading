"""
Test unified monitoring abstraction with different backends
"""

import os
import pytest
import json
import time
from unittest.mock import patch, MagicMock

from cryptotrading.core.monitoring import (
    MonitoringFactory,
    get_monitor,
    LightweightMonitoring,
    FullMonitoring,
    NoOpMonitoring,
    MetricType,
    LogLevel
)


class TestMonitoringFactory:
    """Test monitoring factory environment detection"""
    
    def test_vercel_environment(self, monkeypatch):
        """Test Vercel environment uses lightweight monitoring"""
        monkeypatch.setenv('VERCEL', '1')
        MonitoringFactory.clear_cache()
        
        monitor = MonitoringFactory.get_monitor()
        assert isinstance(monitor, LightweightMonitoring)
    
    def test_disabled_monitoring(self, monkeypatch):
        """Test disabled monitoring returns NoOp"""
        monkeypatch.setenv('DISABLE_MONITORING', 'true')
        MonitoringFactory.clear_cache()
        
        monitor = MonitoringFactory.get_monitor()
        assert isinstance(monitor, NoOpMonitoring)
    
    def test_explicit_monitoring_type(self):
        """Test explicit monitoring type selection"""
        monitor = MonitoringFactory.get_monitor(monitoring_type='noop')
        assert isinstance(monitor, NoOpMonitoring)
    
    def test_singleton_behavior(self):
        """Test that factory returns same instance for same parameters"""
        monitor1 = MonitoringFactory.get_monitor('lightweight', 'test-service', 'dev')
        monitor2 = MonitoringFactory.get_monitor('lightweight', 'test-service', 'dev')
        assert monitor1 is monitor2
        
        # Different parameters should return different instances
        monitor3 = MonitoringFactory.get_monitor('lightweight', 'other-service', 'dev')
        assert monitor1 is not monitor3


class TestLightweightMonitoring:
    """Test lightweight monitoring implementation"""
    
    @pytest.fixture
    def monitor(self):
        """Create lightweight monitoring instance"""
        return LightweightMonitoring('test-service', 'test')
    
    def test_logging(self, monitor, caplog):
        """Test logging functionality"""
        monitor.log_info("Test info message", {"extra": "data"})
        monitor.log_warning("Test warning")
        monitor.log_error("Test error")
        
        # Check that logs were captured
        assert len(caplog.records) >= 3
        
        # Check log content
        log_messages = [record.message for record in caplog.records]
        assert any("Test info message" in msg for msg in log_messages)
        assert any("Test warning" in msg for msg in log_messages)
        assert any("Test error" in msg for msg in log_messages)
    
    def test_metrics(self, monitor):
        """Test metric recording"""
        # Counter
        monitor.increment_counter('test.counter')
        monitor.increment_counter('test.counter', 5)
        
        # Gauge
        monitor.set_gauge('test.gauge', 42.5)
        
        # Histogram
        monitor.record_histogram('test.histogram', 100)
        
        # Check internal metrics storage
        assert 'test.counter' in monitor._metrics
        assert 'test.gauge' in monitor._metrics
    
    def test_spans(self, monitor, caplog):
        """Test span creation and timing"""
        with monitor.span('test-operation', {'attr': 'value'}) as span:
            span.set_attribute('additional', 'attribute')
            span.add_event('checkpoint', {'event_data': 'value'})
            time.sleep(0.01)  # Ensure some duration
        
        # Check that span completion was logged
        log_messages = [record.message for record in caplog.records]
        assert any("Span completed: test-operation" in msg for msg in log_messages)
    
    def test_error_recording(self, monitor, caplog):
        """Test error recording with breadcrumbs"""
        # Add some breadcrumbs
        monitor.add_breadcrumb("User logged in", "auth")
        monitor.add_breadcrumb("Navigated to dashboard", "navigation")
        
        # Record an error
        try:
            raise ValueError("Test error")
        except ValueError as e:
            monitor.record_error(e, {"context": "test"})
        
        # Check that error was logged with breadcrumbs
        error_logs = [r for r in caplog.records if r.levelname == 'ERROR']
        assert len(error_logs) > 0
        
        # Parse the JSON log to check breadcrumbs
        for record in error_logs:
            if hasattr(record, 'extra_fields'):
                extra = record.extra_fields
                if 'breadcrumbs' in extra:
                    breadcrumbs = extra['breadcrumbs']
                    assert len(breadcrumbs) == 2
                    assert breadcrumbs[0]['message'] == "User logged in"
                    assert breadcrumbs[1]['message'] == "Navigated to dashboard"
    
    def test_user_context(self, monitor):
        """Test user context setting"""
        monitor.set_user_context(
            user_id="user123",
            email="user@example.com",
            username="testuser",
            extra={"role": "admin"}
        )
        
        # Check that user context is added to global tags
        assert monitor.global_tags['user_id'] == "user123"
        assert monitor.global_tags['user_email'] == "user@example.com"
        assert monitor.global_tags['user_username'] == "testuser"
        assert monitor.global_tags['role'] == "admin"
    
    def test_timed_context_manager(self, monitor, caplog):
        """Test timed context manager"""
        with monitor.timed('operation.duration', {'operation': 'test'}):
            time.sleep(0.01)
        
        # Check that timing metric was recorded
        log_messages = [record.message for record in caplog.records]
        assert any("operation.duration" in msg for msg in log_messages)
    
    def test_flush(self, monitor, caplog):
        """Test flushing metrics"""
        # Record some metrics
        monitor.increment_counter('flush.test', 10)
        monitor.set_gauge('flush.gauge', 5.5)
        
        # Flush
        monitor.flush()
        
        # Check that aggregated metrics were logged
        log_messages = [record.message for record in caplog.records]
        assert any("Aggregated metric: flush.test" in msg for msg in log_messages)
        assert any("Aggregated metric: flush.gauge" in msg for msg in log_messages)
        
        # Check that metrics were cleared
        assert len(monitor._metrics) == 0


class TestNoOpMonitoring:
    """Test no-op monitoring implementation"""
    
    @pytest.fixture
    def monitor(self):
        """Create no-op monitoring instance"""
        return NoOpMonitoring()
    
    def test_all_operations_are_noop(self, monitor):
        """Test that all operations do nothing"""
        # None of these should raise exceptions
        monitor.log_info("test")
        monitor.increment_counter("test")
        monitor.set_gauge("test", 1)
        monitor.record_histogram("test", 1)
        
        with monitor.span("test") as span:
            span.set_attribute("key", "value")
            span.add_event("event")
            span.record_exception(Exception("test"))
        
        monitor.record_error(Exception("test"))
        monitor.add_breadcrumb("test")
        monitor.set_user_context("user1")
        monitor.set_tag("key", "value")
        monitor.flush()
        
        # All operations should complete without error
        assert True


class TestMonitoringInterface:
    """Test monitoring interface contracts"""
    
    def test_metric_types(self):
        """Test metric type enum"""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"
    
    def test_log_levels(self):
        """Test log level enum"""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"
    
    def test_interface_methods(self):
        """Test that all implementations have required methods"""
        for monitor_class in [LightweightMonitoring, NoOpMonitoring]:
            monitor = monitor_class()
            
            # Check all required methods exist
            assert hasattr(monitor, 'record_metric')
            assert hasattr(monitor, 'log')
            assert hasattr(monitor, 'start_span')
            assert hasattr(monitor, 'record_error')
            assert hasattr(monitor, 'add_breadcrumb')
            assert hasattr(monitor, 'set_user_context')
            assert hasattr(monitor, 'set_tag')
            assert hasattr(monitor, 'flush')
            
            # Check convenience methods
            assert hasattr(monitor, 'log_debug')
            assert hasattr(monitor, 'log_info')
            assert hasattr(monitor, 'log_warning')
            assert hasattr(monitor, 'log_error')
            assert hasattr(monitor, 'log_critical')
            assert hasattr(monitor, 'increment_counter')
            assert hasattr(monitor, 'set_gauge')
            assert hasattr(monitor, 'record_histogram')
            assert hasattr(monitor, 'timed')
            assert hasattr(monitor, 'span')


class TestEnvironmentIntegration:
    """Test monitoring integration with environment detection"""
    
    def test_production_vercel(self, monkeypatch):
        """Test production Vercel environment"""
        monkeypatch.setenv('VERCEL', '1')
        monkeypatch.setenv('ENVIRONMENT', 'production')
        MonitoringFactory.clear_cache()
        
        monitor = get_monitor('crypto-prod')
        assert isinstance(monitor, LightweightMonitoring)
        assert monitor.environment == 'production'
        assert monitor.service_name == 'crypto-prod'
    
    def test_local_development(self, monkeypatch):
        """Test local development environment"""
        monkeypatch.delenv('VERCEL', raising=False)
        monkeypatch.setenv('ENVIRONMENT', 'development')
        monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
        MonitoringFactory.clear_cache()
        
        monitor = get_monitor('crypto-dev')
        # Should be FullMonitoring in dev, but falls back to Lightweight if OTEL not available
        assert monitor.service_name == 'crypto-dev'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])