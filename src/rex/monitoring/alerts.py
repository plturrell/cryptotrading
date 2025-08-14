"""
Production error alerting system
Supports multiple notification channels and intelligent alert routing
"""

import asyncio
import json
import logging
import smtplib
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import hashlib
import os

from ..logging.production_logger import get_logger

logger = get_logger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SMS = "sms"

@dataclass
class Alert:
    """Alert message structure"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    component: str
    timestamp: str
    source: str
    details: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    resolved: bool = False
    resolved_at: Optional[str] = None

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: Callable
    severity: AlertSeverity
    channels: List[AlertChannel]
    component: str
    cooldown_minutes: int = 15
    max_alerts_per_hour: int = 10
    enabled: bool = True

class AlertManager:
    """Production alert management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.cooldowns: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, List[datetime]] = {}
        
        # Notification handlers
        self.notification_handlers = {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SLACK: self._send_slack_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.PAGERDUTY: self._send_pagerduty_alert,
            AlertChannel.SMS: self._send_sms_alert
        }
        
        # Setup default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules for common scenarios"""
        
        # Database connection failures
        self.register_rule(AlertRule(
            name="database_connection_failure",
            condition=lambda context: context.get('error_type') == 'database_connection',
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
            component="database",
            cooldown_minutes=5
        ))
        
        # High error rate
        self.register_rule(AlertRule(
            name="high_error_rate",
            condition=lambda context: context.get('error_rate', 0) > 0.1,  # 10% error rate
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            component="application",
            cooldown_minutes=10
        ))
        
        # Memory usage critical
        self.register_rule(AlertRule(
            name="memory_usage_critical",
            condition=lambda context: context.get('memory_percent', 0) > 90,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            component="system",
            cooldown_minutes=15
        ))
        
        # Disk space critical
        self.register_rule(AlertRule(
            name="disk_space_critical",
            condition=lambda context: context.get('disk_percent', 0) > 95,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            component="system",
            cooldown_minutes=30
        ))
        
        # Authentication failures
        self.register_rule(AlertRule(
            name="auth_failure_spike",
            condition=lambda context: context.get('auth_failures_per_minute', 0) > 50,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            component="security",
            cooldown_minutes=5
        ))
        
        # Agent failures
        self.register_rule(AlertRule(
            name="agent_failure",
            condition=lambda context: context.get('component') == 'agent' and context.get('success') == False,
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.EMAIL],
            component="agent",
            cooldown_minutes=10
        ))
    
    def register_rule(self, rule: AlertRule):
        """Register an alert rule"""
        self.rules[rule.name] = rule
        self.alert_counts[rule.name] = []
        logger.info(f"Registered alert rule: {rule.name}")
    
    async def process_event(self, event: Dict[str, Any]):
        """Process an event and trigger alerts if rules match"""
        triggered_rules = []
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check if rule condition is met
                if rule.condition(event):
                    # Check cooldown
                    if self._is_in_cooldown(rule_name):
                        continue
                    
                    # Check rate limiting
                    if self._is_rate_limited(rule_name):
                        logger.warning(f"Alert rule {rule_name} rate limited")
                        continue
                    
                    # Create alert
                    alert = await self._create_alert(rule, event)
                    
                    # Send notifications
                    await self._send_alert(alert, rule.channels)
                    
                    # Update cooldown and rate limiting
                    self.cooldowns[rule_name] = datetime.utcnow()
                    self.alert_counts[rule_name].append(datetime.utcnow())
                    
                    triggered_rules.append(rule_name)
                    
            except Exception as e:
                logger.error(f"Error processing alert rule {rule_name}: {e}")
        
        return triggered_rules
    
    async def _create_alert(self, rule: AlertRule, event: Dict[str, Any]) -> Alert:
        """Create alert from rule and event"""
        alert_id = self._generate_alert_id(rule, event)
        
        alert = Alert(
            id=alert_id,
            title=self._generate_alert_title(rule, event),
            message=self._generate_alert_message(rule, event),
            severity=rule.severity,
            component=rule.component,
            timestamp=datetime.utcnow().isoformat(),
            source=event.get('source', 'unknown'),
            details=event,
            tags=event.get('tags', [])
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Keep history size manageable
        if len(self.alert_history) > 10000:
            self.alert_history = self.alert_history[-5000:]
        
        logger.info(f"Created alert: {alert.title}",
                   alert_id=alert_id,
                   severity=alert.severity.value,
                   component=alert.component)
        
        return alert
    
    def _generate_alert_id(self, rule: AlertRule, event: Dict[str, Any]) -> str:
        """Generate unique alert ID"""
        content = f"{rule.name}_{rule.component}_{event.get('source', '')}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_alert_title(self, rule: AlertRule, event: Dict[str, Any]) -> str:
        """Generate alert title"""
        component = rule.component.title()
        severity = rule.severity.value.upper()
        
        if rule.name == "database_connection_failure":
            return f"[{severity}] Database Connection Failure"
        elif rule.name == "high_error_rate":
            rate = event.get('error_rate', 0) * 100
            return f"[{severity}] High Error Rate: {rate:.1f}%"
        elif rule.name == "memory_usage_critical":
            usage = event.get('memory_percent', 0)
            return f"[{severity}] Critical Memory Usage: {usage:.1f}%"
        elif rule.name == "disk_space_critical":
            usage = event.get('disk_percent', 0)
            return f"[{severity}] Critical Disk Usage: {usage:.1f}%"
        elif rule.name == "auth_failure_spike":
            failures = event.get('auth_failures_per_minute', 0)
            return f"[{severity}] Authentication Failure Spike: {failures}/min"
        elif rule.name == "agent_failure":
            agent_id = event.get('agent_id', 'unknown')
            return f"[{severity}] Agent Failure: {agent_id}"
        else:
            return f"[{severity}] {component} Alert: {rule.name}"
    
    def _generate_alert_message(self, rule: AlertRule, event: Dict[str, Any]) -> str:
        """Generate detailed alert message"""
        base_message = f"Alert triggered for rule: {rule.name}\n"
        base_message += f"Component: {rule.component}\n"
        base_message += f"Severity: {rule.severity.value}\n"
        base_message += f"Timestamp: {datetime.utcnow().isoformat()}\n\n"
        
        # Add event-specific details
        if rule.name == "database_connection_failure":
            error = event.get('error', 'Unknown error')
            base_message += f"Database connection failed: {error}\n"
            base_message += "This may indicate database server issues or network connectivity problems."
        
        elif rule.name == "high_error_rate":
            rate = event.get('error_rate', 0) * 100
            total_requests = event.get('total_requests', 0)
            error_count = event.get('error_count', 0)
            base_message += f"Error rate: {rate:.2f}% ({error_count}/{total_requests} requests)\n"
            base_message += "High error rates may indicate application issues or external service problems."
        
        elif rule.name == "memory_usage_critical":
            usage = event.get('memory_percent', 0)
            used_gb = event.get('memory_used_gb', 0)
            total_gb = event.get('memory_total_gb', 0)
            base_message += f"Memory usage: {usage:.1f}% ({used_gb:.1f}GB / {total_gb:.1f}GB)\n"
            base_message += "High memory usage may lead to application performance issues or crashes."
        
        elif rule.name == "disk_space_critical":
            usage = event.get('disk_percent', 0)
            free_gb = event.get('disk_free_gb', 0)
            base_message += f"Disk usage: {usage:.1f}% ({free_gb:.1f}GB free)\n"
            base_message += "Low disk space may prevent logging, database operations, or file uploads."
        
        elif rule.name == "auth_failure_spike":
            failures = event.get('auth_failures_per_minute', 0)
            base_message += f"Authentication failures: {failures} per minute\n"
            base_message += "This may indicate a brute force attack or authentication service issues."
        
        elif rule.name == "agent_failure":
            agent_id = event.get('agent_id', 'unknown')
            error = event.get('error', 'Unknown error')
            operation = event.get('operation', 'unknown')
            base_message += f"Agent: {agent_id}\n"
            base_message += f"Operation: {operation}\n"
            base_message += f"Error: {error}\n"
        
        # Add common troubleshooting information
        base_message += f"\n--- Event Details ---\n"
        for key, value in event.items():
            if key not in ['details', 'traceback']:
                base_message += f"{key}: {value}\n"
        
        return base_message
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if rule is in cooldown period"""
        if rule_name not in self.cooldowns:
            return False
        
        rule = self.rules[rule_name]
        last_alert = self.cooldowns[rule_name]
        cooldown_end = last_alert + timedelta(minutes=rule.cooldown_minutes)
        
        return datetime.utcnow() < cooldown_end
    
    def _is_rate_limited(self, rule_name: str) -> bool:
        """Check if rule has exceeded rate limit"""
        rule = self.rules[rule_name]
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old entries
        self.alert_counts[rule_name] = [
            timestamp for timestamp in self.alert_counts[rule_name]
            if timestamp > hour_ago
        ]
        
        return len(self.alert_counts[rule_name]) >= rule.max_alerts_per_hour
    
    async def _send_alert(self, alert: Alert, channels: List[AlertChannel]):
        """Send alert to specified channels"""
        tasks = []
        
        for channel in channels:
            if channel in self.notification_handlers:
                task = asyncio.create_task(
                    self.notification_handlers[channel](alert)
                )
                tasks.append(task)
        
        # Send all notifications concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        for channel, result in zip(channels, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to send alert to {channel.value}: {result}")
            else:
                logger.info(f"Alert sent to {channel.value}", alert_id=alert.id)
    
    async def _send_email_alert(self, alert: Alert):
        """Send email notification"""
        smtp_host = self.config.get('smtp_host', 'localhost')
        smtp_port = self.config.get('smtp_port', 587)
        smtp_username = self.config.get('smtp_username')
        smtp_password = self.config.get('smtp_password')
        from_email = self.config.get('from_email', 'alerts@reks.com')
        to_emails = self.config.get('alert_emails', ['admin@reks.com'])
        
        try:
            msg = MimeMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = alert.title
            
            # Create HTML body
            html_body = f"""
            <html>
            <body>
                <h2>{alert.title}</h2>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Component:</strong> {alert.component}</p>
                <p><strong>Timestamp:</strong> {alert.timestamp}</p>
                <hr>
                <pre>{alert.message}</pre>
                {f'<hr><h3>Details:</h3><pre>{json.dumps(alert.details, indent=2)}</pre>' if alert.details else ''}
            </body>
            </html>
            """
            
            msg.attach(MimeText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if smtp_username and smtp_password:
                    server.starttls()
                    server.login(smtp_username, smtp_password)
                
                server.send_message(msg)
            
        except Exception as e:
            raise Exception(f"Email sending failed: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack notification"""
        webhook_url = self.config.get('slack_webhook_url')
        if not webhook_url:
            raise Exception("Slack webhook URL not configured")
        
        # Choose color based on severity
        color_map = {
            AlertSeverity.LOW: "#36a64f",      # Green
            AlertSeverity.MEDIUM: "#ff9900",   # Orange
            AlertSeverity.HIGH: "#ff6600",     # Red-Orange
            AlertSeverity.CRITICAL: "#ff0000"  # Red
        }
        
        slack_message = {
            "text": f"Alert: {alert.title}",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#cccccc"),
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Component",
                            "value": alert.component,
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp,
                            "short": True
                        }
                    ],
                    "text": alert.message[:500] + "..." if len(alert.message) > 500 else alert.message
                }
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=slack_message) as response:
                    if response.status != 200:
                        raise Exception(f"Slack API returned {response.status}")
        
        except Exception as e:
            raise Exception(f"Slack sending failed: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook notification"""
        webhook_url = self.config.get('webhook_url')
        if not webhook_url:
            raise Exception("Webhook URL not configured")
        
        payload = asdict(alert)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status not in [200, 201, 202]:
                        raise Exception(f"Webhook returned {response.status}")
        
        except Exception as e:
            raise Exception(f"Webhook sending failed: {e}")
    
    async def _send_pagerduty_alert(self, alert: Alert):
        """Send PagerDuty notification"""
        integration_key = self.config.get('pagerduty_integration_key')
        if not integration_key:
            raise Exception("PagerDuty integration key not configured")
        
        pagerduty_payload = {
            "routing_key": integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.title,
                "severity": alert.severity.value,
                "source": alert.source,
                "component": alert.component,
                "custom_details": alert.details
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=pagerduty_payload
                ) as response:
                    if response.status != 202:
                        raise Exception(f"PagerDuty API returned {response.status}")
        
        except Exception as e:
            raise Exception(f"PagerDuty sending failed: {e}")
    
    async def _send_sms_alert(self, alert: Alert):
        """Send SMS notification via Twilio"""
        try:
            # Get Twilio configuration
            account_sid = self.config.get('twilio_account_sid')
            auth_token = self.config.get('twilio_auth_token')
            from_number = self.config.get('twilio_from_number')
            to_numbers = self.config.get('alert_phone_numbers', [])
            
            if not all([account_sid, auth_token, from_number]):
                logger.warning("Twilio SMS configuration missing. Skipping SMS alert.")
                return
            
            if not to_numbers:
                logger.warning("No phone numbers configured for SMS alerts")
                return
            
            # Import Twilio client
            try:
                from twilio.rest import Client
            except ImportError:
                logger.error("Twilio library not installed. Install with: pip install twilio")
                # Fall back to HTTP API call
                await self._send_sms_via_http_api(alert)
                return
            
            # Initialize Twilio client
            client = Client(account_sid, auth_token)
            
            # Format SMS message
            message_body = f"🚨 ALERT: {alert.title}\n"
            message_body += f"Severity: {alert.severity.value.upper()}\n"
            message_body += f"Component: {alert.component}\n"
            message_body += f"Time: {alert.timestamp}\n"
            
            if len(alert.message) <= 100:
                message_body += f"Details: {alert.message}"
            else:
                message_body += f"Details: {alert.message[:97]}..."
            
            # Keep message under 160 characters for standard SMS
            if len(message_body) > 160:
                message_body = message_body[:157] + "..."
            
            # Send to all configured numbers
            sent_count = 0
            for phone_number in to_numbers:
                try:
                    message = client.messages.create(
                        body=message_body,
                        from_=from_number,
                        to=phone_number
                    )
                    
                    logger.info(f"SMS alert sent to {phone_number}", 
                               alert_id=alert.id,
                               message_sid=message.sid)
                    sent_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to send SMS to {phone_number}: {e}")
            
            if sent_count > 0:
                logger.info(f"SMS alert sent successfully to {sent_count} recipients")
            else:
                raise Exception("Failed to send SMS to any recipients")
                
        except Exception as e:
            logger.error(f"SMS alert failed: {e}")
            raise
    
    async def _send_sms_via_http_api(self, alert: Alert):
        """Fallback SMS sending via Twilio HTTP API"""
        import aiohttp
        import base64
        
        account_sid = self.config.get('twilio_account_sid')
        auth_token = self.config.get('twilio_auth_token')
        from_number = self.config.get('twilio_from_number')
        to_numbers = self.config.get('alert_phone_numbers', [])
        
        if not all([account_sid, auth_token, from_number, to_numbers]):
            logger.warning("Twilio configuration incomplete for HTTP API fallback")
            return
        
        # Create basic auth header
        credentials = f"{account_sid}:{auth_token}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        # Format message
        message_body = f"🚨 {alert.title} - {alert.severity.value.upper()} - {alert.component}"
        
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        
        sent_count = 0
        async with aiohttp.ClientSession() as session:
            for phone_number in to_numbers:
                try:
                    data = {
                        'From': from_number,
                        'To': phone_number,
                        'Body': message_body
                    }
                    
                    async with session.post(url, headers=headers, data=data) as response:
                        if response.status == 201:
                            result = await response.json()
                            logger.info(f"SMS sent via HTTP API to {phone_number}",
                                       message_sid=result.get('sid'))
                            sent_count += 1
                        else:
                            error_text = await response.text()
                            logger.error(f"HTTP API SMS failed for {phone_number}: {error_text}")
                
                except Exception as e:
                    logger.error(f"HTTP API SMS error for {phone_number}: {e}")
        
        if sent_count == 0:
            raise Exception("HTTP API SMS fallback failed for all recipients")
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """Mark alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow().isoformat()
            
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert.title}",
                       alert_id=alert_id,
                       resolved_by=resolved_by)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting statistics"""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert.timestamp) > last_24h
        ]
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                alert for alert in recent_alerts
                if alert.severity == severity
            ])
        
        return {
            "active_alerts": len(self.active_alerts),
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "alerts_last_24h": len(recent_alerts),
            "severity_distribution": severity_counts,
            "top_components": self._get_top_components(recent_alerts)
        }
    
    def _get_top_components(self, alerts: List[Alert]) -> Dict[str, int]:
        """Get top components by alert count"""
        component_counts = {}
        for alert in alerts:
            component_counts[alert.component] = component_counts.get(alert.component, 0) + 1
        
        # Sort by count and return top 5
        sorted_components = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_components[:5])

# Global alert manager instance
alert_manager = AlertManager()

async def send_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.MEDIUM,
    component: str = "application",
    details: Dict[str, Any] = None
):
    """Convenience function to send custom alerts"""
    event = {
        'title': title,
        'message': message,
        'severity': severity.value,
        'component': component,
        'source': 'manual',
        'details': details or {}
    }
    
    await alert_manager.process_event(event)