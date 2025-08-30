"""
Real Alert System with Slack, Email, and Webhook integrations
No mock implementations - actual notification delivery
"""

import asyncio
import json
import logging
import os
import smtplib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class AlertChannel:
    """Configuration for an alert channel"""

    name: str
    type: str  # slack, email, webhook, sms
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[str] = None  # None means all severities


@dataclass
class Alert:
    """Alert to be sent"""

    id: str
    timestamp: datetime
    title: str
    message: str
    severity: str
    category: str
    source: str  # error_tracker, anomaly_detector, etc.
    context: Dict[str, Any]
    channels: List[str] = None  # Specific channels, None = all applicable


class RealAlertSystem:
    """Real alert system with actual integrations"""

    def __init__(self, db_path: str = "cryptotrading.db"):
        self.db_path = db_path
        self.channels: Dict[str, AlertChannel] = {}
        self.alert_history = []
        self.rate_limits = {}  # Channel -> last alert time
        self.rate_limit_window = timedelta(minutes=5)  # Min time between alerts
        self._init_database()
        self._setup_default_channels()

    def _init_database(self):
        """Initialize database for alert tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_history (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                title TEXT,
                message TEXT,
                severity TEXT,
                category TEXT,
                source TEXT,
                channels_sent TEXT,
                delivery_status TEXT,
                context TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_channels (
                name TEXT PRIMARY KEY,
                type TEXT,
                config TEXT,
                enabled BOOLEAN DEFAULT TRUE,
                severity_filter TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used DATETIME
            )
        """
        )

        conn.commit()
        conn.close()

    def _setup_default_channels(self):
        """Setup default alert channels from environment"""
        # Slack integration
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if slack_webhook:
            self.add_channel(
                AlertChannel(
                    name="slack_alerts",
                    type="slack",
                    config={
                        "webhook_url": slack_webhook,
                        "channel": "#alerts",
                        "username": "CryptoTrading Alert Bot",
                    },
                    severity_filter=["HIGH", "CRITICAL"],
                )
            )

        # Email integration
        smtp_host = os.getenv("SMTP_HOST")
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASSWORD")
        alert_email = os.getenv("ALERT_EMAIL")

        if all([smtp_host, smtp_user, smtp_pass, alert_email]):
            self.add_channel(
                AlertChannel(
                    name="email_alerts",
                    type="email",
                    config={
                        "smtp_host": smtp_host,
                        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                        "username": smtp_user,
                        "password": smtp_pass,
                        "from_email": smtp_user,
                        "to_emails": [alert_email],
                        "use_tls": True,
                    },
                    severity_filter=["CRITICAL"],
                )
            )

        # Generic webhook
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        if webhook_url:
            self.add_channel(
                AlertChannel(
                    name="webhook_alerts",
                    type="webhook",
                    config={
                        "url": webhook_url,
                        "method": "POST",
                        "headers": {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {os.getenv('WEBHOOK_TOKEN', '')}",
                        },
                    },
                )
            )

        # PagerDuty integration
        pagerduty_key = os.getenv("PAGERDUTY_INTEGRATION_KEY")
        if pagerduty_key:
            self.add_channel(
                AlertChannel(
                    name="pagerduty",
                    type="pagerduty",
                    config={"integration_key": pagerduty_key, "service": "CryptoTrading Platform"},
                    severity_filter=["CRITICAL"],
                )
            )

    def add_channel(self, channel: AlertChannel):
        """Add an alert channel"""
        self.channels[channel.name] = channel

        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO alert_channels
            (name, type, config, enabled, severity_filter)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                channel.name,
                channel.type,
                json.dumps(channel.config),
                channel.enabled,
                json.dumps(channel.severity_filter or []),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Added alert channel: {channel.name} ({channel.type})")

    async def send_alert(self, alert: Alert) -> Dict[str, Any]:
        """Send alert to configured channels"""
        results = {
            "alert_id": alert.id,
            "timestamp": alert.timestamp.isoformat(),
            "channels_attempted": [],
            "channels_success": [],
            "channels_failed": [],
            "total_sent": 0,
        }

        # Determine which channels to use
        target_channels = []
        for channel_name, channel in self.channels.items():
            if not channel.enabled:
                continue

            # Check if specific channels requested
            if alert.channels and channel_name not in alert.channels:
                continue

            # Check severity filter
            if channel.severity_filter and alert.severity not in channel.severity_filter:
                continue

            # Check rate limiting
            if self._is_rate_limited(channel_name):
                logger.info(f"Skipping channel {channel_name} due to rate limiting")
                continue

            target_channels.append(channel)

        # Send to each channel
        for channel in target_channels:
            results["channels_attempted"].append(channel.name)

            try:
                success = await self._send_to_channel(alert, channel)
                if success:
                    results["channels_success"].append(channel.name)
                    results["total_sent"] += 1
                    self.rate_limits[channel.name] = datetime.now()
                else:
                    results["channels_failed"].append(channel.name)

            except Exception as e:
                logger.error(f"Failed to send alert to {channel.name}: {e}")
                results["channels_failed"].append(channel.name)

        # Store alert in history
        self._store_alert_history(alert, results)

        return results

    async def _send_to_channel(self, alert: Alert, channel: AlertChannel) -> bool:
        """Send alert to a specific channel"""
        try:
            if channel.type == "slack":
                return await self._send_slack_alert(alert, channel)
            elif channel.type == "email":
                return await self._send_email_alert(alert, channel)
            elif channel.type == "webhook":
                return await self._send_webhook_alert(alert, channel)
            elif channel.type == "pagerduty":
                return await self._send_pagerduty_alert(alert, channel)
            else:
                logger.warning(f"Unknown channel type: {channel.type}")
                return False

        except Exception as e:
            logger.error(f"Error sending to {channel.name}: {e}")
            return False

    async def _send_slack_alert(self, alert: Alert, channel: AlertChannel) -> bool:
        """Send alert to Slack"""
        webhook_url = channel.config["webhook_url"]

        # Create Slack message with rich formatting
        color_map = {"LOW": "good", "MEDIUM": "warning", "HIGH": "danger", "CRITICAL": "danger"}

        emoji_map = {
            "LOW": ":information_source:",
            "MEDIUM": ":warning:",
            "HIGH": ":exclamation:",
            "CRITICAL": ":rotating_light:",
        }

        slack_payload = {
            "channel": channel.config.get("channel", "#alerts"),
            "username": channel.config.get("username", "Alert Bot"),
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "warning"),
                    "title": f"{emoji_map.get(alert.severity, '')} {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity, "short": True},
                        {"title": "Category", "value": alert.category, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True,
                        },
                    ],
                    "footer": "CryptoTrading Platform",
                    "ts": int(alert.timestamp.timestamp()),
                }
            ],
        }

        # Add context if available
        if alert.context:
            context_text = "\n".join([f"*{k}:* {v}" for k, v in alert.context.items()[:5]])
            slack_payload["attachments"][0]["fields"].append(
                {"title": "Details", "value": context_text, "short": False}
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=slack_payload) as response:
                if response.status == 200:
                    logger.info(f"Slack alert sent successfully to {channel.name}")
                    return True
                else:
                    logger.error(f"Slack alert failed: {response.status}")
                    return False

    async def _send_email_alert(self, alert: Alert, channel: AlertChannel) -> bool:
        """Send alert via email"""
        config = channel.config

        # Create email message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.severity}] {alert.title}"
        msg["From"] = config["from_email"]
        msg["To"] = ", ".join(config["to_emails"])

        # Plain text version
        text_content = f"""
CRYPTOTRADING PLATFORM ALERT

Severity: {alert.severity}
Category: {alert.category}  
Source: {alert.source}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Message:
{alert.message}

Context:
{json.dumps(alert.context, indent=2) if alert.context else 'None'}

Alert ID: {alert.id}
"""

        # HTML version
        html_content = f"""
<html>
<head></head>
<body>
    <h2 style="color: {'red' if alert.severity in ['HIGH', 'CRITICAL'] else 'orange'};">
        CryptoTrading Platform Alert
    </h2>
    
    <table style="border-collapse: collapse; width: 100%;">
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Severity:</strong></td>
            <td style="border: 1px solid #ddd; padding: 8px; color: {'red' if alert.severity in ['HIGH', 'CRITICAL'] else 'orange'};">
                {alert.severity}
            </td></tr>
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Category:</strong></td>
            <td style="border: 1px solid #ddd; padding: 8px;">{alert.category}</td></tr>
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Source:</strong></td>
            <td style="border: 1px solid #ddd; padding: 8px;">{alert.source}</td></tr>
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Time:</strong></td>
            <td style="border: 1px solid #ddd; padding: 8px;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</td></tr>
    </table>
    
    <h3>Message:</h3>
    <p>{alert.message}</p>
    
    <h3>Context:</h3>
    <pre>{json.dumps(alert.context, indent=2) if alert.context else 'None'}</pre>
    
    <hr>
    <small>Alert ID: {alert.id}</small>
</body>
</html>
"""

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        # Send email
        try:
            with smtplib.SMTP(config["smtp_host"], config["smtp_port"]) as server:
                if config.get("use_tls", True):
                    server.starttls()
                server.login(config["username"], config["password"])
                server.send_message(msg)

            logger.info(f"Email alert sent successfully to {channel.name}")
            return True

        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            return False

    async def _send_webhook_alert(self, alert: Alert, channel: AlertChannel) -> bool:
        """Send alert to generic webhook"""
        config = channel.config

        payload = {
            "alert_id": alert.id,
            "timestamp": alert.timestamp.isoformat(),
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity,
            "category": alert.category,
            "source": alert.source,
            "context": alert.context,
        }

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=config.get("method", "POST"),
                url=config["url"],
                json=payload,
                headers=config.get("headers", {}),
            ) as response:
                if 200 <= response.status < 300:
                    logger.info(f"Webhook alert sent successfully to {channel.name}")
                    return True
                else:
                    logger.error(f"Webhook alert failed: {response.status}")
                    return False

    async def _send_pagerduty_alert(self, alert: Alert, channel: AlertChannel) -> bool:
        """Send alert to PagerDuty"""
        config = channel.config

        # PagerDuty Events API v2
        url = "https://events.pagerduty.com/v2/enqueue"

        payload = {
            "routing_key": config["integration_key"],
            "event_action": "trigger",
            "dedup_key": f"cryptotrading_{alert.category}_{alert.id}",
            "payload": {
                "summary": alert.title,
                "source": alert.source,
                "severity": alert.severity.lower(),
                "component": alert.category,
                "group": "cryptotrading",
                "class": alert.category,
                "custom_details": {
                    "message": alert.message,
                    "alert_id": alert.id,
                    "context": alert.context,
                },
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 202:  # PagerDuty returns 202 for success
                    logger.info(f"PagerDuty alert sent successfully to {channel.name}")
                    return True
                else:
                    logger.error(f"PagerDuty alert failed: {response.status}")
                    return False

    def _is_rate_limited(self, channel_name: str) -> bool:
        """Check if channel is rate limited"""
        if channel_name not in self.rate_limits:
            return False

        last_alert = self.rate_limits[channel_name]
        return datetime.now() - last_alert < self.rate_limit_window

    def _store_alert_history(self, alert: Alert, results: Dict):
        """Store alert in history database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO alert_history
            (id, title, message, severity, category, source, channels_sent, 
             delivery_status, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                alert.id,
                alert.title,
                alert.message,
                alert.severity,
                alert.category,
                alert.source,
                json.dumps(results["channels_success"]),
                json.dumps(results),
                json.dumps(alert.context),
            ),
        )

        conn.commit()
        conn.close()

    def get_alert_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        # Total alerts
        cursor.execute("SELECT COUNT(*) FROM alert_history WHERE timestamp > ?", (cutoff,))
        total_alerts = cursor.fetchone()[0]

        # By severity
        cursor.execute(
            """
            SELECT severity, COUNT(*) 
            FROM alert_history 
            WHERE timestamp > ?
            GROUP BY severity
        """,
            (cutoff,),
        )
        by_severity = dict(cursor.fetchall())

        # By source
        cursor.execute(
            """
            SELECT source, COUNT(*)
            FROM alert_history
            WHERE timestamp > ?
            GROUP BY source
        """,
            (cutoff,),
        )
        by_source = dict(cursor.fetchall())

        # Channel performance
        cursor.execute(
            """
            SELECT channels_sent, COUNT(*)
            FROM alert_history
            WHERE timestamp > ?
            GROUP BY channels_sent
        """,
            (cutoff,),
        )
        channel_usage = {}
        for channels_json, count in cursor.fetchall():
            channels = json.loads(channels_json or "[]")
            for channel in channels:
                channel_usage[channel] = channel_usage.get(channel, 0) + count

        conn.close()

        return {
            "total_alerts": total_alerts,
            "by_severity": by_severity,
            "by_source": by_source,
            "channel_usage": channel_usage,
            "active_channels": len([c for c in self.channels.values() if c.enabled]),
            "period_hours": hours,
        }


# Global instance
alert_system = RealAlertSystem()


async def send_alert(
    title: str,
    message: str,
    severity: str = "MEDIUM",
    category: str = "system",
    source: str = "unknown",
    context: Dict = None,
) -> Dict[str, Any]:
    """Convenience function to send an alert"""
    alert = Alert(
        id=f"alert_{datetime.now().timestamp()}",
        timestamp=datetime.now(),
        title=title,
        message=message,
        severity=severity,
        category=category,
        source=source,
        context=context or {},
    )

    return await alert_system.send_alert(alert)


def setup_environment_channels():
    """Setup channels from environment variables"""
    # This is called automatically during initialization
    # But can be called again to reload config
    alert_system._setup_default_channels()
