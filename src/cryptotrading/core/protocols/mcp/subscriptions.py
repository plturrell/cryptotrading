"""
MCP Resource Subscriptions Implementation
Implements resource subscriptions for real-time updates
"""

from typing import Dict, Any, List, Optional, Callable, Set, AsyncGenerator
from dataclasses import dataclass, asdict
import asyncio
import logging
from datetime import datetime, timedelta
import weakref
import json
from enum import Enum

logger = logging.getLogger(__name__)


class SubscriptionState(Enum):
    """Subscription states"""
    ACTIVE = "active"
    PAUSED = "paused" 
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class ResourceSubscription:
    """Resource subscription definition"""
    uri: str
    subscriber_id: str
    created_at: datetime
    state: SubscriptionState = SubscriptionState.ACTIVE
    last_update: Optional[datetime] = None
    error_count: int = 0
    max_errors: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "uri": self.uri,
            "subscriberId": self.subscriber_id,
            "createdAt": self.created_at.isoformat(),
            "state": self.state.value,
            "lastUpdate": self.last_update.isoformat() if self.last_update else None,
            "errorCount": self.error_count
        }


@dataclass
class ResourceUpdate:
    """Resource update notification"""
    uri: str
    content: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP notification format"""
        result = {
            "method": "notifications/resources/updated",
            "params": {
                "uri": self.uri,
                "timestamp": self.timestamp.isoformat()
            }
        }
        
        if self.content:
            result["params"]["content"] = self.content
        if self.error:
            result["params"]["error"] = self.error
        
        return result


class ResourceWatcher:
    """Base class for resource watchers"""
    
    def __init__(self, uri: str):
        self.uri = uri
        self.running = False
        self._subscribers: Set[str] = set()
        self._update_handlers: List[Callable[[ResourceUpdate], None]] = []
    
    def add_subscriber(self, subscriber_id: str):
        """Add subscriber"""
        self._subscribers.add(subscriber_id)
        if not self.running:
            asyncio.create_task(self.start_watching())
    
    def remove_subscriber(self, subscriber_id: str):
        """Remove subscriber"""
        self._subscribers.discard(subscriber_id)
        if not self._subscribers and self.running:
            asyncio.create_task(self.stop_watching())
    
    def add_update_handler(self, handler: Callable[[ResourceUpdate], None]):
        """Add update handler"""
        self._update_handlers.append(handler)
    
    def _notify_update(self, update: ResourceUpdate):
        """Notify handlers of update"""
        for handler in self._update_handlers:
            try:
                handler(update)
            except Exception as e:
                logger.error(f"Error in update handler: {e}")
    
    async def start_watching(self):
        """Start watching for changes"""
        self.running = True
        logger.info(f"Started watching resource: {self.uri}")
    
    async def stop_watching(self):
        """Stop watching for changes"""
        self.running = False
        logger.info(f"Stopped watching resource: {self.uri}")
    
    async def get_current_content(self) -> Optional[Dict[str, Any]]:
        """Get current resource content"""
        raise NotImplementedError


class FileResourceWatcher(ResourceWatcher):
    """File system resource watcher"""
    
    def __init__(self, uri: str, file_path: str):
        super().__init__(uri)
        self.file_path = file_path
        self._last_modified = None
        self._watch_task = None
    
    async def start_watching(self):
        """Start watching file for changes"""
        await super().start_watching()
        self._watch_task = asyncio.create_task(self._watch_file())
    
    async def stop_watching(self):
        """Stop watching file"""
        await super().stop_watching()
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
    
    async def _watch_file(self):
        """Watch file for modifications"""
        import os
        import aiofiles
        
        while self.running:
            try:
                if os.path.exists(self.file_path):
                    stat = os.stat(self.file_path)
                    modified = stat.st_mtime
                    
                    if self._last_modified is None:
                        self._last_modified = modified
                    elif modified > self._last_modified:
                        self._last_modified = modified
                        
                        # Read new content
                        content = await self.get_current_content()
                        update = ResourceUpdate(
                            uri=self.uri,
                            content=content
                        )
                        self._notify_update(update)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error watching file {self.file_path}: {e}")
                update = ResourceUpdate(
                    uri=self.uri,
                    error=str(e)
                )
                self._notify_update(update)
                await asyncio.sleep(5)  # Back off on error
    
    async def get_current_content(self) -> Optional[Dict[str, Any]]:
        """Get current file content"""
        import aiofiles
        
        try:
            async with aiofiles.open(self.file_path, 'r') as f:
                content = await f.read()
                
            # Try to parse as JSON, fallback to text
            try:
                data = json.loads(content)
                return {
                    "type": "json",
                    "data": data
                }
            except json.JSONDecodeError:
                return {
                    "type": "text",
                    "text": content
                }
        except Exception as e:
            logger.error(f"Error reading file {self.file_path}: {e}")
            return None


class DatabaseResourceWatcher(ResourceWatcher):
    """Database resource watcher"""
    
    def __init__(self, uri: str, table: str, query: str):
        super().__init__(uri)
        self.table = table
        self.query = query
        self._last_checksum = None
        self._watch_task = None
    
    async def start_watching(self):
        """Start watching database for changes"""
        await super().start_watching()
        self._watch_task = asyncio.create_task(self._watch_database())
    
    async def stop_watching(self):
        """Stop watching database"""
        await super().stop_watching()
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
    
    async def _watch_database(self):
        """Watch database for changes"""
        while self.running:
            try:
                content = await self.get_current_content()
                if content:
                    # Simple checksum-based change detection
                    checksum = hash(json.dumps(content, sort_keys=True))
                    
                    if self._last_checksum is None:
                        self._last_checksum = checksum
                    elif checksum != self._last_checksum:
                        self._last_checksum = checksum
                        
                        update = ResourceUpdate(
                            uri=self.uri,
                            content=content
                        )
                        self._notify_update(update)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error watching database: {e}")
                update = ResourceUpdate(
                    uri=self.uri,
                    error=str(e)
                )
                self._notify_update(update)
                await asyncio.sleep(10)  # Back off on error
    
    async def get_current_content(self) -> Optional[Dict[str, Any]]:
        """Get current database content"""
        # This would integrate with actual database
        # For now, return mock data
        return {
            "table": self.table,
            "timestamp": datetime.utcnow().isoformat(),
            "rows": []  # Would contain actual query results
        }


class APIResourceWatcher(ResourceWatcher):
    """API endpoint resource watcher"""
    
    def __init__(self, uri: str, api_url: str, headers: Optional[Dict[str, str]] = None):
        super().__init__(uri)
        self.api_url = api_url
        self.headers = headers or {}
        self._last_etag = None
        self._last_modified = None
        self._watch_task = None
    
    async def start_watching(self):
        """Start watching API for changes"""
        await super().start_watching()
        self._watch_task = asyncio.create_task(self._watch_api())
    
    async def stop_watching(self):
        """Stop watching API"""
        await super().stop_watching()
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
    
    async def _watch_api(self):
        """Watch API endpoint for changes"""
        import aiohttp
        
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    # Use conditional requests if possible
                    headers = self.headers.copy()
                    if self._last_etag:
                        headers['If-None-Match'] = self._last_etag
                    if self._last_modified:
                        headers['If-Modified-Since'] = self._last_modified
                    
                    async with session.get(self.api_url, headers=headers) as response:
                        if response.status == 304:
                            # Not modified
                            pass
                        elif response.status == 200:
                            # Update available
                            self._last_etag = response.headers.get('ETag')
                            self._last_modified = response.headers.get('Last-Modified')
                            
                            content = await self.get_current_content()
                            if content:
                                update = ResourceUpdate(
                                    uri=self.uri,
                                    content=content
                                )
                                self._notify_update(update)
                        else:
                            logger.warning(f"API returned status {response.status}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error watching API {self.api_url}: {e}")
                update = ResourceUpdate(
                    uri=self.uri,
                    error=str(e)
                )
                self._notify_update(update)
                await asyncio.sleep(30)  # Back off on error
    
    async def get_current_content(self) -> Optional[Dict[str, Any]]:
        """Get current API content"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, headers=self.headers) as response:
                    if response.status == 200:
                        if response.headers.get('Content-Type', '').startswith('application/json'):
                            data = await response.json()
                            return {
                                "type": "json",
                                "data": data,
                                "headers": dict(response.headers)
                            }
                        else:
                            text = await response.text()
                            return {
                                "type": "text",
                                "text": text,
                                "headers": dict(response.headers)
                            }
        except Exception as e:
            logger.error(f"Error fetching API content: {e}")
            return None


class SubscriptionManager:
    """Manager for resource subscriptions"""
    
    def __init__(self):
        self.subscriptions: Dict[str, ResourceSubscription] = {}
        self.watchers: Dict[str, ResourceWatcher] = {}
        self.notification_handlers: List[Callable[[ResourceUpdate], None]] = []
    
    def add_notification_handler(self, handler: Callable[[ResourceUpdate], None]):
        """Add notification handler"""
        self.notification_handlers.append(handler)
    
    def _notify_subscribers(self, update: ResourceUpdate):
        """Notify all handlers of resource update"""
        for handler in self.notification_handlers:
            try:
                handler(update)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
    
    async def subscribe(self, uri: str, subscriber_id: str) -> ResourceSubscription:
        """Subscribe to resource updates"""
        subscription_key = f"{uri}:{subscriber_id}"
        
        if subscription_key in self.subscriptions:
            # Already subscribed
            return self.subscriptions[subscription_key]
        
        # Create new subscription
        subscription = ResourceSubscription(
            uri=uri,
            subscriber_id=subscriber_id,
            created_at=datetime.utcnow()
        )
        
        self.subscriptions[subscription_key] = subscription
        
        # Create or get watcher
        if uri not in self.watchers:
            watcher = self._create_watcher(uri)
            if watcher:
                self.watchers[uri] = watcher
                watcher.add_update_handler(self._notify_subscribers)
            else:
                raise ValueError(f"Cannot create watcher for URI: {uri}")
        
        # Add subscriber to watcher
        self.watchers[uri].add_subscriber(subscriber_id)
        
        logger.info(f"Subscribed {subscriber_id} to {uri}")
        return subscription
    
    async def unsubscribe(self, uri: str, subscriber_id: str):
        """Unsubscribe from resource updates"""
        subscription_key = f"{uri}:{subscriber_id}"
        
        if subscription_key not in self.subscriptions:
            return  # Not subscribed
        
        # Remove subscription
        subscription = self.subscriptions[subscription_key]
        subscription.state = SubscriptionState.CANCELLED
        del self.subscriptions[subscription_key]
        
        # Remove from watcher
        if uri in self.watchers:
            self.watchers[uri].remove_subscriber(subscriber_id)
            
            # Clean up watcher if no more subscribers
            if not self.watchers[uri]._subscribers:
                await self.watchers[uri].stop_watching()
                del self.watchers[uri]
        
        logger.info(f"Unsubscribed {subscriber_id} from {uri}")
    
    def _create_watcher(self, uri: str) -> Optional[ResourceWatcher]:
        """Create appropriate watcher for URI"""
        if uri.startswith("file://"):
            # File watcher
            file_path = uri[7:]  # Remove file:// prefix
            return FileResourceWatcher(uri, file_path)
        
        elif uri.startswith("db://") or uri.startswith("database://"):
            # Database watcher
            # Parse db://table/query format
            parts = uri.split("://", 1)[1].split("/", 1)
            table = parts[0]
            query = parts[1] if len(parts) > 1 else f"SELECT * FROM {table}"
            return DatabaseResourceWatcher(uri, table, query)
        
        elif uri.startswith("http://") or uri.startswith("https://"):
            # API watcher
            return APIResourceWatcher(uri, uri)
        
        elif uri.startswith("config://"):
            # Configuration watcher (treat as file)
            config_path = uri.replace("config://", "./config/") + ".json"
            return FileResourceWatcher(uri, config_path)
        
        elif uri.startswith("data://"):
            # Data resource watcher (could be database or file)
            # For now, treat as mock database
            table = uri.split("://")[1].split("/")[0]
            return DatabaseResourceWatcher(uri, table, f"SELECT * FROM {table}")
        
        else:
            logger.error(f"Unsupported URI scheme: {uri}")
            return None
    
    def list_subscriptions(self, subscriber_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List subscriptions, optionally filtered by subscriber"""
        subscriptions = list(self.subscriptions.values())
        
        if subscriber_id:
            subscriptions = [s for s in subscriptions if s.subscriber_id == subscriber_id]
        
        return [s.to_dict() for s in subscriptions]
    
    async def get_subscription_status(self, uri: str, subscriber_id: str) -> Optional[Dict[str, Any]]:
        """Get subscription status"""
        subscription_key = f"{uri}:{subscriber_id}"
        subscription = self.subscriptions.get(subscription_key)
        
        if not subscription:
            return None
        
        status = subscription.to_dict()
        
        # Add watcher status
        if uri in self.watchers:
            watcher = self.watchers[uri]
            status["watcherRunning"] = watcher.running
            status["subscriberCount"] = len(watcher._subscribers)
        
        return status


# Global subscription manager
subscription_manager = SubscriptionManager()


# Helper functions
def create_file_subscription(file_path: str, subscriber_id: str) -> str:
    """Create file subscription and return URI"""
    import os
    uri = f"file://{os.path.abspath(file_path)}"
    return uri


def create_database_subscription(table: str, query: str, subscriber_id: str) -> str:
    """Create database subscription and return URI"""
    uri = f"db://{table}/{query}"
    return uri


def create_api_subscription(api_url: str, subscriber_id: str) -> str:
    """Create API subscription and return URI"""
    return api_url  # API URLs are used directly as URIs