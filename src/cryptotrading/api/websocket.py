"""
WebSocket API - Real-time streaming endpoints
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Set

from flask import Blueprint, request
from flask_socketio import SocketIO, disconnect, emit, join_room, leave_room

from ..infrastructure.monitoring import get_logger
from ..services.market_service import MarketDataService
from ..services.ml_service import EnhancedMLService

logger = get_logger("websocket")

# WebSocket blueprint
websocket_bp = Blueprint("websocket", __name__)


class WebSocketManager:
    """Manages WebSocket connections and real-time data streaming"""

    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.market_service = MarketDataService()
        self.ml_service = EnhancedMLService()

        # Active subscriptions
        self.market_subscriptions: Dict[str, Set[str]] = {}  # symbol -> {session_ids}
        self.prediction_subscriptions: Dict[str, Set[str]] = {}  # symbol -> {session_ids}

        # Streaming tasks
        self.streaming_tasks: Dict[str, asyncio.Task] = {}

        # Register event handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register WebSocket event handlers"""

        @self.socketio.on("connect")
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            emit("connected", {"status": "Connected to CryptoTrading WebSocket"})

        @self.socketio.on("disconnect")
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
            self._cleanup_client_subscriptions(request.sid)

        @self.socketio.on("subscribe_market")
        def handle_market_subscription(data):
            """Subscribe to real-time market data"""
            try:
                symbol = data.get("symbol", "").upper()
                if not symbol:
                    emit("error", {"message": "Symbol is required"})
                    return

                # Add to subscriptions
                if symbol not in self.market_subscriptions:
                    self.market_subscriptions[symbol] = set()

                self.market_subscriptions[symbol].add(request.sid)
                join_room(f"market_{symbol}")

                # Start streaming if first subscriber
                if len(self.market_subscriptions[symbol]) == 1:
                    self._start_market_streaming(symbol)

                emit(
                    "subscribed",
                    {
                        "type": "market",
                        "symbol": symbol,
                        "message": f"Subscribed to {symbol} market data",
                    },
                )

                logger.info(f"Client {request.sid} subscribed to market data for {symbol}")

            except Exception as e:
                logger.error(f"Market subscription error: {e}")
                emit("error", {"message": f"Subscription failed: {str(e)}"})

        @self.socketio.on("unsubscribe_market")
        def handle_market_unsubscription(data):
            """Unsubscribe from market data"""
            try:
                symbol = data.get("symbol", "").upper()

                if symbol in self.market_subscriptions:
                    self.market_subscriptions[symbol].discard(request.sid)
                    leave_room(f"market_{symbol}")

                    # Stop streaming if no more subscribers
                    if len(self.market_subscriptions[symbol]) == 0:
                        self._stop_market_streaming(symbol)
                        del self.market_subscriptions[symbol]

                emit(
                    "unsubscribed",
                    {
                        "type": "market",
                        "symbol": symbol,
                        "message": f"Unsubscribed from {symbol} market data",
                    },
                )

                logger.info(f"Client {request.sid} unsubscribed from market data for {symbol}")

            except Exception as e:
                logger.error(f"Market unsubscription error: {e}")
                emit("error", {"message": f"Unsubscription failed: {str(e)}"})

        @self.socketio.on("subscribe_predictions")
        def handle_prediction_subscription(data):
            """Subscribe to real-time ML predictions"""
            try:
                symbol = data.get("symbol", "").upper()
                interval = data.get("interval", 60)  # seconds

                if not symbol:
                    emit("error", {"message": "Symbol is required"})
                    return

                # Add to subscriptions
                if symbol not in self.prediction_subscriptions:
                    self.prediction_subscriptions[symbol] = set()

                self.prediction_subscriptions[symbol].add(request.sid)
                join_room(f"predictions_{symbol}")

                # Start prediction streaming if first subscriber
                if len(self.prediction_subscriptions[symbol]) == 1:
                    self._start_prediction_streaming(symbol, interval)

                emit(
                    "subscribed",
                    {
                        "type": "predictions",
                        "symbol": symbol,
                        "interval": interval,
                        "message": f"Subscribed to {symbol} predictions (every {interval}s)",
                    },
                )

                logger.info(f"Client {request.sid} subscribed to predictions for {symbol}")

            except Exception as e:
                logger.error(f"Prediction subscription error: {e}")
                emit("error", {"message": f"Subscription failed: {str(e)}"})

        @self.socketio.on("get_realtime_analysis")
        def handle_realtime_analysis(data):
            """Get real-time analysis for a symbol"""
            try:
                symbol = data.get("symbol", "").upper()

                if not symbol:
                    emit("error", {"message": "Symbol is required"})
                    return

                # Run async analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Get market data
                    market_data = loop.run_until_complete(
                        self.market_service.get_realtime_price(symbol)
                    )

                    # Get ML prediction
                    prediction = loop.run_until_complete(
                        self.ml_service.get_prediction(symbol, "1h", "ensemble")
                    )

                    analysis = {
                        "symbol": symbol,
                        "timestamp": datetime.utcnow().isoformat(),
                        "market_data": market_data,
                        "prediction": prediction,
                        "type": "realtime_analysis",
                    }

                    emit("realtime_analysis", analysis)

                finally:
                    loop.close()

            except Exception as e:
                logger.error(f"Realtime analysis error: {e}")
                emit("error", {"message": f"Analysis failed: {str(e)}"})

    def _start_market_streaming(self, symbol: str):
        """Start streaming market data for a symbol"""
        task_key = f"market_{symbol}"

        if task_key not in self.streaming_tasks:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def stream_market_data():
                while symbol in self.market_subscriptions:
                    try:
                        # Get latest market data
                        data = await self.market_service.get_realtime_price(symbol)

                        # Add timestamp
                        data["timestamp"] = datetime.utcnow().isoformat()
                        data["type"] = "market_update"

                        # Emit to all subscribers
                        self.socketio.emit("market_data", data, room=f"market_{symbol}")

                        # Wait before next update
                        await asyncio.sleep(5)  # Update every 5 seconds

                    except Exception as e:
                        logger.error(f"Market streaming error for {symbol}: {e}")
                        await asyncio.sleep(10)  # Wait longer on error

            task = loop.create_task(stream_market_data())
            self.streaming_tasks[task_key] = task

            logger.info(f"Started market streaming for {symbol}")

    def _stop_market_streaming(self, symbol: str):
        """Stop streaming market data for a symbol"""
        task_key = f"market_{symbol}"

        if task_key in self.streaming_tasks:
            self.streaming_tasks[task_key].cancel()
            del self.streaming_tasks[task_key]
            logger.info(f"Stopped market streaming for {symbol}")

    def _start_prediction_streaming(self, symbol: str, interval: int):
        """Start streaming ML predictions for a symbol"""
        task_key = f"predictions_{symbol}"

        if task_key not in self.streaming_tasks:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def stream_predictions():
                while symbol in self.prediction_subscriptions:
                    try:
                        # Get latest prediction
                        prediction = await self.ml_service.get_prediction(symbol, "1h", "ensemble")

                        # Add timestamp and type
                        prediction["timestamp"] = datetime.utcnow().isoformat()
                        prediction["type"] = "prediction_update"

                        # Emit to all subscribers
                        self.socketio.emit(
                            "prediction_data", prediction, room=f"predictions_{symbol}"
                        )

                        # Wait for next prediction
                        await asyncio.sleep(interval)

                    except Exception as e:
                        logger.error(f"Prediction streaming error for {symbol}: {e}")
                        await asyncio.sleep(interval * 2)  # Wait longer on error

            task = loop.create_task(stream_predictions())
            self.streaming_tasks[task_key] = task

            logger.info(f"Started prediction streaming for {symbol} (interval: {interval}s)")

    def _cleanup_client_subscriptions(self, session_id: str):
        """Clean up subscriptions for disconnected client"""
        # Clean up market subscriptions
        for symbol in list(self.market_subscriptions.keys()):
            self.market_subscriptions[symbol].discard(session_id)
            if len(self.market_subscriptions[symbol]) == 0:
                self._stop_market_streaming(symbol)
                del self.market_subscriptions[symbol]

        # Clean up prediction subscriptions
        for symbol in list(self.prediction_subscriptions.keys()):
            self.prediction_subscriptions[symbol].discard(session_id)
            if len(self.prediction_subscriptions[symbol]) == 0:
                task_key = f"predictions_{symbol}"
                if task_key in self.streaming_tasks:
                    self.streaming_tasks[task_key].cancel()
                    del self.streaming_tasks[task_key]
                del self.prediction_subscriptions[symbol]

        logger.info(f"Cleaned up subscriptions for client {session_id}")

    def broadcast_alert(self, alert_type: str, message: str, data: Dict[str, Any] = None):
        """Broadcast system-wide alert"""
        alert = {
            "type": "system_alert",
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data or {},
        }

        self.socketio.emit("system_alert", alert)
        logger.info(f"Broadcasted {alert_type} alert: {message}")


# Global WebSocket manager instance
websocket_manager: WebSocketManager = None


def init_websocket(socketio: SocketIO):
    """Initialize WebSocket manager"""
    global websocket_manager
    websocket_manager = WebSocketManager(socketio)
    return websocket_manager


def get_websocket_manager() -> WebSocketManager:
    """Get WebSocket manager instance"""
    if websocket_manager is None:
        raise RuntimeError("WebSocket manager not initialized")
    return websocket_manager
