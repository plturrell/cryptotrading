"""
MCP Tools for Database Management Agent
Exposes comprehensive database operations via Model Context Protocol
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
import json
from datetime import datetime, timedelta
import pandas as pd

# Import database components
from ...data.database.models import (
    User, AIAnalysis, MarketData, TradingSignal, Portfolio, 
    Transaction, Alert, BacktestResult, Base
)
from ...infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)

class DatabaseMCPTools:
    """MCP tools for Database Management operations"""
    
    def __init__(self):
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions"""
        return [
            {
                "name": "query_market_data",
                "description": "Query historical market data from database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Cryptocurrency symbols to query"
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of records to return",
                            "default": 1000
                        }
                    },
                    "required": ["symbols"]
                }
            },
            {
                "name": "store_ai_analysis",
                "description": "Store AI analysis results in database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol"
                        },
                        "model": {
                            "type": "string",
                            "description": "AI model used (grok4, perplexity, etc.)"
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis (signal, news, market)"
                        },
                        "signal": {
                            "type": "string",
                            "description": "Trading signal (BUY/SELL/HOLD)",
                            "enum": ["BUY", "SELL", "HOLD"]
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score (0.0 to 1.0)"
                        },
                        "analysis": {
                            "type": "string",
                            "description": "Analysis text"
                        },
                        "raw_response": {
                            "type": "string",
                            "description": "Raw AI model response"
                        }
                    },
                    "required": ["symbol", "model", "analysis_type", "analysis"]
                }
            },
            {
                "name": "get_portfolio_data",
                "description": "Retrieve portfolio data for a user",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "integer",
                            "description": "User ID"
                        },
                        "include_transactions": {
                            "type": "boolean",
                            "description": "Include transaction history",
                            "default": False
                        },
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string"},
                                "end": {"type": "string"}
                            },
                            "description": "Date range filter"
                        }
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "store_trading_signal",
                "description": "Store trading signal in database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol"
                        },
                        "signal_type": {
                            "type": "string",
                            "description": "Type of signal",
                            "enum": ["BUY", "SELL", "HOLD"]
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Signal confidence (0.0 to 1.0)"
                        },
                        "price": {
                            "type": "number",
                            "description": "Price at signal generation"
                        },
                        "source": {
                            "type": "string",
                            "description": "Signal source (technical_analysis, ai_model, etc.)"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional signal metadata"
                        }
                    },
                    "required": ["symbol", "signal_type", "confidence", "price", "source"]
                }
            },
            {
                "name": "get_backtest_results",
                "description": "Retrieve backtesting results from database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "strategy_name": {
                            "type": "string",
                            "description": "Strategy name filter"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Symbol filter"
                        },
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string"},
                                "end": {"type": "string"}
                            },
                            "description": "Date range filter"
                        },
                        "min_return": {
                            "type": "number",
                            "description": "Minimum return filter"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return",
                            "default": 100
                        }
                    }
                }
            },
            {
                "name": "store_backtest_result",
                "description": "Store backtesting result in database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "strategy_name": {
                            "type": "string",
                            "description": "Strategy name"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol"
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Backtest start date"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "Backtest end date"
                        },
                        "initial_capital": {
                            "type": "number",
                            "description": "Initial capital amount"
                        },
                        "final_capital": {
                            "type": "number",
                            "description": "Final capital amount"
                        },
                        "total_return": {
                            "type": "number",
                            "description": "Total return percentage"
                        },
                        "sharpe_ratio": {
                            "type": "number",
                            "description": "Sharpe ratio"
                        },
                        "max_drawdown": {
                            "type": "number",
                            "description": "Maximum drawdown"
                        },
                        "total_trades": {
                            "type": "integer",
                            "description": "Total number of trades"
                        },
                        "win_rate": {
                            "type": "number",
                            "description": "Win rate percentage"
                        },
                        "config": {
                            "type": "object",
                            "description": "Strategy configuration"
                        },
                        "results": {
                            "type": "object",
                            "description": "Detailed results data"
                        }
                    },
                    "required": ["strategy_name", "symbol", "start_date", "end_date", 
                               "initial_capital", "final_capital", "total_return"]
                }
            },
            {
                "name": "get_database_health",
                "description": "Get database health and statistics",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_table_stats": {
                            "type": "boolean",
                            "description": "Include table-level statistics",
                            "default": True
                        },
                        "check_connections": {
                            "type": "boolean",
                            "description": "Check database connections",
                            "default": True
                        }
                    }
                }
            },
            {
                "name": "purge_expired_data",
                "description": "Purge expired data based on retention policies",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Table to clean up",
                            "enum": ["market_data", "ai_analyses", "trading_signals", "transactions", "alerts"]
                        },
                        "retention_days": {
                            "type": "integer",
                            "description": "Number of days to retain data",
                            "default": 365
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "Perform dry run without actual deletion",
                            "default": True
                        }
                    },
                    "required": ["table_name"]
                }
            }
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        try:
            if tool_name == "query_market_data":
                return await self._query_market_data(arguments)
            elif tool_name == "store_ai_analysis":
                return await self._store_ai_analysis(arguments)
            elif tool_name == "get_portfolio_data":
                return await self._get_portfolio_data(arguments)
            elif tool_name == "store_trading_signal":
                return await self._store_trading_signal(arguments)
            elif tool_name == "get_backtest_results":
                return await self._get_backtest_results(arguments)
            elif tool_name == "store_backtest_result":
                return await self._store_backtest_result(arguments)
            elif tool_name == "get_database_health":
                return await self._get_database_health(arguments)
            elif tool_name == "purge_expired_data":
                return await self._purge_expired_data(arguments)
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
    
    async def _query_market_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query market data from database"""
        symbols = args["symbols"]
        start_date = args.get("start_date")
        end_date = args.get("end_date")
        limit = args.get("limit", 1000)
        
        try:
            db = UnifiedDatabase()
            await db.initialize()
            
            # Build query conditions
            conditions = []
            if symbols:
                conditions.append(f"symbol IN ({','.join(['?' for _ in symbols])})")
            if start_date:
                conditions.append("timestamp >= ?")
            if end_date:
                conditions.append("timestamp <= ?")
            
            # Build query
            query = "SELECT * FROM market_data"
            params = []
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                params.extend(symbols)
                if start_date:
                    params.append(start_date)
                if end_date:
                    params.append(end_date)
            
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            # Execute query
            results = await db.execute_query(query, params)
            
            # Process results
            data_summary = {}
            for symbol in symbols:
                symbol_data = [r for r in results if r.get('symbol') == symbol]
                if symbol_data:
                    data_summary[symbol] = {
                        "records": len(symbol_data),
                        "latest_price": symbol_data[0].get('price'),
                        "date_range": {
                            "start": symbol_data[-1].get('timestamp'),
                            "end": symbol_data[0].get('timestamp')
                        }
                    }
            
            return {
                "success": True,
                "total_records": len(results),
                "symbols_found": len(data_summary),
                "data_summary": data_summary,
                "query_params": {
                    "symbols": symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to query market data: {str(e)}"
            }
    
    async def _store_ai_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store AI analysis in database"""
        try:
            db = UnifiedDatabase()
            await db.initialize()
            
            # Prepare data
            analysis_data = {
                "symbol": args["symbol"],
                "model": args["model"],
                "analysis_type": args["analysis_type"],
                "signal": args.get("signal"),
                "confidence": args.get("confidence"),
                "analysis": args["analysis"],
                "raw_response": args.get("raw_response"),
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Insert into database
            query = """
                INSERT INTO ai_analyses (symbol, model, analysis_type, signal, confidence, 
                                       analysis, raw_response, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = [
                analysis_data["symbol"],
                analysis_data["model"],
                analysis_data["analysis_type"],
                analysis_data["signal"],
                analysis_data["confidence"],
                analysis_data["analysis"],
                analysis_data["raw_response"],
                analysis_data["created_at"]
            ]
            
            result = await db.execute_query(query, params)
            
            return {
                "success": True,
                "analysis_id": result.get("lastrowid") if result else None,
                "stored_data": analysis_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to store AI analysis: {str(e)}"
            }
    
    async def _get_portfolio_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get portfolio data for user"""
        user_id = args["user_id"]
        include_transactions = args.get("include_transactions", False)
        date_range = args.get("date_range")
        
        try:
            db = UnifiedDatabase()
            await db.initialize()
            
            # Get portfolio data
            portfolio_query = "SELECT * FROM portfolios WHERE user_id = ?"
            portfolio_results = await db.execute_query(portfolio_query, [user_id])
            
            portfolio_data = {
                "user_id": user_id,
                "portfolios": portfolio_results,
                "total_value": sum(p.get("total_value", 0) for p in portfolio_results),
                "asset_count": len(portfolio_results)
            }
            
            # Include transactions if requested
            if include_transactions:
                tx_query = "SELECT * FROM transactions WHERE user_id = ?"
                tx_params = [user_id]
                
                if date_range:
                    tx_query += " AND created_at BETWEEN ? AND ?"
                    tx_params.extend([date_range["start"], date_range["end"]])
                
                tx_query += " ORDER BY created_at DESC LIMIT 100"
                
                transactions = await db.execute_query(tx_query, tx_params)
                portfolio_data["recent_transactions"] = transactions
                portfolio_data["transaction_count"] = len(transactions)
            
            return {
                "success": True,
                "portfolio": portfolio_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get portfolio data: {str(e)}"
            }
    
    async def _store_trading_signal(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store trading signal in database"""
        try:
            db = UnifiedDatabase()
            await db.initialize()
            
            signal_data = {
                "symbol": args["symbol"],
                "signal_type": args["signal_type"],
                "confidence": args["confidence"],
                "price": args["price"],
                "source": args["source"],
                "metadata": json.dumps(args.get("metadata", {})),
                "created_at": datetime.utcnow().isoformat()
            }
            
            query = """
                INSERT INTO trading_signals (symbol, signal_type, confidence, price, 
                                           source, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            params = [
                signal_data["symbol"],
                signal_data["signal_type"],
                signal_data["confidence"],
                signal_data["price"],
                signal_data["source"],
                signal_data["metadata"],
                signal_data["created_at"]
            ]
            
            result = await db.execute_query(query, params)
            
            return {
                "success": True,
                "signal_id": result.get("lastrowid") if result else None,
                "stored_signal": signal_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to store trading signal: {str(e)}"
            }
    
    async def _get_backtest_results(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get backtest results from database"""
        try:
            db = UnifiedDatabase()
            await db.initialize()
            
            # Build query
            query = "SELECT * FROM backtest_results"
            params = []
            conditions = []
            
            if args.get("strategy_name"):
                conditions.append("strategy_name = ?")
                params.append(args["strategy_name"])
            
            if args.get("symbol"):
                conditions.append("symbol = ?")
                params.append(args["symbol"])
            
            if args.get("min_return"):
                conditions.append("total_return >= ?")
                params.append(args["min_return"])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            
            if args.get("limit"):
                query += f" LIMIT {args['limit']}"
            
            results = await db.execute_query(query, params)
            
            # Calculate summary statistics
            if results:
                returns = [r.get("total_return", 0) for r in results]
                summary = {
                    "total_backtests": len(results),
                    "avg_return": sum(returns) / len(returns),
                    "best_return": max(returns),
                    "worst_return": min(returns),
                    "profitable_strategies": len([r for r in returns if r > 0])
                }
            else:
                summary = {"total_backtests": 0}
            
            return {
                "success": True,
                "results": results,
                "summary": summary
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get backtest results: {str(e)}"
            }
    
    async def _store_backtest_result(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store backtest result in database"""
        try:
            db = UnifiedDatabase()
            await db.initialize()
            
            backtest_data = {
                "strategy_name": args["strategy_name"],
                "symbol": args["symbol"],
                "start_date": args["start_date"],
                "end_date": args["end_date"],
                "initial_capital": args["initial_capital"],
                "final_capital": args["final_capital"],
                "total_return": args["total_return"],
                "sharpe_ratio": args.get("sharpe_ratio"),
                "max_drawdown": args.get("max_drawdown"),
                "total_trades": args.get("total_trades"),
                "win_rate": args.get("win_rate"),
                "config": json.dumps(args.get("config", {})),
                "results": json.dumps(args.get("results", {})),
                "created_at": datetime.utcnow().isoformat()
            }
            
            query = """
                INSERT INTO backtest_results (strategy_name, symbol, start_date, end_date,
                                            initial_capital, final_capital, total_return,
                                            sharpe_ratio, max_drawdown, total_trades,
                                            win_rate, config, results, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = list(backtest_data.values())
            
            result = await db.execute_query(query, params)
            
            return {
                "success": True,
                "backtest_id": result.get("lastrowid") if result else None,
                "stored_result": backtest_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to store backtest result: {str(e)}"
            }
    
    async def _get_database_health(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get database health statistics"""
        try:
            db = UnifiedDatabase()
            await db.initialize()
            
            health_data = {
                "status": "healthy",
                "connection_status": "connected",
                "database_type": db.db_type if hasattr(db, 'db_type') else "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if args.get("include_table_stats", True):
                # Get table statistics
                tables = ["users", "market_data", "ai_analyses", "trading_signals", 
                         "portfolios", "transactions", "backtest_results", "alerts"]
                
                table_stats = {}
                for table in tables:
                    try:
                        count_result = await db.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                        table_stats[table] = {
                            "record_count": count_result[0]["count"] if count_result else 0,
                            "status": "accessible"
                        }
                    except Exception as e:
                        table_stats[table] = {
                            "record_count": 0,
                            "status": f"error: {str(e)}"
                        }
                
                health_data["table_statistics"] = table_stats
                health_data["total_records"] = sum(
                    stats.get("record_count", 0) for stats in table_stats.values()
                )
            
            return {
                "success": True,
                "health": health_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get database health: {str(e)}",
                "status": "unhealthy"
            }
    
    async def _purge_expired_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Purge expired data from database"""
        table_name = args["table_name"]
        retention_days = args.get("retention_days", 365)
        dry_run = args.get("dry_run", True)
        
        try:
            db = UnifiedDatabase()
            await db.initialize()
            
            # Calculate cutoff date
            cutoff_date = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
            
            # First, count records to be deleted
            count_query = f"SELECT COUNT(*) as count FROM {table_name} WHERE created_at < ?"
            count_result = await db.execute_query(count_query, [cutoff_date])
            records_to_delete = count_result[0]["count"] if count_result else 0
            
            cleanup_result = {
                "table": table_name,
                "retention_days": retention_days,
                "cutoff_date": cutoff_date,
                "records_to_delete": records_to_delete,
                "dry_run": dry_run
            }
            
            if not dry_run and records_to_delete > 0:
                # Perform actual deletion
                delete_query = f"DELETE FROM {table_name} WHERE created_at < ?"
                delete_result = await db.execute_query(delete_query, [cutoff_date])
                cleanup_result["records_deleted"] = delete_result.get("rowcount", 0)
                cleanup_result["status"] = "completed"
            else:
                cleanup_result["status"] = "dry_run" if dry_run else "no_records_to_delete"
            
            return {
                "success": True,
                "cleanup": cleanup_result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to cleanup old data: {str(e)}"
            }

# Export for MCP server registration
database_mcp_tools = DatabaseMCPTools()
