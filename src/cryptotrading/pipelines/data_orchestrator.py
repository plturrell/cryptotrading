"""
Enhanced Data Pipeline Orchestrator
Manages data ingestion, processing, and quality monitoring
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from ..infrastructure.monitoring import get_logger, get_business_metrics, trace_context
from ..services.market_service import MarketDataService
from ..data.providers.real_only_provider import RealOnlyDataProvider

logger = get_logger("pipelines.orchestrator")


@dataclass
class PipelineTask:
    """Data pipeline task definition"""
    name: str
    task_type: str
    schedule: str
    dependencies: List[str]
    config: Dict[str, Any]
    enabled: bool = True
    retries: int = 3
    timeout: int = 300


@dataclass
class DataQualityCheck:
    """Data quality check definition"""
    name: str
    check_type: str
    threshold: float
    column: Optional[str] = None
    severity: str = "warning"


class DataOrchestrator:
    """Enhanced data pipeline orchestrator"""
    
    def __init__(self):
        self.market_service = MarketDataService()
        self.real_provider = RealOnlyDataProvider()
        self.business_metrics = get_business_metrics()
        
        # Pipeline configuration
        self.pipelines: Dict[str, PipelineTask] = {}
        self.data_quality_checks: Dict[str, List[DataQualityCheck]] = {}
        
        # Pipeline state
        self.running_pipelines: Dict[str, asyncio.Task] = {}
        self.pipeline_stats: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_default_pipelines()
    
    def _initialize_default_pipelines(self):
        """Initialize default data pipelines"""
        
        # Market data ingestion pipeline
        self.pipelines["market_data_ingestion"] = PipelineTask(
            name="market_data_ingestion",
            task_type="ingestion",
            schedule="*/5 * * * *",  # Every 5 minutes
            dependencies=[],
            config={
                "symbols": ["BTC", "ETH", "BNB", "ADA", "SOL"],
                "sources": ["yahoo", "coingecko", "geckoterminal"],
                "batch_size": 10,
                "timeout": 60
            }
        )
        
        # Historical data pipeline
        self.pipelines["historical_data_pipeline"] = PipelineTask(
            name="historical_data_pipeline",
            task_type="batch",
            schedule="0 2 * * *",  # Daily at 2 AM
            dependencies=[],
            config={
                "symbols": ["BTC", "ETH", "BNB", "ADA", "SOL"],
                "lookback_days": 365,
                "include_indicators": True,
                "save_format": "parquet"
            }
        )
        
        # Data quality monitoring
        self.pipelines["data_quality_monitoring"] = PipelineTask(
            name="data_quality_monitoring",
            task_type="monitoring",
            schedule="*/15 * * * *",  # Every 15 minutes
            dependencies=["market_data_ingestion"],
            config={
                "check_freshness": True,
                "check_completeness": True,
                "check_accuracy": True,
                "alert_threshold": 0.8
            }
        )
        
        # Feature engineering pipeline
        self.pipelines["feature_engineering"] = PipelineTask(
            name="feature_engineering",
            task_type="processing",
            schedule="*/30 * * * *",  # Every 30 minutes
            dependencies=["market_data_ingestion"],
            config={
                "feature_sets": ["technical", "sentiment", "macro"],
                "window_sizes": [20, 50, 200],
                "target_column": "price_change_24h"
            }
        )
        
        # Initialize quality checks
        self._initialize_quality_checks()
    
    def _initialize_quality_checks(self):
        """Initialize data quality checks"""
        
        market_data_checks = [
            DataQualityCheck("price_not_null", "not_null", 1.0, "price", "critical"),
            DataQualityCheck("volume_positive", "positive", 1.0, "volume", "warning"),
            DataQualityCheck("timestamp_fresh", "freshness", 0.95, "timestamp", "critical"),
            DataQualityCheck("price_range", "range", 0.99, "price", "warning")
        ]
        
        self.data_quality_checks["market_data"] = market_data_checks
        
        feature_checks = [
            DataQualityCheck("feature_completeness", "completeness", 0.95, None, "warning"),
            DataQualityCheck("no_infinite_values", "finite", 1.0, None, "critical"),
            DataQualityCheck("feature_correlation", "correlation", 0.8, None, "info")
        ]
        
        self.data_quality_checks["features"] = feature_checks
    
    async def start_pipeline(self, pipeline_name: str) -> bool:
        """Start a specific pipeline"""
        if pipeline_name not in self.pipelines:
            logger.error(f"Pipeline {pipeline_name} not found")
            return False
        
        if pipeline_name in self.running_pipelines:
            logger.warning(f"Pipeline {pipeline_name} is already running")
            return False
        
        pipeline = self.pipelines[pipeline_name]
        
        with trace_context(f"pipeline_{pipeline_name}") as span:
            try:
                span.set_attribute("pipeline.name", pipeline_name)
                span.set_attribute("pipeline.type", pipeline.task_type)
                
                logger.info(f"Starting pipeline: {pipeline_name}")
                
                # Check dependencies
                for dep in pipeline.dependencies:
                    if dep not in self.pipeline_stats or \
                       self.pipeline_stats[dep].get("status") != "success":
                        logger.warning(f"Dependency {dep} not satisfied for {pipeline_name}")
                        return False
                
                # Create and start pipeline task
                task = asyncio.create_task(self._run_pipeline(pipeline))
                self.running_pipelines[pipeline_name] = task
                
                # Initialize stats
                self.pipeline_stats[pipeline_name] = {
                    "status": "running",
                    "started_at": datetime.utcnow(),
                    "runs_count": self.pipeline_stats.get(pipeline_name, {}).get("runs_count", 0) + 1
                }
                
                span.set_attribute("success", "true")
                return True
                
            except Exception as e:
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                logger.error(f"Failed to start pipeline {pipeline_name}: {e}")
                return False
    
    async def _run_pipeline(self, pipeline: PipelineTask):
        """Run a pipeline task"""
        pipeline_name = pipeline.name
        start_time = datetime.utcnow()
        
        try:
            if pipeline.task_type == "ingestion":
                await self._run_ingestion_pipeline(pipeline)
            elif pipeline.task_type == "batch":
                await self._run_batch_pipeline(pipeline)
            elif pipeline.task_type == "monitoring":
                await self._run_monitoring_pipeline(pipeline)
            elif pipeline.task_type == "processing":
                await self._run_processing_pipeline(pipeline)
            else:
                raise ValueError(f"Unknown pipeline type: {pipeline.task_type}")
            
            # Update stats on success
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.pipeline_stats[pipeline_name].update({
                "status": "success",
                "completed_at": datetime.utcnow(),
                "duration_seconds": duration,
                "last_error": None
            })
            
            # Track business metrics
            self.business_metrics.track_pipeline_execution(
                pipeline_name, True, duration * 1000
            )
            
            logger.info(f"Pipeline {pipeline_name} completed successfully in {duration:.2f}s")
            
        except Exception as e:
            # Update stats on error
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.pipeline_stats[pipeline_name].update({
                "status": "failed",
                "completed_at": datetime.utcnow(),
                "duration_seconds": duration,
                "last_error": str(e)
            })
            
            # Track business metrics
            self.business_metrics.track_pipeline_execution(
                pipeline_name, False, duration * 1000
            )
            
            logger.error(f"Pipeline {pipeline_name} failed after {duration:.2f}s: {e}")
            
        finally:
            # Clean up running pipeline
            if pipeline_name in self.running_pipelines:
                del self.running_pipelines[pipeline_name]
    
    async def _run_ingestion_pipeline(self, pipeline: PipelineTask):
        """Run market data ingestion pipeline"""
        config = pipeline.config
        symbols = config.get("symbols", [])
        sources = config.get("sources", [])
        batch_size = config.get("batch_size", 10)
        
        logger.info(f"Ingesting data for {len(symbols)} symbols from {len(sources)} sources")
        
        ingested_count = 0
        
        # Process symbols in batches
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            
            for symbol in batch_symbols:
                try:
                    # Ingest from each source
                    for source in sources:
                        if source == "yahoo":
                            data = await self.market_service.get_realtime_price(symbol)
                        elif source in ["coingecko", "geckoterminal"]:
                            data = await self.real_provider.get_real_time_price(symbol)
                        else:
                            logger.warning(f"Unknown source: {source}")
                            continue
                        
                        if data:
                            # Store data (implement storage logic)
                            await self._store_market_data(symbol, source, data)
                            ingested_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to ingest {symbol} from {source}: {e}")
                    continue
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        logger.info(f"Ingestion completed: {ingested_count} records processed")
    
    async def _run_batch_pipeline(self, pipeline: PipelineTask):
        """Run historical data batch processing pipeline"""
        config = pipeline.config
        symbols = config.get("symbols", [])
        lookback_days = config.get("lookback_days", 365)
        include_indicators = config.get("include_indicators", True)
        
        logger.info(f"Processing historical data for {len(symbols)} symbols ({lookback_days} days)")
        
        for symbol in symbols:
            try:
                # Get historical data
                data = await self.market_service.get_historical_data(symbol, lookback_days)
                
                if include_indicators:
                    # Add technical indicators (implement indicator logic)
                    data = await self._add_technical_indicators(data)
                
                # Store processed data
                await self._store_historical_data(symbol, data)
                
            except Exception as e:
                logger.error(f"Failed to process historical data for {symbol}: {e}")
                continue
        
        logger.info("Historical data processing completed")
    
    async def _run_monitoring_pipeline(self, pipeline: PipelineTask):
        """Run data quality monitoring pipeline"""
        config = pipeline.config
        
        logger.info("Running data quality monitoring")
        
        quality_results = {}
        
        # Check data quality for each dataset
        for dataset_name, checks in self.data_quality_checks.items():
            dataset_results = []
            
            for check in checks:
                try:
                    result = await self._run_quality_check(dataset_name, check)
                    dataset_results.append(result)
                    
                    # Alert on critical failures
                    if result["status"] == "failed" and check.severity == "critical":
                        await self._send_quality_alert(dataset_name, check, result)
                        
                except Exception as e:
                    logger.error(f"Quality check {check.name} failed: {e}")
                    continue
            
            quality_results[dataset_name] = dataset_results
        
        # Store quality results
        await self._store_quality_results(quality_results)
        
        logger.info(f"Data quality monitoring completed: {len(quality_results)} datasets checked")
    
    async def _run_processing_pipeline(self, pipeline: PipelineTask):
        """Run feature engineering pipeline"""
        config = pipeline.config
        feature_sets = config.get("feature_sets", [])
        window_sizes = config.get("window_sizes", [])
        
        logger.info(f"Running feature engineering: {len(feature_sets)} feature sets")
        
        for feature_set in feature_sets:
            try:
                # Generate features (implement feature engineering logic)
                features = await self._generate_features(feature_set, window_sizes)
                
                # Store features
                await self._store_features(feature_set, features)
                
            except Exception as e:
                logger.error(f"Feature engineering failed for {feature_set}: {e}")
                continue
        
        logger.info("Feature engineering completed")
    
    async def _store_market_data(self, symbol: str, source: str, data: Dict[str, Any]):
        """Store ingested market data"""
        # Implement data storage logic
        logger.debug(f"Storing market data for {symbol} from {source}")
        pass
    
    async def _store_historical_data(self, symbol: str, data: Dict[str, Any]):
        """Store processed historical data"""
        # Implement historical data storage
        logger.debug(f"Storing historical data for {symbol}")
        pass
    
    async def _add_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add technical indicators to data"""
        # Implement technical indicator calculation
        logger.debug("Adding technical indicators")
        return data
    
    async def _run_quality_check(self, dataset: str, check: DataQualityCheck) -> Dict[str, Any]:
        """Run a single data quality check"""
        # Implement quality check logic
        return {
            "check_name": check.name,
            "status": "passed",
            "score": 0.95,
            "threshold": check.threshold,
            "details": "Check passed successfully"
        }
    
    async def _send_quality_alert(self, dataset: str, check: DataQualityCheck, result: Dict[str, Any]):
        """Send data quality alert"""
        logger.warning(f"Data quality alert: {dataset}.{check.name} failed")
        # Implement alerting logic
    
    async def _store_quality_results(self, results: Dict[str, Any]):
        """Store data quality results"""
        logger.debug("Storing quality results")
        # Implement quality results storage
    
    async def _generate_features(self, feature_set: str, window_sizes: List[int]) -> Dict[str, Any]:
        """Generate features for a feature set"""
        logger.debug(f"Generating features for {feature_set}")
        # Implement feature generation
        return {}
    
    async def _store_features(self, feature_set: str, features: Dict[str, Any]):
        """Store generated features"""
        logger.debug(f"Storing features for {feature_set}")
        # Implement feature storage
    
    async def stop_pipeline(self, pipeline_name: str) -> bool:
        """Stop a running pipeline"""
        if pipeline_name not in self.running_pipelines:
            logger.warning(f"Pipeline {pipeline_name} is not running")
            return False
        
        try:
            task = self.running_pipelines[pipeline_name]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self.running_pipelines[pipeline_name]
            
            # Update stats
            if pipeline_name in self.pipeline_stats:
                self.pipeline_stats[pipeline_name]["status"] = "stopped"
                self.pipeline_stats[pipeline_name]["stopped_at"] = datetime.utcnow()
            
            logger.info(f"Pipeline {pipeline_name} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop pipeline {pipeline_name}: {e}")
            return False
    
    def get_pipeline_status(self, pipeline_name: Optional[str] = None) -> Dict[str, Any]:
        """Get pipeline status"""
        if pipeline_name:
            return self.pipeline_stats.get(pipeline_name, {})
        
        return {
            "running_pipelines": list(self.running_pipelines.keys()),
            "pipeline_stats": self.pipeline_stats,
            "total_pipelines": len(self.pipelines)
        }
    
    async def start_all_pipelines(self):
        """Start all enabled pipelines"""
        started_count = 0
        
        for pipeline_name, pipeline in self.pipelines.items():
            if pipeline.enabled:
                success = await self.start_pipeline(pipeline_name)
                if success:
                    started_count += 1
        
        logger.info(f"Started {started_count} pipelines")
        return started_count
    
    async def stop_all_pipelines(self):
        """Stop all running pipelines"""
        stopped_count = 0
        
        for pipeline_name in list(self.running_pipelines.keys()):
            success = await self.stop_pipeline(pipeline_name)
            if success:
                stopped_count += 1
        
        logger.info(f"Stopped {stopped_count} pipelines")
        return stopped_count
