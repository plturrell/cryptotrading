"""
A/B Testing Framework for MCTS Algorithm Variants
Enables systematic comparison of different MCTS configurations and algorithms
"""
import asyncio
import time
import statistics
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import random
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VariantType(Enum):
    """Types of algorithm variants"""
    EXPLORATION_PARAM = "exploration_param"
    SIMULATION_DEPTH = "simulation_depth"
    RAVE_WEIGHTING = "rave_weighting"
    PROGRESSIVE_WIDENING = "progressive_widening"
    TREE_PRUNING = "tree_pruning"
    CONVERGENCE_THRESHOLD = "convergence_threshold"
    PARALLEL_SIMULATIONS = "parallel_simulations"


@dataclass
class VariantConfig:
    """Configuration for an algorithm variant"""
    variant_id: str
    variant_type: VariantType
    name: str
    description: str
    parameters: Dict[str, Any]
    weight: float = 1.0  # Traffic allocation weight
    is_control: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'variant_id': self.variant_id,
            'variant_type': self.variant_type.value,
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'weight': self.weight,
            'is_control': self.is_control,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ExperimentResult:
    """Result from running an experiment variant"""
    variant_id: str
    execution_time: float
    iterations_completed: int
    expected_value: float
    confidence: float
    convergence_reason: str
    memory_usage_mb: float
    tree_size: int
    efficiency_gain: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'variant_id': self.variant_id,
            'execution_time': self.execution_time,
            'iterations_completed': self.iterations_completed,
            'expected_value': self.expected_value,
            'confidence': self.confidence,
            'convergence_reason': self.convergence_reason,
            'memory_usage_mb': self.memory_usage_mb,
            'tree_size': self.tree_size,
            'efficiency_gain': self.efficiency_gain,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ExperimentStats:
    """Statistical analysis of experiment results"""
    variant_id: str
    sample_size: int
    mean_execution_time: float
    std_execution_time: float
    mean_expected_value: float
    std_expected_value: float
    mean_confidence: float
    mean_efficiency_gain: float
    success_rate: float
    percentile_95_time: float
    percentile_99_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'variant_id': self.variant_id,
            'sample_size': self.sample_size,
            'mean_execution_time': self.mean_execution_time,
            'std_execution_time': self.std_execution_time,
            'mean_expected_value': self.mean_expected_value,
            'std_expected_value': self.std_expected_value,
            'mean_confidence': self.mean_confidence,
            'mean_efficiency_gain': self.mean_efficiency_gain,
            'success_rate': self.success_rate,
            'percentile_95_time': self.percentile_95_time,
            'percentile_99_time': self.percentile_99_time
        }


class ExperimentStrategy(ABC):
    """Abstract base class for experiment strategies"""
    
    @abstractmethod
    async def run_variant(self, variant: VariantConfig, 
                         mcts_agent: Any, 
                         test_parameters: Dict[str, Any]) -> ExperimentResult:
        """Run a specific variant and return results"""
        pass
    
    @abstractmethod
    def get_variant_hash(self, variant: VariantConfig, 
                        test_parameters: Dict[str, Any]) -> str:
        """Generate a hash for caching/deduplication"""
        pass


class MCTSExperimentStrategy(ExperimentStrategy):
    """Default strategy for MCTS experiments"""
    
    async def run_variant(self, variant: VariantConfig, 
                         mcts_agent: Any, 
                         test_parameters: Dict[str, Any]) -> ExperimentResult:
        """Run MCTS with variant configuration"""
        start_time = time.time()
        
        # Apply variant configuration to agent
        original_config = self._backup_agent_config(mcts_agent)
        self._apply_variant_config(mcts_agent, variant)
        
        try:
            # Set up environment
            mcts_agent.environment = mcts_agent._create_test_environment(test_parameters)
            
            # Run MCTS
            result = await mcts_agent.run_mcts_parallel(
                iterations=test_parameters.get('iterations', 1000)
            )
            
            execution_time = time.time() - start_time
            
            # Extract metrics
            stats = result.get('stats', {})
            
            return ExperimentResult(
                variant_id=variant.variant_id,
                execution_time=execution_time,
                iterations_completed=stats.get('iterations', 0),
                expected_value=result.get('expected_value', 0),
                confidence=result.get('confidence', 0),
                convergence_reason=stats.get('convergence_reason', 'unknown'),
                memory_usage_mb=self._estimate_memory_usage(stats),
                tree_size=stats.get('tree_size', 0),
                efficiency_gain=stats.get('efficiency_gain', 0)
            )
            
        except Exception as e:
            logger.error(f"Variant {variant.variant_id} failed: {e}")
            return ExperimentResult(
                variant_id=variant.variant_id,
                execution_time=time.time() - start_time,
                iterations_completed=0,
                expected_value=0,
                confidence=0,
                convergence_reason=f"error_{type(e).__name__}",
                memory_usage_mb=0,
                tree_size=0,
                efficiency_gain=0
            )
        
        finally:
            # Restore original configuration
            self._restore_agent_config(mcts_agent, original_config)
    
    def _backup_agent_config(self, agent: Any) -> Dict[str, Any]:
        """Backup current agent configuration"""
        return {
            'exploration_constant': agent.config.exploration_constant,
            'simulation_depth': agent.config.simulation_depth,
            'enable_rave': agent.config.enable_rave,
            'enable_progressive_widening': agent.config.enable_progressive_widening,
            'parallel_simulations': agent.config.parallel_simulations
        }
    
    def _apply_variant_config(self, agent: Any, variant: VariantConfig):
        """Apply variant configuration to agent"""
        params = variant.parameters
        
        if variant.variant_type == VariantType.EXPLORATION_PARAM:
            agent.config.exploration_constant = params.get('c_param', 1.4)
        
        elif variant.variant_type == VariantType.SIMULATION_DEPTH:
            agent.config.simulation_depth = params.get('depth', 10)
        
        elif variant.variant_type == VariantType.RAVE_WEIGHTING:
            agent.config.enable_rave = params.get('enabled', True)
        
        elif variant.variant_type == VariantType.PROGRESSIVE_WIDENING:
            agent.config.enable_progressive_widening = params.get('enabled', True)
        
        elif variant.variant_type == VariantType.PARALLEL_SIMULATIONS:
            agent.config.parallel_simulations = params.get('count', 4)
    
    def _restore_agent_config(self, agent: Any, original_config: Dict[str, Any]):
        """Restore agent configuration"""
        agent.config.exploration_constant = original_config['exploration_constant']
        agent.config.simulation_depth = original_config['simulation_depth']
        agent.config.enable_rave = original_config['enable_rave']
        agent.config.enable_progressive_widening = original_config['enable_progressive_widening']
        agent.config.parallel_simulations = original_config['parallel_simulations']
    
    def _estimate_memory_usage(self, stats: Dict[str, Any]) -> float:
        """Estimate memory usage from stats"""
        tree_size = stats.get('tree_size', 0)
        return tree_size * 0.0002  # Rough estimate: 200 bytes per node
    
    def get_variant_hash(self, variant: VariantConfig, 
                        test_parameters: Dict[str, Any]) -> str:
        """Generate hash for variant + parameters"""
        combined_data = {
            'variant': variant.to_dict(),
            'test_params': test_parameters
        }
        combined_str = json.dumps(combined_data, sort_keys=True)
        return hashlib.md5(combined_str.encode()).hexdigest()


class ABTestManager:
    """Manages A/B tests for MCTS algorithm variants"""
    
    def __init__(self, strategy: ExperimentStrategy = None):
        self.strategy = strategy or MCTSExperimentStrategy()
        self.experiments: Dict[str, List[VariantConfig]] = {}
        self.results: Dict[str, List[ExperimentResult]] = {}
        self.result_cache: Dict[str, ExperimentResult] = {}
        
    def create_experiment(self, experiment_id: str, 
                         variants: List[VariantConfig]) -> str:
        """Create a new A/B test experiment"""
        if experiment_id in self.experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")
        
        # Validate variants
        control_count = sum(1 for v in variants if v.is_control)
        if control_count != 1:
            raise ValueError("Exactly one variant must be marked as control")
        
        self.experiments[experiment_id] = variants
        self.results[experiment_id] = []
        
        logger.info(f"Created experiment {experiment_id} with {len(variants)} variants")
        return experiment_id
    
    def add_predefined_variants(self, experiment_id: str) -> List[VariantConfig]:
        """Add common predefined variants for testing"""
        variants = [
            # Control variant
            VariantConfig(
                variant_id="control",
                variant_type=VariantType.EXPLORATION_PARAM,
                name="Control (default)",
                description="Default MCTS configuration",
                parameters={'c_param': 1.4},
                is_control=True
            ),
            
            # Exploration parameter variants
            VariantConfig(
                variant_id="high_exploration",
                variant_type=VariantType.EXPLORATION_PARAM,
                name="High Exploration",
                description="Increased exploration parameter",
                parameters={'c_param': 2.0}
            ),
            
            VariantConfig(
                variant_id="low_exploration", 
                variant_type=VariantType.EXPLORATION_PARAM,
                name="Low Exploration",
                description="Decreased exploration parameter",
                parameters={'c_param': 1.0}
            ),
            
            # Simulation depth variants
            VariantConfig(
                variant_id="deep_simulation",
                variant_type=VariantType.SIMULATION_DEPTH,
                name="Deep Simulation",
                description="Increased simulation depth",
                parameters={'depth': 15}
            ),
            
            VariantConfig(
                variant_id="shallow_simulation",
                variant_type=VariantType.SIMULATION_DEPTH,
                name="Shallow Simulation", 
                description="Decreased simulation depth",
                parameters={'depth': 5}
            ),
            
            # RAVE variants
            VariantConfig(
                variant_id="no_rave",
                variant_type=VariantType.RAVE_WEIGHTING,
                name="No RAVE",
                description="Disable RAVE algorithm",
                parameters={'enabled': False}
            ),
            
            # Parallel simulation variants
            VariantConfig(
                variant_id="high_parallel",
                variant_type=VariantType.PARALLEL_SIMULATIONS,
                name="High Parallelism",
                description="More parallel simulations",
                parameters={'count': 8}
            )
        ]
        
        self.create_experiment(experiment_id, variants)
        return variants
    
    async def run_experiment(self, experiment_id: str, 
                           mcts_agent: Any,
                           test_parameters: Dict[str, Any],
                           runs_per_variant: int = 10,
                           max_concurrent: int = 3) -> Dict[str, Any]:
        """Run A/B test experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        variants = self.experiments[experiment_id]
        all_results = []
        
        logger.info(f"Starting experiment {experiment_id} with {runs_per_variant} runs per variant")
        
        # Run variants with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_variant_with_semaphore(variant: VariantConfig, run_id: int):
            async with semaphore:
                # Check cache first
                cache_key = f"{self.strategy.get_variant_hash(variant, test_parameters)}_{run_id}"
                if cache_key in self.result_cache:
                    return self.result_cache[cache_key]
                
                result = await self.strategy.run_variant(variant, mcts_agent, test_parameters)
                self.result_cache[cache_key] = result
                return result
        
        # Create tasks for all variant runs
        tasks = []
        for variant in variants:
            for run_id in range(runs_per_variant):
                task = run_variant_with_semaphore(variant, run_id)
                tasks.append(task)
        
        # Execute all tasks
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Store results
        self.results[experiment_id] = results
        
        # Analyze results
        analysis = self._analyze_results(experiment_id, results)
        
        logger.info(f"Experiment {experiment_id} completed in {total_time:.2f}s")
        
        return {
            'experiment_id': experiment_id,
            'total_runs': len(results),
            'execution_time': total_time,
            'analysis': analysis,
            'raw_results': [r.to_dict() for r in results]
        }
    
    def _analyze_results(self, experiment_id: str, 
                        results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze experiment results"""
        # Group results by variant
        variant_results = {}
        for result in results:
            if result.variant_id not in variant_results:
                variant_results[result.variant_id] = []
            variant_results[result.variant_id].append(result)
        
        # Calculate statistics for each variant
        variant_stats = {}
        for variant_id, variant_results_list in variant_results.items():
            stats = self._calculate_variant_stats(variant_id, variant_results_list)
            variant_stats[variant_id] = stats
        
        # Find best performing variants
        best_by_value = max(variant_stats.items(), 
                           key=lambda x: x[1].mean_expected_value)
        best_by_time = min(variant_stats.items(),
                          key=lambda x: x[1].mean_execution_time)
        best_by_efficiency = max(variant_stats.items(),
                               key=lambda x: x[1].mean_efficiency_gain)
        
        # Statistical significance tests
        significance_tests = self._run_significance_tests(variant_stats)
        
        return {
            'variant_stats': {k: v.to_dict() for k, v in variant_stats.items()},
            'best_by_expected_value': {
                'variant_id': best_by_value[0],
                'mean_value': best_by_value[1].mean_expected_value
            },
            'best_by_execution_time': {
                'variant_id': best_by_time[0],
                'mean_time': best_by_time[1].mean_execution_time
            },
            'best_by_efficiency': {
                'variant_id': best_by_efficiency[0],
                'mean_efficiency': best_by_efficiency[1].mean_efficiency_gain
            },
            'significance_tests': significance_tests,
            'recommendations': self._generate_recommendations(variant_stats, significance_tests)
        }
    
    def _calculate_variant_stats(self, variant_id: str, 
                               results: List[ExperimentResult]) -> ExperimentStats:
        """Calculate statistics for a variant"""
        if not results:
            return ExperimentStats(
                variant_id=variant_id,
                sample_size=0,
                mean_execution_time=0,
                std_execution_time=0,
                mean_expected_value=0,
                std_expected_value=0,
                mean_confidence=0,
                mean_efficiency_gain=0,
                success_rate=0,
                percentile_95_time=0,
                percentile_99_time=0
            )
        
        successful_results = [r for r in results if r.iterations_completed > 0]
        success_rate = len(successful_results) / len(results)
        
        if not successful_results:
            successful_results = results  # Use all results if none successful
        
        execution_times = [r.execution_time for r in successful_results]
        expected_values = [r.expected_value for r in successful_results]
        confidences = [r.confidence for r in successful_results]
        efficiency_gains = [r.efficiency_gain for r in successful_results]
        
        return ExperimentStats(
            variant_id=variant_id,
            sample_size=len(results),
            mean_execution_time=statistics.mean(execution_times),
            std_execution_time=statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            mean_expected_value=statistics.mean(expected_values),
            std_expected_value=statistics.stdev(expected_values) if len(expected_values) > 1 else 0,
            mean_confidence=statistics.mean(confidences),
            mean_efficiency_gain=statistics.mean(efficiency_gains),
            success_rate=success_rate,
            percentile_95_time=statistics.quantiles(execution_times, n=20)[18] if len(execution_times) >= 20 else max(execution_times),
            percentile_99_time=statistics.quantiles(execution_times, n=100)[98] if len(execution_times) >= 100 else max(execution_times)
        )
    
    def _run_significance_tests(self, variant_stats: Dict[str, ExperimentStats]) -> Dict[str, Any]:
        """Run basic statistical significance tests"""
        # Find control variant
        control_variant = None
        for variant_id, stats in variant_stats.items():
            # Assume control is either named 'control' or has 'control' in name
            if 'control' in variant_id.lower():
                control_variant = variant_id
                break
        
        if not control_variant:
            return {'error': 'No control variant found'}
        
        control_stats = variant_stats[control_variant]
        comparisons = {}
        
        for variant_id, stats in variant_stats.items():
            if variant_id == control_variant:
                continue
            
            # Simple effect size calculation (Cohen's d)
            pooled_std = ((control_stats.std_expected_value ** 2 + stats.std_expected_value ** 2) / 2) ** 0.5
            effect_size = (stats.mean_expected_value - control_stats.mean_expected_value) / pooled_std if pooled_std > 0 else 0
            
            # Performance improvement calculation
            value_improvement = ((stats.mean_expected_value - control_stats.mean_expected_value) / 
                               abs(control_stats.mean_expected_value)) if control_stats.mean_expected_value != 0 else 0
            
            time_improvement = ((control_stats.mean_execution_time - stats.mean_execution_time) /
                              control_stats.mean_execution_time) if control_stats.mean_execution_time > 0 else 0
            
            comparisons[variant_id] = {
                'effect_size': effect_size,
                'value_improvement_pct': value_improvement * 100,
                'time_improvement_pct': time_improvement * 100,
                'is_better_value': stats.mean_expected_value > control_stats.mean_expected_value,
                'is_faster': stats.mean_execution_time < control_stats.mean_execution_time
            }
        
        return {
            'control_variant': control_variant,
            'comparisons': comparisons
        }
    
    def _generate_recommendations(self, variant_stats: Dict[str, ExperimentStats],
                                significance_tests: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if 'comparisons' not in significance_tests:
            return ["Unable to generate recommendations without control variant"]
        
        comparisons = significance_tests['comparisons']
        
        # Find best performers
        best_value_variants = [
            variant_id for variant_id, comp in comparisons.items()
            if comp['value_improvement_pct'] > 5  # At least 5% improvement
        ]
        
        best_time_variants = [
            variant_id for variant_id, comp in comparisons.items()
            if comp['time_improvement_pct'] > 10  # At least 10% faster
        ]
        
        if best_value_variants:
            recommendations.append(
                f"Consider adopting {best_value_variants[0]} for better expected values "
                f"({comparisons[best_value_variants[0]]['value_improvement_pct']:.1f}% improvement)"
            )
        
        if best_time_variants:
            recommendations.append(
                f"Consider adopting {best_time_variants[0]} for faster execution "
                f"({comparisons[best_time_variants[0]]['time_improvement_pct']:.1f}% faster)"
            )
        
        # Check for variants that are both better
        win_win_variants = [
            variant_id for variant_id in comparisons.keys()
            if (comparisons[variant_id]['is_better_value'] and 
                comparisons[variant_id]['is_faster'])
        ]
        
        if win_win_variants:
            recommendations.append(
                f"Strongly recommend {win_win_variants[0]} - improves both value and speed"
            )
        
        if not recommendations:
            recommendations.append("Current control variant appears optimal based on this test")
        
        return recommendations
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get summary of experiment results"""
        if experiment_id not in self.results:
            return {'error': f'No results found for experiment {experiment_id}'}
        
        results = self.results[experiment_id]
        return self._analyze_results(experiment_id, results)