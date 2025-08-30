#!/usr/bin/env python3
"""
A2A Data Analysis Agent CLI - Statistical analysis and data validation
Comprehensive CLI interface for Data Analysis Agent with all capabilities
"""

import os
import sys
import asyncio
import click
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set environment variables for CLI
os.environ['ENVIRONMENT'] = 'development'
os.environ['SKIP_DB_INIT'] = 'true'

try:
    from cryptotrading.core.agents.data_analysis_agent import DataAnalysisAgent
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback minimal Data Analysis agent for CLI testing...")
    
    class FallbackDataAnalysisAgent:
        """Minimal Data Analysis agent for CLI testing when imports fail"""
        def __init__(self):
            self.agent_id = "data_analysis_agent"
            self.capabilities = [
                'validate_data_quality', 'analyze_data_distribution', 'compute_correlation_matrix',
                'detect_outliers', 'compute_rolling_statistics', 'statistical_analysis',
                'data_validation', 'quality_assessment'
            ]
            
        async def validate_data_quality(self, dataset, rules=None):
            """Mock data quality validation"""
            return {
                "dataset": dataset,
                "total_records": 1000,
                "valid_records": 950,
                "invalid_records": 50,
                "quality_score": 0.95,
                "issues": ["Missing values: 30", "Duplicate records: 20"],
                "rules_applied": rules or ["completeness", "uniqueness", "validity"],
                "timestamp": datetime.now().isoformat()
            }
            
        async def analyze_data_distribution(self, dataset, column=None):
            """Mock data distribution analysis"""
            return {
                "dataset": dataset,
                "column": column or "price",
                "distribution_type": "normal",
                "mean": 50000.0,
                "median": 49500.0,
                "std_dev": 5000.0,
                "skewness": 0.15,
                "kurtosis": -0.8,
                "percentiles": {"25%": 47000, "50%": 49500, "75%": 53000},
                "timestamp": datetime.now().isoformat()
            }
            
        async def compute_correlation_matrix(self, dataset, columns=None):
            """Mock correlation matrix computation"""
            columns = columns or ["price", "volume", "rsi", "macd"]
            correlations = {}
            
            for i, col1 in enumerate(columns):
                correlations[col1] = {}
                for j, col2 in enumerate(columns):
                    if i == j:
                        correlations[col1][col2] = 1.0
                    else:
                        correlations[col1][col2] = round(0.1 + (i + j) * 0.1, 2)
            
            return {
                "dataset": dataset,
                "columns": columns,
                "correlation_matrix": correlations,
                "strong_correlations": [("price", "volume", 0.75)],
                "timestamp": datetime.now().isoformat()
            }
            
        async def detect_outliers(self, dataset, method="iqr", threshold=1.5):
            """Mock outlier detection"""
            return {
                "dataset": dataset,
                "method": method,
                "threshold": threshold,
                "outliers_found": 15,
                "outlier_percentage": 1.5,
                "outlier_indices": [45, 123, 456, 789, 901],
                "outlier_values": [85000, 12000, 95000, 8000, 105000],
                "timestamp": datetime.now().isoformat()
            }
            
        async def compute_rolling_statistics(self, dataset, window=30, statistics=None):
            """Mock rolling statistics computation"""
            stats = statistics or ["mean", "std", "min", "max"]
            
            return {
                "dataset": dataset,
                "window": window,
                "statistics": stats,
                "results": {
                    "mean": [49500, 50200, 49800, 50100],
                    "std": [2500, 2800, 2300, 2600],
                    "min": [45000, 46000, 47000, 46500],
                    "max": [55000, 56000, 54000, 55500]
                },
                "periods_computed": 4,
                "timestamp": datetime.now().isoformat()
            }
            
        async def statistical_analysis(self, dataset, tests=None):
            """Mock statistical analysis"""
            tests = tests or ["normality", "stationarity", "autocorrelation"]
            
            return {
                "dataset": dataset,
                "tests_performed": tests,
                "results": {
                    "normality": {"test": "Shapiro-Wilk", "p_value": 0.15, "is_normal": True},
                    "stationarity": {"test": "ADF", "p_value": 0.02, "is_stationary": True},
                    "autocorrelation": {"lag_1": 0.85, "lag_5": 0.45, "lag_10": 0.25}
                },
                "summary": "Data shows normal distribution with strong stationarity",
                "timestamp": datetime.now().isoformat()
            }
            
        async def data_validation(self, dataset, schema=None):
            """Mock data validation against schema"""
            return {
                "dataset": dataset,
                "schema": schema or "default_schema",
                "validation_passed": True,
                "errors": [],
                "warnings": ["Column 'optional_field' has 10% missing values"],
                "validated_records": 1000,
                "timestamp": datetime.now().isoformat()
            }
            
        async def quality_assessment(self, dataset, dimensions=None):
            """Mock comprehensive quality assessment"""
            dimensions = dimensions or ["completeness", "accuracy", "consistency", "timeliness"]
            
            return {
                "dataset": dataset,
                "dimensions": dimensions,
                "scores": {
                    "completeness": 0.95,
                    "accuracy": 0.92,
                    "consistency": 0.88,
                    "timeliness": 0.90
                },
                "overall_score": 0.91,
                "grade": "A-",
                "recommendations": [
                    "Improve data consistency checks",
                    "Implement real-time validation"
                ],
                "timestamp": datetime.now().isoformat()
            }

# Global agent instance
agent = None

def get_agent():
    """Get or create agent instance"""
    global agent
    if agent is None:
        try:
            agent = DataAnalysisAgent()
        except:
            agent = FallbackDataAnalysisAgent()
    return agent

def async_command(f):
    """Decorator to run async commands"""
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper

@click.group()
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """A2A Data Analysis Agent CLI - Statistical analysis and data validation"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

@cli.command()
@click.argument('dataset')
@click.option('--rules', help='Comma-separated validation rules')
@click.pass_context
@async_command
async def validate(ctx, dataset, rules):
    """Validate data quality"""
    agent = get_agent()
    
    try:
        rules_list = rules.split(',') if rules else None
        result = await agent.validate_data_quality(dataset, rules_list)
        
        click.echo(f"✅ Data Quality Validation - {dataset}")
        click.echo("=" * 50)
        click.echo(f"Total Records: {result['total_records']}")
        click.echo(f"Valid Records: {result['valid_records']}")
        click.echo(f"Invalid Records: {result['invalid_records']}")
        click.echo(f"Quality Score: {result['quality_score']:.1%}")
        
        if result['issues']:
            click.echo("\nIssues Found:")
            for issue in result['issues']:
                click.echo(f"  • {issue}")
        
        if ctx.obj['verbose']:
            click.echo(f"\nRules Applied: {result['rules_applied']}")
            click.echo(f"Timestamp: {result['timestamp']}")
            
    except Exception as e:
        click.echo(f"Error validating data quality: {e}", err=True)

@cli.command()
@click.argument('dataset')
@click.option('--column', help='Specific column to analyze')
@click.pass_context
@async_command
async def distribution(ctx, dataset, column):
    """Analyze data distribution"""
    agent = get_agent()
    
    try:
        result = await agent.analyze_data_distribution(dataset, column)
        
        click.echo(f"📊 Distribution Analysis - {dataset}")
        click.echo("=" * 50)
        click.echo(f"Column: {result['column']}")
        click.echo(f"Distribution Type: {result['distribution_type']}")
        click.echo(f"Mean: {result['mean']:,.2f}")
        click.echo(f"Median: {result['median']:,.2f}")
        click.echo(f"Std Dev: {result['std_dev']:,.2f}")
        click.echo(f"Skewness: {result['skewness']:.3f}")
        click.echo(f"Kurtosis: {result['kurtosis']:.3f}")
        
        click.echo("\nPercentiles:")
        for pct, value in result['percentiles'].items():
            click.echo(f"  {pct}: {value:,.2f}")
        
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result['timestamp']}")
            
    except Exception as e:
        click.echo(f"Error analyzing distribution: {e}", err=True)

@cli.command()
@click.argument('dataset')
@click.option('--columns', help='Comma-separated column list')
@click.pass_context
@async_command
async def correlation(ctx, dataset, columns):
    """Compute correlation matrix"""
    agent = get_agent()
    
    try:
        columns_list = columns.split(',') if columns else None
        result = await agent.compute_correlation_matrix(dataset, columns_list)
        
        click.echo(f"🔗 Correlation Matrix - {dataset}")
        click.echo("=" * 50)
        
        # Display correlation matrix
        cols = result['columns']
        click.echo(f"{'':>10}", end='')
        for col in cols:
            click.echo(f"{col:>10}", end='')
        click.echo()
        
        for col1 in cols:
            click.echo(f"{col1:>10}", end='')
            for col2 in cols:
                corr = result['correlation_matrix'][col1][col2]
                click.echo(f"{corr:>10.2f}", end='')
            click.echo()
        
        if result['strong_correlations']:
            click.echo("\nStrong Correlations:")
            for col1, col2, corr in result['strong_correlations']:
                click.echo(f"  {col1} ↔ {col2}: {corr:.2f}")
        
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result['timestamp']}")
            
    except Exception as e:
        click.echo(f"Error computing correlation: {e}", err=True)

@cli.command()
@click.argument('dataset')
@click.option('--method', default='iqr', help='Outlier detection method (iqr, zscore)')
@click.option('--threshold', default=1.5, help='Threshold for outlier detection')
@click.pass_context
@async_command
async def outliers(ctx, dataset, method, threshold):
    """Detect outliers in dataset"""
    agent = get_agent()
    
    try:
        result = await agent.detect_outliers(dataset, method, threshold)
        
        click.echo(f"🎯 Outlier Detection - {dataset}")
        click.echo("=" * 50)
        click.echo(f"Method: {result['method'].upper()}")
        click.echo(f"Threshold: {result['threshold']}")
        click.echo(f"Outliers Found: {result['outliers_found']}")
        click.echo(f"Outlier Percentage: {result['outlier_percentage']:.1f}%")
        
        if ctx.obj['verbose'] and result['outlier_values']:
            click.echo(f"\nSample Outlier Values:")
            for i, value in enumerate(result['outlier_values'][:5]):
                idx = result['outlier_indices'][i]
                click.echo(f"  Index {idx}: {value:,.2f}")
            click.echo(f"\nTimestamp: {result['timestamp']}")
            
    except Exception as e:
        click.echo(f"Error detecting outliers: {e}", err=True)

@cli.command()
@click.argument('dataset')
@click.option('--window', default=30, help='Rolling window size')
@click.option('--statistics', help='Comma-separated statistics (mean,std,min,max)')
@click.pass_context
@async_command
async def rolling(ctx, dataset, window, statistics):
    """Compute rolling statistics"""
    agent = get_agent()
    
    try:
        stats_list = statistics.split(',') if statistics else None
        result = await agent.compute_rolling_statistics(dataset, window, stats_list)
        
        click.echo(f"📈 Rolling Statistics - {dataset}")
        click.echo("=" * 50)
        click.echo(f"Window Size: {result['window']}")
        click.echo(f"Periods Computed: {result['periods_computed']}")
        
        click.echo("\nStatistics:")
        for stat, values in result['results'].items():
            click.echo(f"  {stat.upper()}: {values[:3]}...")
        
        if ctx.obj['verbose']:
            click.echo(f"\nAll Statistics: {result['statistics']}")
            click.echo(f"Timestamp: {result['timestamp']}")
            
    except Exception as e:
        click.echo(f"Error computing rolling statistics: {e}", err=True)

@cli.command()
@click.argument('dataset')
@click.option('--tests', help='Comma-separated test list (normality,stationarity,autocorrelation)')
@click.pass_context
@async_command
async def statistical(ctx, dataset, tests):
    """Perform statistical analysis"""
    agent = get_agent()
    
    try:
        tests_list = tests.split(',') if tests else None
        result = await agent.statistical_analysis(dataset, tests_list)
        
        click.echo(f"📊 Statistical Analysis - {dataset}")
        click.echo("=" * 50)
        
        for test_name, test_result in result['results'].items():
            click.echo(f"\n{test_name.upper()}:")
            if isinstance(test_result, dict):
                for key, value in test_result.items():
                    click.echo(f"  {key}: {value}")
            else:
                click.echo(f"  Result: {test_result}")
        
        click.echo(f"\nSummary: {result['summary']}")
        
        if ctx.obj['verbose']:
            click.echo(f"\nTests Performed: {result['tests_performed']}")
            click.echo(f"Timestamp: {result['timestamp']}")
            
    except Exception as e:
        click.echo(f"Error in statistical analysis: {e}", err=True)

@cli.command()
@click.argument('dataset')
@click.option('--schema', help='Schema name for validation')
@click.pass_context
@async_command
async def schema(ctx, dataset, schema):
    """Validate data against schema"""
    agent = get_agent()
    
    try:
        result = await agent.data_validation(dataset, schema)
        
        click.echo(f"🔍 Schema Validation - {dataset}")
        click.echo("=" * 50)
        click.echo(f"Schema: {result['schema']}")
        click.echo(f"Validation: {'✅ PASSED' if result['validation_passed'] else '❌ FAILED'}")
        click.echo(f"Validated Records: {result['validated_records']}")
        
        if result['errors']:
            click.echo("\nErrors:")
            for error in result['errors']:
                click.echo(f"  ❌ {error}")
        
        if result['warnings']:
            click.echo("\nWarnings:")
            for warning in result['warnings']:
                click.echo(f"  ⚠️  {warning}")
        
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result['timestamp']}")
            
    except Exception as e:
        click.echo(f"Error in schema validation: {e}", err=True)

@cli.command()
@click.argument('dataset')
@click.option('--dimensions', help='Comma-separated quality dimensions')
@click.pass_context
@async_command
async def quality(ctx, dataset, dimensions):
    """Comprehensive quality assessment"""
    agent = get_agent()
    
    try:
        dimensions_list = dimensions.split(',') if dimensions else None
        result = await agent.quality_assessment(dataset, dimensions_list)
        
        click.echo(f"🏆 Quality Assessment - {dataset}")
        click.echo("=" * 50)
        click.echo(f"Overall Score: {result['overall_score']:.1%}")
        click.echo(f"Grade: {result['grade']}")
        
        click.echo("\nDimension Scores:")
        for dimension, score in result['scores'].items():
            click.echo(f"  {dimension.title()}: {score:.1%}")
        
        if result['recommendations']:
            click.echo("\nRecommendations:")
            for rec in result['recommendations']:
                click.echo(f"  • {rec}")
        
        if ctx.obj['verbose']:
            click.echo(f"\nDimensions: {result['dimensions']}")
            click.echo(f"Timestamp: {result['timestamp']}")
            
    except Exception as e:
        click.echo(f"Error in quality assessment: {e}", err=True)

@cli.command()
@click.pass_context
def capabilities(ctx):
    """List agent capabilities"""
    agent = get_agent()
    
    click.echo("🔧 Data Analysis Agent Capabilities:")
    click.echo()
    for i, capability in enumerate(agent.capabilities, 1):
        click.echo(f"{i:2d}. {capability.replace('_', ' ').title()}")

@cli.command()
@click.pass_context
def status(ctx):
    """Get agent status and health"""
    agent = get_agent()
    
    click.echo("🏥 Data Analysis Agent Status:")
    click.echo(f"Agent ID: {agent.agent_id}")
    click.echo(f"Capabilities: {len(agent.capabilities)}")
    click.echo("Status: ✅ ACTIVE")
    click.echo(f"Timestamp: {datetime.now().isoformat()}")

if __name__ == '__main__':
    cli()
