"""
Technical Analysis Skills Package
Modular technical analysis skills implemented as STRAND tools for crypto trading
"""

from .skill_1_momentum_indicators import create_basic_indicators_tools
from .skill_2_momentum_volatility import create_momentum_volatility_tools
from .skill_3_volume_analysis import create_volume_analysis_tools
from .skill_4_support_resistance import create_support_resistance_tools
from .skill_5_chart_patterns import create_chart_pattern_tools
from .skill_6_harmonic_patterns import create_advanced_pattern_tools
from .skill_7_comprehensive_system import create_comprehensive_system_tools
from .skill_8_dashboard import create_dashboard_tools
from .technical_analysis_agent import TechnicalAnalysisAgent, create_technical_analysis_agent
from .grok_insights_integration import create_grok_insights_tools
from .visualization_engine import create_visualization_tools
from .performance_optimization import create_performance_tools

__all__ = [
    'create_basic_indicators_tools',
    'create_momentum_volatility_tools', 
    'create_volume_analysis_tools',
    'create_support_resistance_tools',
    'create_chart_pattern_tools',
    'create_advanced_pattern_tools',
    'create_comprehensive_system_tools',
    'create_dashboard_tools',
    'create_grok_insights_tools',
    'create_visualization_tools',
    'create_performance_tools',
    'TechnicalAnalysisAgent',
    'create_technical_analysis_agent'
]

# Export skill creation functions for STRAND tool registration
def get_all_ta_skills():
    """
    Get all technical analysis skill creators for STRAND framework integration
    
    Returns:
        Dictionary mapping skill names to their tool creation functions
    """
    return {
        'basic_indicators': create_basic_indicators_tools,
        'momentum_volatility': create_momentum_volatility_tools,
        'volume_analysis': create_volume_analysis_tools,
        'support_resistance': create_support_resistance_tools,
        'chart_patterns': create_chart_pattern_tools,
        'advanced_patterns': create_advanced_pattern_tools,
        'comprehensive_system': create_comprehensive_system_tools,
        'dashboard': create_dashboard_tools
    }
