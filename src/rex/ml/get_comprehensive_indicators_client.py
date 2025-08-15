"""
Comprehensive Indicators Client Factory
Provides singleton access to comprehensive indicators client
"""

from .comprehensive_indicators_client import ComprehensiveIndicatorsClient

# Singleton instance
_comprehensive_indicators_client = None


def get_comprehensive_indicators_client() -> ComprehensiveIndicatorsClient:
    """Get or create the comprehensive indicators client instance"""
    global _comprehensive_indicators_client
    if _comprehensive_indicators_client is None:
        _comprehensive_indicators_client = ComprehensiveIndicatorsClient()
    return _comprehensive_indicators_client


# Backward compatibility
def get_comprehensive_metrics_client():
    """Backward compatibility wrapper"""
    return get_comprehensive_indicators_client()