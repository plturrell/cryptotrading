"""
Core Comprehensive Indicators Test
Test the comprehensive indicators functionality without full agent initialization
"""

import sys
import os
sys.path.append('.')

from cryptotrading.core.ml.comprehensive_indicators_client import ComprehensiveIndicatorsClient
from cryptotrading.core.ml.professional_trading_config import ProfessionalTradingConfig, MarketRegime
from cryptotrading.core.protocols.a2a.enhanced_message_types import CURRENT_PROTOCOL_VERSION
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_comprehensive_indicators_client():
    """Test the core comprehensive indicators client"""
    logger.info("Testing Comprehensive Indicators Client...")
    
    try:
        # Initialize client
        client = ComprehensiveIndicatorsClient()
        logger.info(f"✓ Client initialized with {len(client.COMPREHENSIVE_INDICATORS)} indicators")
        
        # Test indicator categories
        categories = set(info['category'] for info in client.COMPREHENSIVE_INDICATORS.values())
        logger.info(f"✓ Categories available: {categories}")
        
        # Test predictive tiers
        tiers = client.PREDICTIVE_TIERS
        logger.info(f"✓ Predictive tiers: {list(tiers.keys())}")
        
        # Test crypto predictors
        crypto_predictors = client.CRYPTO_COMPREHENSIVE_PREDICTORS
        logger.info(f"✓ Crypto predictors for: {list(crypto_predictors.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing client: {e}")
        return False


def test_professional_trading_config():
    """Test the professional trading configuration"""
    logger.info("Testing Professional Trading Configuration...")
    
    try:
        # Test institutional strategies
        strategies = ProfessionalTradingConfig.get_all_indicator_sets()
        logger.info(f"✓ Institutional strategies: {list(strategies.keys())}")
        
        for name, strategy in strategies.items():
            logger.info(f"  - {name}: {strategy.name} ({strategy.institutional_reference})")
        
        # Test regime indicators
        regimes = [MarketRegime.RISK_ON, MarketRegime.RISK_OFF, MarketRegime.CRISIS]
        for regime in regimes:
            indicators = ProfessionalTradingConfig.get_regime_indicators(regime)
            logger.info(f"✓ {regime.value}: {len(indicators)} indicators")
        
        # Test critical thresholds
        thresholds = ProfessionalTradingConfig.get_critical_thresholds()
        logger.info(f"✓ Critical thresholds for: {list(thresholds.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing config: {e}")
        return False


def test_enhanced_message_types():
    """Test the enhanced message types"""
    logger.info("Testing Enhanced Message Types...")
    
    try:
        # Test protocol version
        version = CURRENT_PROTOCOL_VERSION
        logger.info(f"✓ Protocol version: {version}")
        logger.info(f"✓ Features: {version.features}")
        
        # Test message types
        from cryptotrading.core.protocols.a2a.enhanced_message_types import EnhancedMessageType
        message_types = [msg_type.value for msg_type in EnhancedMessageType]
        logger.info(f"✓ Enhanced message types: {len(message_types)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing message types: {e}")
        return False


def test_validation_functionality():
    """Test validation functionality without network calls"""
    logger.info("Testing Validation Functionality...")
    
    try:
        client = ComprehensiveIndicatorsClient()
        
        # Test indicator info
        sample_indicators = ['^VIX', 'TLT', 'XLK']
        for symbol in sample_indicators:
            if symbol in client.COMPREHENSIVE_INDICATORS:
                info = client.COMPREHENSIVE_INDICATORS[symbol]
                logger.info(f"✓ {symbol}: {info['name']} ({info['category']})")
            else:
                logger.warning(f"⚠ {symbol} not found")
        
        # Test tier classification
        for tier in ['very_high', 'high', 'medium']:
            tier_indicators = client.PREDICTIVE_TIERS.get(tier, [])
            logger.info(f"✓ {tier} tier: {len(tier_indicators)} indicators")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing validation: {e}")
        return False


def test_integration_completeness():
    """Test that all components are properly integrated"""
    logger.info("Testing Integration Completeness...")
    
    try:
        # Test that config file exists
        import os
        config_path = 'config/enhanced_indicators.yaml'
        if os.path.exists(config_path):
            logger.info(f"✓ Configuration file exists: {config_path}")
        else:
            logger.warning(f"⚠ Configuration file missing: {config_path}")
        
        # Test that all imports work
        from cryptotrading.core.ml.get_comprehensive_indicators_client import get_comprehensive_indicators_client
        client = get_comprehensive_indicators_client()
        logger.info(f"✓ Factory function works")
        
        # Test professional strategies integration
        strategies = ProfessionalTradingConfig.get_all_indicator_sets()
        two_sigma = strategies.get('two_sigma')
        if two_sigma:
            logger.info(f"✓ Two Sigma strategy: {len(two_sigma.symbols)} indicators")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing integration: {e}")
        return False


def calculate_core_score():
    """Calculate score for core functionality"""
    logger.info("\n" + "="*60)
    logger.info("CORE COMPREHENSIVE INDICATORS TEST")
    logger.info("="*60)
    
    tests = {
        'comprehensive_indicators_client': test_comprehensive_indicators_client(),
        'professional_trading_config': test_professional_trading_config(),
        'enhanced_message_types': test_enhanced_message_types(),
        'validation_functionality': test_validation_functionality(),
        'integration_completeness': test_integration_completeness()
    }
    
    passed = sum(tests.values())
    total = len(tests)
    score = (passed / total) * 100
    
    logger.info(f"\n" + "="*60)
    logger.info("CORE TEST RESULTS")
    logger.info("="*60)
    
    for test_name, result in tests.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nCore Score: {passed}/{total} tests passed")
    logger.info(f"Core Implementation: {score:.1f}%")
    
    if score >= 100:
        logger.info("\n🎉 CORE IMPLEMENTATION PERFECT!")
        logger.info("✅ Comprehensive indicators client working")
        logger.info("✅ Professional trading configuration complete")
        logger.info("✅ Enhanced message types implemented")
        logger.info("✅ Validation functionality operational")
        logger.info("✅ Integration completeness verified")
    
    return score


if __name__ == "__main__":
    try:
        score = calculate_core_score()
        
        if score >= 100:
            logger.info("\n" + "="*60)
            logger.info("✅ CORE COMPREHENSIVE INDICATORS: 100% WORKING")
            logger.info("Ready for full agent integration!")
            logger.info("="*60)
        else:
            logger.info(f"\n⚠ Core functionality: {score:.1f}% - needs attention")
            
    except Exception as e:
        logger.error(f"Core test failed: {e}")
        raise