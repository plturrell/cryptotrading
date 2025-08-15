"""
Validation script for enhanced professional trading indicators
Checks availability and data quality of all configured tickers on Yahoo Finance
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from comprehensive_indicators_client import ComprehensiveIndicatorsClient
from professional_trading_config import ProfessionalTradingConfig, MarketRegime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_all_indicators():
    """Validate all indicators in the enhanced configuration"""
    logger.info("Starting validation of enhanced professional trading indicators...")
    
    # Initialize client
    client = ComprehensiveIndicatorsClient()
    
    # Validate all tickers
    logger.info(f"Validating {len(client.COMPREHENSIVE_INDICATORS)} indicators...")
    validation_results = client.validate_all_tickers()
    
    # Summary statistics
    total = len(client.COMPREHENSIVE_INDICATORS)
    available = len(validation_results['available'])
    unavailable = len(validation_results['unavailable'])
    errors = len(validation_results['errors'])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total indicators: {total}")
    logger.info(f"Available: {available} ({available/total*100:.1f}%)")
    logger.info(f"Unavailable: {unavailable} ({unavailable/total*100:.1f}%)")
    logger.info(f"Errors: {errors} ({errors/total*100:.1f}%)")
    logger.info(f"{'='*60}\n")
    
    # Detailed results by category
    categories = {}
    for symbol, info in client.COMPREHENSIVE_INDICATORS.items():
        category = info['category']
        if category not in categories:
            categories[category] = {'total': 0, 'available': 0, 'indicators': []}
        categories[category]['total'] += 1
        
        # Check if available
        is_available = any(v['symbol'] == symbol for v in validation_results['available'])
        if is_available:
            categories[category]['available'] += 1
        categories[category]['indicators'].append({
            'symbol': symbol,
            'name': info['name'],
            'available': is_available
        })
    
    logger.info("RESULTS BY CATEGORY:")
    logger.info("-" * 60)
    for category, data in sorted(categories.items()):
        availability_rate = data['available'] / data['total'] * 100
        logger.info(f"\n{category.upper()} ({data['available']}/{data['total']} = {availability_rate:.1f}% available)")
        for indicator in data['indicators']:
            status = "✓" if indicator['available'] else "✗"
            logger.info(f"  {status} {indicator['symbol']:12} - {indicator['name']}")
    
    # Check data quality for available tickers
    logger.info(f"\n{'='*60}")
    logger.info("DATA QUALITY CHECK")
    logger.info(f"{'='*60}")
    
    quality_issues = []
    for ticker_info in validation_results['available']:
        symbol = ticker_info['symbol']
        
        # Check data points
        if ticker_info.get('data_points_1y', 0) < 200:  # Less than 200 trading days
            quality_issues.append({
                'symbol': symbol,
                'issue': 'Insufficient 1-year data',
                'data_points': ticker_info.get('data_points_1y', 0)
            })
        
        # Check if it has volume data (for ETFs)
        if not ticker_info.get('has_volume', True):
            quality_issues.append({
                'symbol': symbol,
                'issue': 'No volume data',
                'type': 'Index or special instrument'
            })
    
    if quality_issues:
        logger.info("\nPotential data quality considerations:")
        for issue in quality_issues:
            logger.info(f"  • {issue['symbol']}: {issue['issue']}")
    else:
        logger.info("\n✓ All available tickers have good data quality")
    
    # Test institutional indicator sets
    logger.info(f"\n{'='*60}")
    logger.info("INSTITUTIONAL INDICATOR SETS VALIDATION")
    logger.info(f"{'='*60}")
    
    all_sets = ProfessionalTradingConfig.get_all_indicator_sets()
    
    for set_name, indicator_set in all_sets.items():
        logger.info(f"\n{set_name.upper()}: {indicator_set.name}")
        logger.info(f"Institution: {indicator_set.institutional_reference}")
        logger.info(f"Strategy: {indicator_set.strategy_type.value}")
        
        # Check availability of all symbols in the set
        available_count = 0
        for symbol in indicator_set.symbols:
            is_available = any(v['symbol'] == symbol for v in validation_results['available'])
            if is_available:
                available_count += 1
            status = "✓" if is_available else "✗"
            weight = indicator_set.weights.get(symbol, 0)
            logger.info(f"  {status} {symbol:12} (weight: {weight:.2%})")
        
        availability_rate = available_count / len(indicator_set.symbols) * 100
        logger.info(f"  Set availability: {available_count}/{len(indicator_set.symbols)} = {availability_rate:.1f}%")
    
    # Test regime indicators
    logger.info(f"\n{'='*60}")
    logger.info("MARKET REGIME INDICATORS VALIDATION")
    logger.info(f"{'='*60}")
    
    for regime in MarketRegime:
        regime_indicators = ProfessionalTradingConfig.get_regime_indicators(regime)
        logger.info(f"\n{regime.value.upper()} regime indicators:")
        
        available_count = 0
        for symbol in regime_indicators:
            is_available = any(v['symbol'] == symbol for v in validation_results['available'])
            if is_available:
                available_count += 1
            status = "✓" if is_available else "✗"
            logger.info(f"  {status} {symbol}")
        
        if regime_indicators:
            availability_rate = available_count / len(regime_indicators) * 100
            logger.info(f"  Regime availability: {available_count}/{len(regime_indicators)} = {availability_rate:.1f}%")
    
    # Save detailed results
    output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'validation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': total,
                'available': available,
                'unavailable': unavailable,
                'errors': errors
            },
            'validation_results': validation_results,
            'categories': categories,
            'quality_issues': quality_issues
        }, f, indent=2)
    
    logger.info(f"\n✓ Detailed results saved to: {output_path}")
    
    return validation_results


def test_sample_data_retrieval():
    """Test actual data retrieval for a few key indicators"""
    logger.info(f"\n{'='*60}")
    logger.info("SAMPLE DATA RETRIEVAL TEST")
    logger.info(f"{'='*60}")
    
    client = ComprehensiveIndicatorsClient()
    
    # Test key indicators from each category
    test_symbols = [
        '^VIX',    # Volatility
        '^TNX',    # Treasury yield
        'TIP',     # TIPS ETF
        'DX-Y.NYB',# Dollar index
        'XLK',     # Tech sector
        'GC=F',    # Gold futures
        'EEM',     # Emerging markets
        'UUP',     # Dollar bull ETF
        '^SKEW'    # Tail risk
    ]
    
    for symbol in test_symbols:
        logger.info(f"\nTesting {symbol}...")
        try:
            # Get 5 days of data
            data = client.get_comprehensive_data(symbol, days_back=5)
            if not data.empty:
                latest = data.iloc[-1]
                logger.info(f"  ✓ Successfully retrieved {len(data)} data points")
                logger.info(f"    Latest Close: {latest['Close']:.2f}")
                logger.info(f"    Date: {data.index[-1].strftime('%Y-%m-%d')}")
                if 'Volume' in data.columns and data['Volume'].sum() > 0:
                    logger.info(f"    Volume: {latest['Volume']:,.0f}")
            else:
                logger.info(f"  ✗ No data returned")
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
    
    # Test crypto predictors
    logger.info(f"\n{'='*60}")
    logger.info("CRYPTO PREDICTOR TEST")
    logger.info(f"{'='*60}")
    
    test_crypto = 'BTC'
    logger.info(f"\nTesting predictors for {test_crypto}...")
    
    try:
        predictors = client.get_comprehensive_predictors_for_crypto(test_crypto, days_back=5)
        logger.info(f"Successfully loaded {len(predictors)} predictors for {test_crypto}")
        for symbol, data in predictors.items():
            logger.info(f"  ✓ {symbol}: {len(data)} data points")
    except Exception as e:
        logger.error(f"Error loading predictors: {e}")


if __name__ == "__main__":
    # Run validation
    validation_results = validate_all_indicators()
    
    # Run sample data test
    test_sample_data_retrieval()
    
    logger.info("\n✓ Validation complete!")