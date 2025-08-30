#!/usr/bin/env python3
"""
RECONCILE 58 FACTORS WITH AVAILABLE DATA SOURCES
Identify which factors can be calculated with current data loaders
"""

import sys

sys.path.append("src")

from typing import Dict, List, Set

from cryptotrading.core.factors import ALL_FACTORS, DataSource, FactorCategory

# Data sources we actually have implemented
IMPLEMENTED_SOURCES = {
    DataSource.YAHOO,  # Yahoo Finance - fully implemented
    DataSource.FRED,  # FRED - fully implemented
}

# Data we have loaded in the database
LOADED_DATA = {
    "crypto": ["BTC-USD", "ETH-USD"],  # 2 years of daily OHLCV from Yahoo
    "macro": ["DGS10", "WALCL", "RRPONTSYD", "WTREGEN"],  # FRED data
}


def can_calculate_factor(factor) -> bool:
    """Check if we can calculate a factor with available data sources"""
    # Check if any required source is implemented
    for source in factor.required_sources:
        if source in IMPLEMENTED_SOURCES:
            return True
    return False


def get_calculation_method(factor) -> str:
    """Determine how to calculate a factor with available data"""
    if DataSource.YAHOO in factor.required_sources:
        return "Yahoo Finance"
    elif DataSource.FRED in factor.required_sources:
        return "FRED"
    elif factor.is_derived:
        # Check if dependencies can be calculated
        deps_available = all(
            any(f.name == dep and can_calculate_factor(f) for f in ALL_FACTORS)
            for dep in factor.dependencies
        )
        if deps_available:
            return "Derived from available factors"
    return "NOT AVAILABLE"


def analyze_factors():
    """Analyze all 58 factors for data availability"""
    print("📊 FACTOR RECONCILIATION WITH AVAILABLE DATA SOURCES")
    print("=" * 80)

    results = {"calculable": [], "not_calculable": [], "partially_calculable": []}

    # Analyze each factor
    for i, factor in enumerate(ALL_FACTORS, 1):
        can_calc = can_calculate_factor(factor)
        method = get_calculation_method(factor)

        factor_info = {
            "number": i,
            "name": factor.name,
            "category": factor.category.value,
            "required_sources": [s.value for s in factor.required_sources],
            "is_derived": factor.is_derived,
            "dependencies": factor.dependencies,
            "calculation_method": method,
        }

        if method != "NOT AVAILABLE":
            results["calculable"].append(factor_info)
        else:
            results["not_calculable"].append(factor_info)

    # Print summary by category
    print("\n📈 SUMMARY BY CATEGORY")
    print("-" * 80)

    category_stats = {}
    for category in FactorCategory:
        factors_in_cat = [f for f in ALL_FACTORS if f.category == category]
        calculable_in_cat = [
            f for f in factors_in_cat if get_calculation_method(f) != "NOT AVAILABLE"
        ]

        category_stats[category.value] = {
            "total": len(factors_in_cat),
            "calculable": len(calculable_in_cat),
            "percentage": (len(calculable_in_cat) / len(factors_in_cat) * 100)
            if factors_in_cat
            else 0,
        }

        print(
            f"{category.value.upper():<20}: {len(calculable_in_cat):2d}/{len(factors_in_cat):2d} "
            f"({category_stats[category.value]['percentage']:5.1f}%)"
        )

    # Print calculable factors
    print("\n✅ CALCULABLE FACTORS WITH CURRENT DATA")
    print("-" * 80)
    print(f"{'#':<3} {'Factor Name':<25} {'Category':<15} {'Data Source':<20}")
    print("-" * 80)

    for factor in results["calculable"]:
        print(
            f"{factor['number']:<3} {factor['name']:<25} {factor['category']:<15} {factor['calculation_method']:<20}"
        )

    # Print factors that need additional data sources
    print("\n❌ FACTORS REQUIRING UNAVAILABLE DATA SOURCES")
    print("-" * 80)
    print(f"{'#':<3} {'Factor Name':<25} {'Category':<15} {'Required Sources':<30}")
    print("-" * 80)

    for factor in results["not_calculable"]:
        sources_str = ", ".join(factor["required_sources"])
        print(
            f"{factor['number']:<3} {factor['name']:<25} {factor['category']:<15} {sources_str:<30}"
        )

    # Data source usage summary
    print("\n📊 DATA SOURCE REQUIREMENTS")
    print("-" * 80)

    source_usage = {}
    for factor in ALL_FACTORS:
        for source in factor.required_sources:
            if source not in source_usage:
                source_usage[source] = 0
            source_usage[source] += 1

    for source, count in sorted(source_usage.items(), key=lambda x: x[1], reverse=True):
        status = "✅ AVAILABLE" if source in IMPLEMENTED_SOURCES else "❌ NOT AVAILABLE"
        print(f"{source.value:<20}: {count:2d} factors - {status}")

    # Overall summary
    print("\n" + "=" * 80)
    print("🎯 OVERALL SUMMARY")
    print("=" * 80)

    total_factors = len(ALL_FACTORS)
    calculable_count = len(results["calculable"])
    percentage = (calculable_count / total_factors) * 100

    print(f"Total Factors: {total_factors}")
    print(f"Calculable with current data: {calculable_count} ({percentage:.1f}%)")
    print(f"Requiring additional sources: {len(results['not_calculable'])} ({100-percentage:.1f}%)")

    print("\n📝 RECOMMENDATIONS:")
    print("-" * 40)

    if percentage < 50:
        print("⚠️  Less than 50% of factors can be calculated with current data sources!")
        print("\nPriority data sources to add:")

        # Find most impactful sources to add
        source_impact = {}
        for factor in results["not_calculable"]:
            for source in factor["required_sources"]:
                if source not in IMPLEMENTED_SOURCES:
                    if source not in source_impact:
                        source_impact[source] = 0
                    source_impact[source] += 1

        for source, impact in sorted(source_impact.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  - {source}: Would enable {impact} additional factors")

    print("\n✅ Currently supported factor types:")
    print("  - Price-based factors using Yahoo Finance OHLCV data")
    print("  - Technical indicators calculable from price data")
    print("  - Macro correlations using FRED economic data")
    print("  - Simple volatility measures from price returns")

    return results


def generate_supported_factors_list():
    """Generate a list of factors we can actually support"""
    print("\n" + "=" * 80)
    print("📋 GENERATING SUPPORTED FACTORS LIST")
    print("=" * 80)

    supported_factors = []

    for factor in ALL_FACTORS:
        if get_calculation_method(factor) != "NOT AVAILABLE":
            supported_factors.append(
                {
                    "name": factor.name,
                    "category": factor.category.value,
                    "description": factor.description,
                    "data_source": get_calculation_method(factor),
                    "calculation_possible": True,
                }
            )

    # Write to file
    import json

    with open("supported_factors.json", "w") as f:
        json.dump(supported_factors, f, indent=2)

    print(f"✅ Wrote {len(supported_factors)} supported factors to supported_factors.json")

    return supported_factors


if __name__ == "__main__":
    # Run reconciliation
    results = analyze_factors()

    # Generate supported factors list
    supported = generate_supported_factors_list()

    print("\n🏁 RECONCILIATION COMPLETE")
    print(f"   We can currently calculate {len(results['calculable'])} out of 58 factors")
    print(f"   This represents {len(results['calculable'])/58*100:.1f}% coverage")
