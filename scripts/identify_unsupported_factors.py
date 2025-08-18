#!/usr/bin/env python3
"""
Identify factors that depend on unavailable data sources
"""

import sys
sys.path.append('src')

from cryptotrading.core.factors.factor_definitions import ALL_FACTORS, DataSource

# Only data sources we actually have implemented
AVAILABLE_SOURCES = {DataSource.YAHOO, DataSource.FRED}

# Categorize factors
supported_factors = []
unsupported_factors = []

for factor in ALL_FACTORS:
    # Check if ANY required source is available
    has_available_source = any(source in AVAILABLE_SOURCES for source in factor.required_sources)
    
    # If no required sources, check if it's derived from available factors
    if not factor.required_sources and factor.is_derived:
        # Check dependencies recursively - simplified check
        has_available_source = True  # Assume derived factors can be calculated
    
    if has_available_source or not factor.required_sources:
        supported_factors.append(factor)
    else:
        unsupported_factors.append(factor)

print("FACTORS TO REMOVE (No available data sources):")
print("=" * 80)

for i, factor in enumerate(unsupported_factors, 1):
    sources = [s.value for s in factor.required_sources]
    print(f"{i}. {factor.name} - Requires: {', '.join(sources)}")

print(f"\nTotal factors to remove: {len(unsupported_factors)}")
print(f"Remaining supported factors: {len(supported_factors)}")

# Generate list of factor names to remove
print("\n\nFACTOR NAMES TO REMOVE:")
print("-" * 40)
for factor in unsupported_factors:
    print(f'    "{factor.name}",')