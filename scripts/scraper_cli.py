#!/usr/bin/env python3
"""
Koyfin Documentation Scraper CLI wrapper
"""

import sys
from pathlib import Path

# Add project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rex.documentation.cli import main

if __name__ == "__main__":
    sys.exit(main())