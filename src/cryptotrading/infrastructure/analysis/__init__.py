"""
Glean Integration Analysis Package
Production-ready code analysis using real Glean concepts with SCIP indexing
Compatible with Vercel deployment (no Docker required)
"""

from .angle_parser import PYTHON_QUERIES, AngleParser, AngleQuery, AngleQueryEngine, create_query

# Legacy components (kept for backward compatibility)
# New Vercel-compatible Glean implementation
from .glean_client import CodeReference, CodeSymbol, GleanClient, GleanQuery
from .glean_storage import GleanFact, GleanStorage, initialize_python_schemas
from .scip_indexer import (
    PythonSCIPIndexer,
    SCIPDocument,
    SCIPIndex,
    SCIPSymbol,
    index_project_for_glean,
)

try:
    from .architecture_validator import ArchitectureValidator, ArchitectureViolation, ViolationType
    from .cli_commands import GleanCLI
    from .code_analyzer import CodeAnalyzer
    from .impact_analyzer import ImpactAnalyzer
    from .realtime_analyzer import AnalysisResult, AnalysisType, RealtimeCodeAnalyzer
except ImportError as e:
    # Handle missing legacy components gracefully
    import logging

    logging.warning(f"Some legacy components unavailable: {e}")
    CodeAnalyzer = None
    ImpactAnalyzer = None
    ArchitectureValidator = None
    ArchitectureViolation = None
    ViolationType = None
    RealtimeCodeAnalyzer = None
    AnalysisType = None
    AnalysisResult = None
    GleanCLI = None

__all__ = [
    # New Vercel-compatible Glean implementation
    "GleanClient",
    "PythonSCIPIndexer",
    "SCIPSymbol",
    "SCIPDocument",
    "SCIPIndex",
    "index_project_for_glean",
    "AngleParser",
    "AngleQuery",
    "AngleQueryEngine",
    "create_query",
    "PYTHON_QUERIES",
    "GleanStorage",
    "GleanFact",
    "initialize_python_schemas",
    # Legacy components
    "CodeAnalyzer",
    "ImpactAnalyzer",
    "ArchitectureValidator",
    "ArchitectureViolation",
    "ViolationType",
    "RealtimeCodeAnalyzer",
    "AnalysisType",
    "AnalysisResult",
    "GleanCLI",
]
