"""
Glean Integration Analysis Package
Production-ready code analysis using real Glean concepts with SCIP indexing
Compatible with Vercel deployment (no Docker required)
"""

# New Vercel-compatible Glean implementation  
from .glean_client import GleanClient
from .scip_indexer import PythonSCIPIndexer, SCIPSymbol, SCIPDocument, SCIPIndex, index_project_for_glean
from .angle_parser import AngleParser, AngleQuery, AngleQueryEngine, create_query, PYTHON_QUERIES
from .glean_storage import GleanStorage, GleanFact, initialize_python_schemas

# Legacy components (kept for backward compatibility)
from .glean_client import GleanQuery, CodeSymbol, CodeReference
try:
    from .code_analyzer import CodeAnalyzer
    from .impact_analyzer import ImpactAnalyzer
    from .architecture_validator import ArchitectureValidator, ArchitectureViolation, ViolationType
    from .realtime_analyzer import RealtimeCodeAnalyzer, AnalysisType, AnalysisResult
    from .cli_commands import GleanCLI
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
    'GleanClient',
    'PythonSCIPIndexer',
    'SCIPSymbol',
    'SCIPDocument',
    'SCIPIndex',
    'index_project_for_glean',
    'AngleParser',
    'AngleQuery', 
    'AngleQueryEngine',
    'create_query',
    'PYTHON_QUERIES',
    'GleanStorage',
    'GleanFact',
    'initialize_python_schemas',
    
    # Legacy components
    'CodeAnalyzer', 
    'ImpactAnalyzer',
    'ArchitectureValidator',
    'ArchitectureViolation',
    'ViolationType',
    'RealtimeCodeAnalyzer',
    'AnalysisType',
    'AnalysisResult',
    'GleanCLI'
]
