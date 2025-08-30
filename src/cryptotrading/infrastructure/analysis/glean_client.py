"""
Glean Client - Main interface
Uses the Vercel-compatible implementation with SCIP indexing
"""

from .angle_parser import AngleQuery as GleanQuery
from .scip_indexer import SCIPSymbol as CodeSymbol
from .vercel_glean_client import VercelGleanClient as GleanClient


class CodeReference:
    """Simple code reference structure for backward compatibility"""

    def __init__(self, file_path: str, line: int, column: int = 0, name: str = ""):
        self.file_path = file_path
        self.line = line
        self.column = column
        self.name = name


__all__ = ["GleanClient", "GleanQuery", "CodeSymbol", "CodeReference"]
