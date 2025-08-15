"""
Shared Types and Data Structures for Calculation Agent

Defines common data types used across all sub-skills
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal


class ComputationMethod(Enum):
    """Available computation methods"""
    SYMBOLIC = "symbolic"
    NUMERIC = "numeric"
    HYBRID = "hybrid"
    AUTO = "auto"


class CalculationType(Enum):
    """Types of calculations supported"""
    ARITHMETIC = "arithmetic"
    ALGEBRAIC = "algebraic"
    CALCULUS = "calculus"
    STATISTICAL = "statistical"
    FINANCIAL = "financial"
    LOGICAL = "logical"
    MATRIX = "matrix"
    OPTIMIZATION = "optimization"


class VerificationStatus(Enum):
    """Verification status for calculations"""
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"
    SKIPPED = "skipped"


@dataclass
class CalculationResult:
    """Result of a calculation with metadata"""
    result: Any
    method_used: ComputationMethod
    calculation_type: CalculationType
    symbolic_form: Optional[str] = None
    numeric_value: Optional[Union[float, complex, List]] = None
    precision: Optional[int] = None
    verification_status: VerificationStatus = VerificationStatus.PENDING
    steps: List[str] = None
    confidence: float = 1.0
    computation_time: float = 0.0
    error: Optional[str] = None
    

@dataclass
class VerificationResult:
    """Result of cross-verification between methods"""
    methods_compared: List[ComputationMethod]
    results_match: bool
    tolerance: float
    discrepancy: Optional[float] = None
    recommendation: str = ""
    confidence: float = 1.0
    timestamp: Optional[datetime] = None


@dataclass
class ReasoningStep:
    """Individual step in reasoning process"""
    step_id: str
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    method_used: str
    confidence: float = 1.0
    timestamp: Optional[datetime] = None


@dataclass
class CoordinationTask:
    """Task for A2A coordination"""
    task_id: str
    expression: str
    assigned_agent: str
    task_type: str
    dependencies: List[str] = None
    status: str = "pending"
    result: Optional[CalculationResult] = None
    created_at: Optional[datetime] = None


@dataclass
class FinancialParameters:
    """Parameters for financial calculations"""
    present_value: Optional[float] = None
    future_value: Optional[float] = None
    rate: Optional[float] = None
    periods: Optional[int] = None
    spot_price: Optional[float] = None
    strike_price: Optional[float] = None
    volatility: Optional[float] = None
    time_to_expiry: Optional[float] = None
    risk_free_rate: Optional[float] = None


@dataclass
class StatisticalData:
    """Container for statistical analysis data"""
    data: List[float]
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    var: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    percentiles: Optional[Dict[str, float]] = None