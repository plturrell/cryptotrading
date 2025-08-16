"""
Architecture Validator - Production implementation for architectural constraint validation
Validates architectural rules and constraints using Glean integration
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import json

from .glean_client import GleanClient

logger = logging.getLogger(__name__)

class ViolationType(Enum):
    LAYERING_VIOLATION = "layering_violation"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    FORBIDDEN_DEPENDENCY = "forbidden_dependency"
    NAMING_CONVENTION = "naming_convention"
    MODULE_SIZE = "module_size"
    COUPLING_VIOLATION = "coupling_violation"

@dataclass
class ArchitectureViolation:
    """Represents an architectural constraint violation"""
    violation_type: ViolationType
    severity: str  # critical, high, medium, low
    description: str
    source: str
    target: Optional[str] = None
    file_path: Optional[str] = None
    line: Optional[int] = None
    recommendation: Optional[str] = None

class ArchitectureValidator:
    """Validates architectural constraints and rules"""
    
    def __init__(self, glean_client: GleanClient):
        self.glean = glean_client
        self.violations: List[ArchitectureViolation] = []
        
        # Define architectural layers (from high to low level)
        self.layers = [
            "cryptotrading.webapp",           # Presentation layer
            "cryptotrading.api",              # API layer  
            "cryptotrading.core.agents",      # Application layer
            "cryptotrading.core.protocols",   # Protocol layer
            "cryptotrading.core.ml",          # ML/AI layer
            "cryptotrading.data",             # Data access layer
            "cryptotrading.infrastructure",   # Infrastructure layer
            "cryptotrading.utils"             # Utility layer
        ]
        
        # Define forbidden dependencies
        self.forbidden_dependencies = {
            # Core should not depend on infrastructure
            "cryptotrading.core": ["cryptotrading.infrastructure"],
            # Data layer should not depend on core
            "cryptotrading.data": ["cryptotrading.core.agents", "cryptotrading.core.protocols"],
            # Utils should not depend on anything except standard library
            "cryptotrading.utils": ["cryptotrading.core", "cryptotrading.data", "cryptotrading.infrastructure"],
            # Infrastructure should not depend on core business logic
            "cryptotrading.infrastructure": ["cryptotrading.core.agents", "cryptotrading.core.protocols"]
        }
        
        # Module size limits
        self.module_size_limits = {
            "max_functions_per_module": 20,
            "max_classes_per_module": 10,
            "max_lines_per_file": 500
        }
        
        # Coupling limits
        self.coupling_limits = {
            "max_dependencies_per_module": 15,
            "max_dependents_per_module": 20
        }
    
    async def validate_architecture(self) -> List[ArchitectureViolation]:
        """Run complete architecture validation"""
        logger.info("Starting comprehensive architecture validation")
        
        self.violations = []
        
        # Run all validation checks
        await self._validate_layering()
        await self._validate_circular_dependencies()
        await self._validate_forbidden_dependencies()
        await self._validate_naming_conventions()
        await self._validate_module_sizes()
        await self._validate_coupling()
        
        logger.info(f"Architecture validation complete: {len(self.violations)} violations found")
        return self.violations
    
    async def _validate_layering(self):
        """Validate layering constraints"""
        logger.info("Validating layering constraints")
        
        layer_indices = {layer: i for i, layer in enumerate(self.layers)}
        
        # Check each module's dependencies
        for layer in self.layers:
            try:
                deps = await self.glean.get_dependencies(layer, depth=1)
                
                for dep in deps["direct"]:
                    # Skip external dependencies
                    if not dep.startswith("cryptotrading"):
                        continue
                    
                    # Find which layer this dependency belongs to
                    dep_layer = self._find_layer_for_module(dep)
                    if not dep_layer:
                        continue
                    
                    # Check if dependency violates layering
                    current_layer_index = layer_indices.get(layer, -1)
                    dep_layer_index = layer_indices.get(dep_layer, -1)
                    
                    if dep_layer_index != -1 and current_layer_index != -1:
                        if dep_layer_index < current_layer_index:
                            # Higher layer depending on lower layer - this is a violation
                            self.violations.append(ArchitectureViolation(
                                violation_type=ViolationType.LAYERING_VIOLATION,
                                severity="high",
                                description=f"Layer {layer} depends on higher layer {dep_layer}",
                                source=layer,
                                target=dep,
                                recommendation=f"Refactor to remove dependency or move {dep} to a lower layer"
                            ))
            
            except Exception as e:
                logger.warning(f"Error validating layering for {layer}: {e}")
    
    def _find_layer_for_module(self, module: str) -> Optional[str]:
        """Find which architectural layer a module belongs to"""
        for layer in self.layers:
            if module.startswith(layer):
                return layer
        return None
    
    async def _validate_circular_dependencies(self):
        """Validate no circular dependencies exist"""
        logger.info("Validating circular dependencies")
        
        try:
            circular_deps = await self.glean.detect_circular_dependencies()
            
            for cycle_info in circular_deps:
                modules = cycle_info.get("modules", [])
                if len(modules) > 1:
                    self.violations.append(ArchitectureViolation(
                        violation_type=ViolationType.CIRCULAR_DEPENDENCY,
                        severity="critical",
                        description=f"Circular dependency detected: {' -> '.join(modules)} -> {modules[0]}",
                        source=modules[0],
                        target=modules[-1],
                        recommendation="Break the cycle by introducing an interface or moving shared code to a common module"
                    ))
        
        except Exception as e:
            logger.warning(f"Error detecting circular dependencies: {e}")
    
    async def _validate_forbidden_dependencies(self):
        """Validate forbidden dependencies are not present"""
        logger.info("Validating forbidden dependencies")
        
        for source_pattern, forbidden_patterns in self.forbidden_dependencies.items():
            try:
                # Find all modules matching the source pattern
                source_modules = await self._find_modules_matching_pattern(source_pattern)
                
                for source_module in source_modules:
                    deps = await self.glean.get_dependencies(source_module, depth=1)
                    
                    for dep in deps["direct"]:
                        # Check if this dependency is forbidden
                        for forbidden_pattern in forbidden_patterns:
                            if dep.startswith(forbidden_pattern):
                                self.violations.append(ArchitectureViolation(
                                    violation_type=ViolationType.FORBIDDEN_DEPENDENCY,
                                    severity="high",
                                    description=f"Forbidden dependency: {source_module} depends on {dep}",
                                    source=source_module,
                                    target=dep,
                                    recommendation=f"Remove dependency on {forbidden_pattern} or refactor architecture"
                                ))
            
            except Exception as e:
                logger.warning(f"Error validating forbidden dependencies for {source_pattern}: {e}")
    
    async def _find_modules_matching_pattern(self, pattern: str) -> List[str]:
        """Find all modules matching a pattern"""
        # This would need to query Glean for all modules
        # For now, return a simplified list based on known structure
        if pattern == "cryptotrading.core":
            return [
                "cryptotrading.core.agents.base",
                "cryptotrading.core.agents.memory", 
                "cryptotrading.core.agents.strands",
                "cryptotrading.core.protocols.mcp",
                "cryptotrading.core.protocols.a2a",
                "cryptotrading.core.ml",
                "cryptotrading.core.ai"
            ]
        elif pattern == "cryptotrading.data":
            return [
                "cryptotrading.data.database",
                "cryptotrading.data.storage",
                "cryptotrading.data.historical"
            ]
        elif pattern == "cryptotrading.infrastructure":
            return [
                "cryptotrading.infrastructure.logging",
                "cryptotrading.infrastructure.monitoring",
                "cryptotrading.infrastructure.security",
                "cryptotrading.infrastructure.registry"
            ]
        elif pattern == "cryptotrading.utils":
            return ["cryptotrading.utils"]
        else:
            return [pattern]
    
    async def _validate_naming_conventions(self):
        """Validate naming conventions"""
        logger.info("Validating naming conventions")
        
        # Get all Python files in the project
        src_path = Path("/Users/apple/projects/cryptotrading/src/cryptotrading")
        
        for py_file in src_path.rglob("*.py"):
            # Check file naming conventions
            if not self._is_valid_python_filename(py_file.name):
                self.violations.append(ArchitectureViolation(
                    violation_type=ViolationType.NAMING_CONVENTION,
                    severity="low",
                    description=f"File name doesn't follow Python conventions: {py_file.name}",
                    source=str(py_file),
                    file_path=str(py_file),
                    recommendation="Use snake_case for Python file names"
                ))
            
            # Check module structure
            relative_path = py_file.relative_to(src_path)
            module_path = str(relative_path).replace('/', '.').replace('.py', '')
            
            # Validate module path follows conventions
            if not self._is_valid_module_path(module_path):
                self.violations.append(ArchitectureViolation(
                    violation_type=ViolationType.NAMING_CONVENTION,
                    severity="medium",
                    description=f"Module path doesn't follow conventions: {module_path}",
                    source=module_path,
                    file_path=str(py_file),
                    recommendation="Ensure module paths follow the established package structure"
                ))
    
    def _is_valid_python_filename(self, filename: str) -> bool:
        """Check if filename follows Python conventions"""
        if filename == "__init__.py":
            return True
        
        # Should be snake_case
        if filename.endswith(".py"):
            name = filename[:-3]
            return name.islower() and '_' in name or name.isalpha()
        
        return False
    
    def _is_valid_module_path(self, module_path: str) -> bool:
        """Check if module path follows established conventions"""
        # Should start with known package structure
        valid_prefixes = [
            "core.agents",
            "core.protocols", 
            "core.ml",
            "core.ai",
            "data.database",
            "data.storage",
            "data.historical",
            "infrastructure.logging",
            "infrastructure.monitoring",
            "infrastructure.security",
            "infrastructure.registry",
            "infrastructure.analysis",
            "utils"
        ]
        
        return any(module_path.startswith(prefix) for prefix in valid_prefixes)
    
    async def _validate_module_sizes(self):
        """Validate module sizes don't exceed limits"""
        logger.info("Validating module sizes")
        
        src_path = Path("/Users/apple/projects/cryptotrading/src/cryptotrading")
        
        for py_file in src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            
            try:
                # Get file symbols
                symbols = await self.glean.get_file_symbols(str(py_file))
                
                function_count = len([s for s in symbols if s.type == "function"])
                class_count = len([s for s in symbols if s.type == "class"])
                
                # Check function count
                if function_count > self.module_size_limits["max_functions_per_module"]:
                    self.violations.append(ArchitectureViolation(
                        violation_type=ViolationType.MODULE_SIZE,
                        severity="medium",
                        description=f"Module has too many functions: {function_count} (max: {self.module_size_limits['max_functions_per_module']})",
                        source=str(py_file),
                        file_path=str(py_file),
                        recommendation="Consider splitting the module into smaller, more focused modules"
                    ))
                
                # Check class count
                if class_count > self.module_size_limits["max_classes_per_module"]:
                    self.violations.append(ArchitectureViolation(
                        violation_type=ViolationType.MODULE_SIZE,
                        severity="medium",
                        description=f"Module has too many classes: {class_count} (max: {self.module_size_limits['max_classes_per_module']})",
                        source=str(py_file),
                        file_path=str(py_file),
                        recommendation="Consider splitting classes into separate modules"
                    ))
                
                # Check file size (lines of code)
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = len([line for line in f if line.strip() and not line.strip().startswith('#')])
                
                if lines > self.module_size_limits["max_lines_per_file"]:
                    self.violations.append(ArchitectureViolation(
                        violation_type=ViolationType.MODULE_SIZE,
                        severity="high",
                        description=f"File is too large: {lines} lines (max: {self.module_size_limits['max_lines_per_file']})",
                        source=str(py_file),
                        file_path=str(py_file),
                        recommendation="Refactor large file into smaller, more focused modules"
                    ))
            
            except Exception as e:
                logger.warning(f"Error validating size for {py_file}: {e}")
    
    async def _validate_coupling(self):
        """Validate coupling constraints"""
        logger.info("Validating coupling constraints")
        
        # Get all modules
        all_modules = []
        for layer in self.layers:
            modules = await self._find_modules_matching_pattern(layer)
            all_modules.extend(modules)
        
        for module in all_modules:
            try:
                # Get module facts
                facts = await self.glean.get_module_facts(module)
                
                dependency_count = facts["dependency_count"]
                dependent_count = facts["dependent_count"]
                
                # Check efferent coupling (dependencies)
                if dependency_count > self.coupling_limits["max_dependencies_per_module"]:
                    self.violations.append(ArchitectureViolation(
                        violation_type=ViolationType.COUPLING_VIOLATION,
                        severity="high",
                        description=f"Module has too many dependencies: {dependency_count} (max: {self.coupling_limits['max_dependencies_per_module']})",
                        source=module,
                        recommendation="Reduce dependencies by using dependency injection or interfaces"
                    ))
                
                # Check afferent coupling (dependents)
                if dependent_count > self.coupling_limits["max_dependents_per_module"]:
                    self.violations.append(ArchitectureViolation(
                        violation_type=ViolationType.COUPLING_VIOLATION,
                        severity="medium",
                        description=f"Module is used by too many other modules: {dependent_count} (max: {self.coupling_limits['max_dependents_per_module']})",
                        source=module,
                        recommendation="Consider splitting the module or creating interfaces to reduce coupling"
                    ))
            
            except Exception as e:
                logger.warning(f"Error validating coupling for {module}: {e}")
    
    def generate_violation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive violation report"""
        if not self.violations:
            return {
                "status": "clean",
                "total_violations": 0,
                "message": "No architectural violations found! 🎉"
            }
        
        # Group violations by type and severity
        by_type = {}
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for violation in self.violations:
            # Group by type
            vtype = violation.violation_type.value
            if vtype not in by_type:
                by_type[vtype] = []
            by_type[vtype].append(violation)
            
            # Count by severity
            by_severity[violation.severity] += 1
        
        # Generate summary
        total_violations = len(self.violations)
        critical_count = by_severity["critical"]
        high_count = by_severity["high"]
        
        if critical_count > 0:
            status = "critical"
            message = f"🚨 {critical_count} critical architectural violations require immediate attention!"
        elif high_count > 0:
            status = "warning"
            message = f"⚠️ {high_count} high-priority architectural violations found"
        else:
            status = "minor"
            message = f"ℹ️ {total_violations} minor architectural violations found"
        
        return {
            "status": status,
            "message": message,
            "total_violations": total_violations,
            "by_severity": by_severity,
            "by_type": {vtype: len(violations) for vtype, violations in by_type.items()},
            "violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity,
                    "description": v.description,
                    "source": v.source,
                    "target": v.target,
                    "file_path": v.file_path,
                    "line": v.line,
                    "recommendation": v.recommendation
                }
                for v in self.violations
            ]
        }
    
    def get_violations_by_severity(self, severity: str) -> List[ArchitectureViolation]:
        """Get violations filtered by severity"""
        return [v for v in self.violations if v.severity == severity]
    
    def get_violations_by_type(self, violation_type: ViolationType) -> List[ArchitectureViolation]:
        """Get violations filtered by type"""
        return [v for v in self.violations if v.violation_type == violation_type]
