"""
Secure Code Execution Sandbox for Strands Framework
Provides secure, isolated code execution with comprehensive safety controls
"""
import asyncio
import subprocess
import tempfile
import os
import signal
import resource
import sys
import ast
import logging
import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import threading
import queue

logger = logging.getLogger(__name__)


class ExecutionResult(Enum):
    """Code execution results"""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    SECURITY_VIOLATION = "security_violation"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    PERMISSION_DENIED = "permission_denied"


class SecurityLevel(Enum):
    """Security levels for code execution"""
    STRICT = "strict"        # Maximum security, minimal capabilities
    NORMAL = "normal"        # Balanced security and functionality
    PERMISSIVE = "permissive"  # More functionality, reduced security


@dataclass
class ExecutionLimits:
    """Resource limits for code execution"""
    max_execution_time: float = 30.0  # seconds
    max_memory_mb: int = 128  # megabytes
    max_cpu_time: float = 10.0  # seconds
    max_output_size: int = 1024 * 100  # 100KB
    max_file_size: int = 1024 * 10  # 10KB
    max_open_files: int = 10
    allow_network: bool = False
    allow_file_write: bool = False
    allow_subprocess: bool = False


@dataclass
class SandboxResult:
    """Result of sandboxed code execution"""
    result: ExecutionResult
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    memory_used: int = 0
    return_value: Any = None
    security_violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SecureASTValidator:
    """Validates Python AST for security violations"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STRICT):
        self.security_level = security_level
        self._setup_allowed_nodes()
    
    def _setup_allowed_nodes(self):
        """Setup allowed AST node types based on security level"""
        # Base allowed nodes (safe operations)
        base_allowed = {
            ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Constant,
            ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Attribute,
            ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.If, ast.For,
            ast.While, ast.Break, ast.Continue, ast.Return, ast.FunctionDef,
            ast.arguments, ast.arg, ast.Call, ast.Load, ast.Store,
            # Math operations
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
            ast.Invert, ast.Not, ast.UAdd, ast.USub,
            # Comparisons
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
            ast.In, ast.NotIn, ast.And, ast.Or,
        }
        
        if self.security_level == SecurityLevel.STRICT:
            self.allowed_nodes = base_allowed
            # Very restrictive - only basic operations
            
        elif self.security_level == SecurityLevel.NORMAL:
            self.allowed_nodes = base_allowed | {
                ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
                ast.Lambda, ast.Try, ast.ExceptHandler, ast.Raise,
                ast.Pass, ast.ClassDef, ast.Global, ast.Nonlocal,
                ast.With, ast.withitem
            }
            
        else:  # PERMISSIVE
            self.allowed_nodes = base_allowed | {
                ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
                ast.Lambda, ast.Try, ast.ExceptHandler, ast.Raise,
                ast.Pass, ast.ClassDef, ast.Global, ast.Nonlocal,
                ast.With, ast.withitem, ast.Yield, ast.YieldFrom,
                ast.AsyncFunctionDef, ast.Await, ast.AsyncWith, ast.AsyncFor
            }
        
        # Always forbidden nodes (dangerous operations)
        self.forbidden_nodes = {
            ast.Import, ast.ImportFrom,  # Prevent imports
            ast.Exec,  # Prevent exec() if it exists
            ast.Eval,  # Prevent eval() if it exists
        }
        
        # Dangerous function calls to check
        self.forbidden_functions = {
            'eval', 'exec', 'compile', '__import__', 'open', 'file',
            'input', 'raw_input', 'reload', 'vars', 'locals', 'globals',
            'dir', 'hasattr', 'getattr', 'setattr', 'delattr',
            'callable', 'isinstance', 'issubclass', 'type', 'super',
            'exit', 'quit', 'help', 'copyright', 'license', 'credits'
        }
        
        if not self.security_level == SecurityLevel.PERMISSIVE:
            self.forbidden_functions.update({
                'print',  # Controlled output only
                'len', 'range', 'enumerate', 'zip', 'map', 'filter',
                'sorted', 'reversed', 'sum', 'max', 'min', 'abs',
                'round', 'pow', 'divmod'
            })
    
    def validate(self, code: str) -> List[str]:
        """
        Validate code for security violations
        
        Args:
            code: Python code to validate
            
        Returns:
            List of security violations found
        """
        violations = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(f"Syntax error: {e}")
            return violations
        
        # Walk the AST and check each node
        for node in ast.walk(tree):
            # Check for forbidden node types
            if type(node) in self.forbidden_nodes:
                violations.append(f"Forbidden operation: {type(node).__name__}")
            
            # Check for forbidden function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.forbidden_functions:
                        violations.append(f"Forbidden function call: {func_name}")
                
                # Check for attribute calls that might be dangerous
                elif isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr
                    if attr_name.startswith('_'):
                        violations.append(f"Access to private attribute: {attr_name}")
            
            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('__'):
                    violations.append(f"Access to dunder attribute: {node.attr}")
        
        return violations


class CodeSandbox:
    """Secure code execution sandbox"""
    
    def __init__(self, limits: ExecutionLimits = None, 
                 security_level: SecurityLevel = SecurityLevel.STRICT):
        self.limits = limits or ExecutionLimits()
        self.security_level = security_level
        self.validator = SecureASTValidator(security_level)
        self.temp_dir = None
        self._setup_sandbox()
    
    def _setup_sandbox(self):
        """Setup sandbox environment"""
        # Create temporary directory for sandbox
        self.temp_dir = tempfile.mkdtemp(prefix="strands_sandbox_")
        
        # Ensure cleanup on exit
        import atexit
        atexit.register(self._cleanup_sandbox)
    
    def _cleanup_sandbox(self):
        """Clean up sandbox resources"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup sandbox directory: {e}")
    
    async def execute_code(self, code: str, context: Dict[str, Any] = None,
                          allowed_imports: List[str] = None) -> SandboxResult:
        """
        Execute code in secure sandbox
        
        Args:
            code: Python code to execute
            context: Safe context variables to provide
            allowed_imports: List of allowed import modules
            
        Returns:
            SandboxResult with execution details
        """
        start_time = time.time()
        result = SandboxResult(result=ExecutionResult.SUCCESS)
        
        try:
            # Step 1: Validate code security
            violations = self.validator.validate(code)
            if violations:
                result.result = ExecutionResult.SECURITY_VIOLATION
                result.security_violations = violations
                result.error = f"Security violations: {'; '.join(violations)}"
                return result
            
            # Step 2: Prepare safe execution context
            safe_context = self._create_safe_context(context or {}, allowed_imports or [])
            
            # Step 3: Execute in subprocess for isolation
            result = await self._execute_in_subprocess(code, safe_context)
            
        except Exception as e:
            result.result = ExecutionResult.RUNTIME_ERROR
            result.error = f"Sandbox error: {e}"
            logger.error(f"Sandbox execution error: {e}", exc_info=True)
        
        result.execution_time = time.time() - start_time
        return result
    
    def _create_safe_context(self, user_context: Dict[str, Any], 
                           allowed_imports: List[str]) -> Dict[str, Any]:
        """Create safe execution context"""
        # Start with minimal safe builtins
        safe_builtins = {
            'True': True, 'False': False, 'None': None,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
        }
        
        if self.security_level != SecurityLevel.STRICT:
            safe_builtins.update({
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'sum': sum, 'max': max, 'min': min,
                'abs': abs, 'round': round, 'sorted': sorted,
                'reversed': reversed
            })
        
        # Add allowed imports
        safe_modules = {}
        for module_name in allowed_imports:
            if module_name in ['math', 'random', 'datetime', 'json']:
                try:
                    safe_modules[module_name] = __import__(module_name)
                except ImportError:
                    pass
        
        # Combine contexts
        safe_context = {
            '__builtins__': safe_builtins,
            **safe_modules,
            **self._sanitize_user_context(user_context)
        }
        
        return safe_context
    
    def _sanitize_user_context(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize user-provided context"""
        sanitized = {}
        
        for key, value in user_context.items():
            # Only allow safe variable names
            if not key.isidentifier() or key.startswith('_'):
                continue
            
            # Only allow safe value types
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                # Recursive sanitization for containers
                if isinstance(value, dict):
                    sanitized[key] = self._sanitize_dict(value)
                elif isinstance(value, (list, tuple)):
                    sanitized[key] = self._sanitize_list(value)
                else:
                    sanitized[key] = value
        
        return sanitized
    
    def _sanitize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary"""
        sanitized = {}
        for k, v in d.items():
            if isinstance(k, str) and k.isidentifier() and not k.startswith('_'):
                if isinstance(v, (str, int, float, bool)):
                    sanitized[k] = v
                elif isinstance(v, dict):
                    sanitized[k] = self._sanitize_dict(v)
                elif isinstance(v, (list, tuple)):
                    sanitized[k] = self._sanitize_list(v)
        return sanitized
    
    def _sanitize_list(self, lst: List[Any]) -> List[Any]:
        """Recursively sanitize list"""
        sanitized = []
        for item in lst:
            if isinstance(item, (str, int, float, bool)):
                sanitized.append(item)
            elif isinstance(item, dict):
                sanitized.append(self._sanitize_dict(item))
            elif isinstance(item, (list, tuple)):
                sanitized.append(self._sanitize_list(item))
        return sanitized
    
    async def _execute_in_subprocess(self, code: str, context: Dict[str, Any]) -> SandboxResult:
        """Execute code in isolated subprocess"""
        # Create sandbox script
        sandbox_script = self._create_sandbox_script(code, context)
        script_path = os.path.join(self.temp_dir, "sandbox_code.py")
        
        with open(script_path, 'w') as f:
            f.write(sandbox_script)
        
        # Execute with resource limits
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir,
                preexec_fn=self._set_resource_limits if os.name != 'nt' else None
            )
            
            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.limits.max_execution_time
                )
                
                return self._parse_subprocess_result(stdout, stderr, process.returncode)
                
            except asyncio.TimeoutError:
                # Kill the process
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass
                
                return SandboxResult(
                    result=ExecutionResult.TIMEOUT,
                    error=f"Execution timed out after {self.limits.max_execution_time}s"
                )
                
        except Exception as e:
            return SandboxResult(
                result=ExecutionResult.RUNTIME_ERROR,
                error=f"Subprocess execution failed: {e}"
            )
    
    def _set_resource_limits(self):
        """Set resource limits for subprocess (Unix only)"""
        try:
            # Set memory limit
            memory_limit = self.limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # Set CPU time limit
            cpu_limit = int(self.limits.max_cpu_time)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            
            # Set file size limit
            file_limit = self.limits.max_file_size
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
            
            # Set number of open files limit
            resource.setrlimit(resource.RLIMIT_NOFILE, (self.limits.max_open_files, self.limits.max_open_files))
            
        except Exception as e:
            logger.warning(f"Failed to set resource limits: {e}")
    
    def _create_sandbox_script(self, code: str, context: Dict[str, Any]) -> str:
        """Create sandbox execution script"""
        # Serialize context safely
        context_json = json.dumps(context, default=str)
        
        script = f'''
import sys
import json
import traceback
import signal
import time

# Set up signal handler for timeout
def timeout_handler(signum, frame):
    print("__SANDBOX_TIMEOUT__")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({int(self.limits.max_execution_time) + 1})

try:
    # Load context
    context = json.loads("""{context_json}""")
    
    # Set up globals
    exec_globals = context.copy()
    exec_locals = {{}}
    
    # Execute user code
    start_time = time.time()
    exec("""
{code}
""", exec_globals, exec_locals)
    
    execution_time = time.time() - start_time
    
    # Return results
    print("__SANDBOX_SUCCESS__")
    print(f"__SANDBOX_TIME__:{execution_time}")
    
    # Print any returned values
    if 'result' in exec_locals:
        print(f"__SANDBOX_RESULT__:{exec_locals['result']}")
    
except MemoryError:
    print("__SANDBOX_MEMORY_ERROR__")
    sys.exit(1)
except SyntaxError as e:
    print(f"__SANDBOX_SYNTAX_ERROR__:{e}")
    sys.exit(1)
except Exception as e:
    print(f"__SANDBOX_RUNTIME_ERROR__:{e}")
    print(f"__SANDBOX_TRACEBACK__:{traceback.format_exc()}")
    sys.exit(1)
'''
        return script
    
    def _parse_subprocess_result(self, stdout: bytes, stderr: bytes, 
                                returncode: int) -> SandboxResult:
        """Parse subprocess execution result"""
        stdout_str = stdout.decode('utf-8', errors='ignore')
        stderr_str = stderr.decode('utf-8', errors='ignore')
        
        result = SandboxResult(result=ExecutionResult.SUCCESS)
        
        # Parse special markers from output
        lines = stdout_str.split('\n')
        output_lines = []
        
        for line in lines:
            if line.startswith('__SANDBOX_'):
                if line == '__SANDBOX_SUCCESS__':
                    result.result = ExecutionResult.SUCCESS
                elif line == '__SANDBOX_TIMEOUT__':
                    result.result = ExecutionResult.TIMEOUT
                elif line == '__SANDBOX_MEMORY_ERROR__':
                    result.result = ExecutionResult.MEMORY_ERROR
                elif line.startswith('__SANDBOX_SYNTAX_ERROR__:'):
                    result.result = ExecutionResult.SYNTAX_ERROR
                    result.error = line.split(':', 1)[1]
                elif line.startswith('__SANDBOX_RUNTIME_ERROR__:'):
                    result.result = ExecutionResult.RUNTIME_ERROR
                    result.error = line.split(':', 1)[1]
                elif line.startswith('__SANDBOX_TIME__:'):
                    try:
                        result.execution_time = float(line.split(':', 1)[1])
                    except ValueError:
                        pass
                elif line.startswith('__SANDBOX_RESULT__:'):
                    try:
                        result.return_value = line.split(':', 1)[1]
                    except Exception:
                        pass
                elif line.startswith('__SANDBOX_TRACEBACK__:'):
                    result.error += f"\n{line.split(':', 1)[1]}"
            else:
                output_lines.append(line)
        
        result.output = '\n'.join(output_lines).strip()
        
        if stderr_str:
            result.error = result.error + f"\nSTDERR: {stderr_str}" if result.error else stderr_str
        
        if returncode != 0 and result.result == ExecutionResult.SUCCESS:
            result.result = ExecutionResult.RUNTIME_ERROR
            if not result.error:
                result.error = f"Process exited with code {returncode}"
        
        return result


class SecureCodeExecutor:
    """High-level secure code executor for Strands framework"""
    
    def __init__(self, default_security_level: SecurityLevel = SecurityLevel.STRICT):
        self.default_security_level = default_security_level
        self.execution_history: List[Dict[str, Any]] = []
        self.security_violations: List[Dict[str, Any]] = []
    
    async def execute_safe_code(self, code: str, tool_name: str,
                               context: Dict[str, Any] = None,
                               security_level: SecurityLevel = None,
                               timeout: float = None) -> Dict[str, Any]:
        """
        Execute code safely with comprehensive security controls
        
        Args:
            code: Python code to execute
            tool_name: Name of the tool executing the code
            context: Safe execution context
            security_level: Override default security level
            timeout: Override default timeout
            
        Returns:
            Dictionary with execution results
        """
        security_level = security_level or self.default_security_level
        
        # Create execution limits
        limits = ExecutionLimits()
        if timeout:
            limits.max_execution_time = timeout
        
        # Create sandbox
        sandbox = CodeSandbox(limits=limits, security_level=security_level)
        
        try:
            # Execute in sandbox
            result = await sandbox.execute_code(code, context)
            
            # Log execution
            execution_record = {
                'tool_name': tool_name,
                'security_level': security_level.value,
                'result': result.result.value,
                'execution_time': result.execution_time,
                'timestamp': datetime.utcnow().isoformat(),
                'has_violations': bool(result.security_violations)
            }
            self.execution_history.append(execution_record)
            
            # Log security violations
            if result.security_violations:
                violation_record = {
                    'tool_name': tool_name,
                    'violations': result.security_violations,
                    'code_hash': hashlib.sha256(code.encode()).hexdigest()[:16],
                    'timestamp': datetime.utcnow().isoformat()
                }
                self.security_violations.append(violation_record)
                logger.warning(f"Security violations in {tool_name}: {result.security_violations}")
            
            # Return structured result
            return {
                'success': result.result == ExecutionResult.SUCCESS,
                'result': result.result.value,
                'output': result.output,
                'error': result.error,
                'execution_time': result.execution_time,
                'return_value': result.return_value,
                'security_violations': result.security_violations,
                'warnings': result.warnings
            }
            
        except Exception as e:
            logger.error(f"Code execution failed for {tool_name}: {e}", exc_info=True)
            return {
                'success': False,
                'result': 'execution_error',
                'error': f"Execution failed: {e}",
                'security_violations': ['execution_system_error']
            }
        finally:
            # Cleanup sandbox
            sandbox._cleanup_sandbox()
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security status and statistics"""
        total_executions = len(self.execution_history)
        violation_count = len(self.security_violations)
        
        recent_executions = [
            record for record in self.execution_history
            if datetime.fromisoformat(record['timestamp']) > datetime.utcnow() - timedelta(hours=1)
        ]
        
        return {
            'total_executions': total_executions,
            'total_violations': violation_count,
            'recent_executions_1h': len(recent_executions),
            'violation_rate': violation_count / max(total_executions, 1),
            'default_security_level': self.default_security_level.value,
            'last_violation': self.security_violations[-1] if self.security_violations else None
        }
    
    def clear_history(self):
        """Clear execution history (for maintenance)"""
        self.execution_history.clear()
        self.security_violations.clear()
        logger.info("Cleared code execution history")