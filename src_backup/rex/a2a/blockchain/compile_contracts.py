#!/usr/bin/env python3
"""
Contract Compilation Script
Compiles Solidity contracts and updates bytecode in Python files
"""

import json
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compile_workflow_instance():
    """Compile WorkflowInstance.sol and extract bytecode"""
    contracts_dir = Path(__file__).parent / "contracts"
    workflow_contract = contracts_dir / "WorkflowInstance.sol"
    
    if not workflow_contract.exists():
        logger.error(f"Contract not found: {workflow_contract}")
        return None
    
    logger.info("Compiling WorkflowInstance.sol with solc...")
    
    try:
        # Check if solc is installed
        result = subprocess.run(["solc", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("solc not installed. Install with: npm install -g solc")
            return None
        
        # Compile contract
        compile_cmd = [
            "solc",
            "--optimize",
            "--combined-json", "abi,bin",
            str(workflow_contract)
        ]
        
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Compilation failed: {result.stderr}")
            return None
        
        # Parse output
        output = json.loads(result.stdout)
        
        # Extract contract info
        contract_key = None
        for key in output["contracts"]:
            if "WorkflowInstance" in key:
                contract_key = key
                break
        
        if not contract_key:
            logger.error("WorkflowInstance contract not found in compilation output")
            return None
        
        contract_info = output["contracts"][contract_key]
        bytecode = "0x" + contract_info["bin"]
        abi = json.loads(contract_info["abi"])
        
        logger.info(f"Contract compiled successfully. Bytecode length: {len(bytecode)}")
        
        # Update workflow_instance_contract.py with real bytecode
        update_python_file(bytecode, abi)
        
        return {
            "bytecode": bytecode,
            "abi": abi
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse compiler output: {e}")
        return None
    except Exception as e:
        logger.error(f"Compilation error: {e}")
        return None

def update_python_file(bytecode: str, abi: list):
    """Update workflow_instance_contract.py with compiled bytecode"""
    py_file = Path(__file__).parent / "workflow_instance_contract.py"
    
    if not py_file.exists():
        logger.error(f"Python file not found: {py_file}")
        return
    
    # Read file
    content = py_file.read_text()
    
    # Replace bytecode
    import_idx = content.find("WORKFLOW_INSTANCE_BYTECODE = None")
    if import_idx == -1:
        logger.error("Could not find WORKFLOW_INSTANCE_BYTECODE in file")
        return
    
    # Create new content with actual bytecode
    new_line = f'WORKFLOW_INSTANCE_BYTECODE = "{bytecode}"'
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        if "WORKFLOW_INSTANCE_BYTECODE = None" in line:
            lines[i] = new_line
            break
    
    # Write back
    py_file.write_text('\n'.join(lines))
    logger.info(f"Updated {py_file} with compiled bytecode")

def main():
    """Main compilation function"""
    logger.info("Starting contract compilation...")
    
    result = compile_workflow_instance()
    
    if result:
        logger.info("✅ Contract compilation complete!")
        logger.info(f"Bytecode: {result['bytecode'][:66]}...")
        logger.info(f"ABI functions: {len([x for x in result['abi'] if x['type'] == 'function'])}")
    else:
        logger.error("❌ Contract compilation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()