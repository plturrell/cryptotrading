# Glean Integration for Crypto Trading Platform

## Overview

This document describes the production-ready integration of Facebook's Glean system into the crypto trading platform for enhanced source code diagnosis and tracing capabilities.

## Features

### Core Analysis Capabilities
- **Dependency Analysis**: Deep dependency graph analysis with configurable depth
- **Impact Analysis**: Change impact assessment with risk scoring and affected test detection
- **Architecture Validation**: Enforcement of architectural constraints and design rules
- **Real-time Monitoring**: File watching with automatic re-analysis on code changes
- **Dead Code Detection**: Identification of unused functions and classes
- **Complexity Analysis**: Code complexity metrics and maintainability scoring

### Production Features
- **Docker Integration**: Automated Glean server management via Docker
- **CLI Interface**: Comprehensive command-line interface for all operations
- **Interactive Visualizations**: HTML dependency graphs with D3.js
- **Async Processing**: Non-blocking analysis with proper error handling
- **Configurable Rules**: Customizable architectural constraints and limits
- **Comprehensive Logging**: Structured logging with multiple output formats

## Architecture

```
src/cryptotrading/infrastructure/analysis/
├── glean_client.py          # Core Glean server communication
├── code_analyzer.py         # Dependency and complexity analysis
├── impact_analyzer.py       # Change impact assessment
├── architecture_validator.py # Architectural constraint validation
├── realtime_analyzer.py     # Real-time file monitoring
├── cli_commands.py          # CLI command implementations
└── __init__.py             # Package exports
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Setup Glean server and environment
python scripts/setup_glean.py
```

### 2. Basic Usage

```bash
# Check status
python scripts/glean_cli.py status

# Run full analysis
python scripts/glean_cli.py analyze

# Analyze specific module dependencies
python scripts/glean_cli.py deps cryptotrading.core.agents.base

# Validate architecture
python scripts/glean_cli.py arch --format detailed

# Start real-time monitoring
python scripts/glean_cli.py monitor
```

### 3. Impact Analysis

```bash
# Analyze impact of file changes
python scripts/glean_cli.py impact src/cryptotrading/core/agents/base.py

# Multiple files with JSON output
python scripts/glean_cli.py impact src/cryptotrading/data/*.py --format json
```

## Configuration

### Glean Server Configuration (`config/glean.json`)

```json
{
  "server": {
    "url": "http://localhost:8888",
    "timeout": 30
  },
  "indexing": {
    "source_path": "src/cryptotrading",
    "languages": ["python"],
    "exclude_patterns": ["__pycache__", "*.pyc"]
  },
  "analysis": {
    "max_depth": 5,
    "timeout": 60,
    "cache_results": true
  }
}
```

### Architectural Rules

The system enforces the following architectural constraints:

#### Layer Dependencies (High to Low)
1. `cryptotrading.webapp` - Presentation layer
2. `cryptotrading.api` - API layer  
3. `cryptotrading.core.agents` - Application layer
4. `cryptotrading.core.protocols` - Protocol layer
5. `cryptotrading.core.ml` - ML/AI layer
6. `cryptotrading.data` - Data access layer
7. `cryptotrading.infrastructure` - Infrastructure layer
8. `cryptotrading.utils` - Utility layer

#### Forbidden Dependencies
- Core should not depend on infrastructure
- Data layer should not depend on core business logic
- Utils should not depend on anything except standard library
- Infrastructure should not depend on core agents/protocols

#### Size Limits
- Max 20 functions per module
- Max 10 classes per module  
- Max 500 lines per file
- Max 15 dependencies per module

## CLI Commands Reference

### Analysis Commands

| Command | Description | Example |
|---------|-------------|---------|
| `deps` | Analyze module dependencies | `glean_cli.py deps cryptotrading.core.agents.base` |
| `impact` | Analyze change impact | `glean_cli.py impact src/file.py` |
| `arch` | Validate architecture | `glean_cli.py arch --format detailed` |
| `analyze` | Full codebase analysis | `glean_cli.py analyze --format json` |

### Monitoring Commands

| Command | Description | Example |
|---------|-------------|---------|
| `monitor` | Start real-time monitoring | `glean_cli.py monitor -a dependency_analysis` |
| `status` | Get monitoring status | `glean_cli.py status` |

### Utility Commands

| Command | Description | Example |
|---------|-------------|---------|
| `setup` | Setup environment | `glean_cli.py setup --docker` |

## Output Formats

### Tree Format (Default for Dependencies)
```
📦 Dependencies for cryptotrading.core.agents.base
==================================================

🔗 Direct Dependencies:
├── cryptotrading.core.protocols.mcp
├── cryptotrading.data.database
└── cryptotrading.utils

🔄 Transitive Dependencies:
├── sqlalchemy
├── aiohttp
└── ... and 15 more
```

### Summary Format (Default for Analysis)
```
🔍 Comprehensive Codebase Analysis Summary
==================================================
✅ Analysis completed in 12.3 seconds

📦 Dependencies: 45 modules analyzed
🏗️ Architecture: ⚠️ 3 violations found
🧮 Complexity: Average 2.1
💀 Dead Code: 2 unused functions detected
```

### JSON Format
```json
{
  "analysis_summary": {
    "duration_seconds": 12.3,
    "success": true
  },
  "dependencies": {
    "modules": ["..."],
    "graph": "..."
  },
  "architecture": {
    "total_violations": 3,
    "by_severity": {"high": 1, "medium": 2}
  }
}
```

## Real-time Monitoring

The real-time analyzer watches for file changes and automatically triggers analysis:

```python
from src.cryptotrading.infrastructure.analysis import RealtimeCodeAnalyzer, GleanClient

# Setup monitoring
glean_client = GleanClient()
analyzer = RealtimeCodeAnalyzer("/path/to/project", glean_client)

# Start watching
await analyzer.start_watching()

# Configure which analyses to run
analyzer.configure_analyses({
    AnalysisType.DEPENDENCY_ANALYSIS,
    AnalysisType.IMPACT_ANALYSIS,
    AnalysisType.ARCHITECTURE_VALIDATION
})
```

## Integration with Existing Tools

### MCP Integration
The Glean integration works alongside the existing MCP (Model Context Protocol) system:

```python
from src.cryptotrading.infrastructure.analysis import GleanCLI
from src.cryptotrading.core.protocols.mcp import MCPServer

# Use Glean analysis in MCP tools
glean_cli = GleanCLI()
analysis_result = await glean_cli.run_full_analysis()
```

### Agent Testing Framework
Glean commands are integrated into the agent testing framework in `framework/agent_testing/`.

## Docker Setup

The Glean server runs in a Docker container for isolation and consistency:

```bash
# Pull and start Glean server
docker pull ghcr.io/facebookincubator/glean/glean-server:latest
docker run -d --name glean-server -p 8888:8888 \
  -v $(pwd)/data/glean:/data \
  -v $(pwd)/src:/workspace:ro \
  ghcr.io/facebookincubator/glean/glean-server:latest
```

## Troubleshooting

### Common Issues

1. **Glean server not responding**
   ```bash
   # Check if container is running
   docker ps | grep glean-server
   
   # Restart if needed
   docker restart glean-server
   ```

2. **Import errors**
   ```bash
   # Ensure src is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

3. **Permission errors**
   ```bash
   # Fix data directory permissions
   sudo chown -R $(whoami) data/glean/
   ```

### Logs

- Glean server logs: `docker logs glean-server`
- Analysis logs: `logs/glean_analysis.log`
- CLI logs: Console output with `--verbose` flag

## Performance Considerations

- **Initial indexing**: May take 2-5 minutes for large codebases
- **Real-time analysis**: Debounced with 1-second delay
- **Memory usage**: ~200MB for typical crypto trading platform size
- **Docker resources**: Allocate at least 1GB RAM for Glean server

## Security

- Glean server runs in isolated Docker container
- Read-only access to source code
- No network access required for analysis
- All data stored locally in `data/glean/`

## Future Enhancements

- [ ] Integration with CI/CD pipelines
- [ ] Slack/Teams notifications for architecture violations
- [ ] Web dashboard for analysis results
- [ ] Custom rule definitions via configuration
- [ ] Integration with code review tools
- [ ] Performance regression detection
- [ ] Test coverage correlation with complexity metrics

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs for error details
3. Ensure all dependencies are installed
4. Verify Docker setup is correct
