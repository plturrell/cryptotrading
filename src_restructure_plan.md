# Source Directory Restructure Plan

## Current Structure Analysis

### Major Issues Identified
1. **Massive Duplication**: 24+ agent classes with overlapping functionality
2. **Non-Standard Layout**: Deep nesting (4+ levels) and mixed concerns
3. **Circular Dependencies**: Multiple competing base classes and inheritance chains
4. **Inconsistent Naming**: `rex` vs `strands` vs `mcp` at same package level

### Current Structure
```
src/
├── rex/                     # Main application (141 files)
│   ├── a2a/                # A2A protocol (61 files)
│   │   ├── agents/         # 24 agent files - MASSIVE DUPLICATION
│   │   ├── blockchain/     # Blockchain functionality
│   │   ├── registry/       # Agent registry
│   │   └── protocols/      # Communication protocols
│   ├── registry/           # DUPLICATE registry
│   ├── memory/             # Memory management
│   ├── database/           # Database models
│   ├── ml/                 # Machine learning
│   ├── observability/      # Monitoring
│   └── [15+ other modules]
├── strands/                # DUPLICATE agent framework (10 files)
├── mcp/                    # MCP protocol (22 files)
└── data/                   # Data utilities (1 file)
```

## Proposed Standard Structure

### New Python Package Layout
```
src/
├── cryptotrading/          # Main application package
│   ├── __init__.py
│   ├── core/              # Core business logic
│   │   ├── __init__.py
│   │   ├── agents/        # Unified agent system
│   │   │   ├── __init__.py
│   │   │   ├── base.py    # Single base agent class
│   │   │   ├── memory.py  # Memory-enabled agents
│   │   │   ├── strands.py # Strands integration
│   │   │   └── specialized/ # Specific agent types
│   │   ├── protocols/     # Communication protocols
│   │   │   ├── __init__.py
│   │   │   ├── a2a/       # A2A protocol
│   │   │   └── mcp/       # MCP protocol
│   │   ├── blockchain/    # Blockchain integration
│   │   └── ml/           # Machine learning
│   ├── data/             # Data access layer
│   │   ├── __init__.py
│   │   ├── database/     # Database models and access
│   │   ├── storage/      # Storage abstractions
│   │   └── historical/   # Historical data
│   ├── infrastructure/   # Infrastructure concerns
│   │   ├── __init__.py
│   │   ├── logging/      # Logging configuration
│   │   ├── monitoring/   # Observability
│   │   ├── security/     # Authentication/authorization
│   │   └── registry/     # Service registry
│   └── utils/           # Utilities and helpers
└── tests/               # Package-level tests
```

## Migration Strategy

### Phase 1: Agent Consolidation (Critical)
**Problem**: 24+ agent classes with overlapping functionality
**Solution**: Create unified agent hierarchy

#### Current Agent Classes to Consolidate:
- `BaseAgent` (rex/a2a/agents/base_classes.py)
- `BaseStrandsAgent` (rex/a2a/agents/base_strands_agent.py)  
- `A2AAgentBase` (rex/a2a/agents/a2a_agent_base.py)
- `BaseMemoryAgent` (rex/a2a/agents/base_memory_agent.py)
- `A2AStrandsAgent` (rex/a2a/agents/a2a_strands_agent.py)
- `MemoryStrandsAgent` (rex/a2a/agents/memory_strands_agent.py)
- `BlockchainStrandsAgent` (rex/a2a/agents/blockchain_strands_agent.py)
- `Agent` (strands/agent.py) - DUPLICATE

#### New Unified Structure:
```python
# cryptotrading/core/agents/base.py
class BaseAgent(ABC):
    """Single base agent class"""

# cryptotrading/core/agents/memory.py  
class MemoryAgent(BaseAgent):
    """Memory-enabled agent"""

# cryptotrading/core/agents/strands.py
class StrandsAgent(MemoryAgent):
    """Strands framework integration"""
```

### Phase 2: Protocol Consolidation
**Problem**: Mixed protocol implementations
**Solution**: Separate protocol packages

#### Move:
- `src/mcp/*` → `cryptotrading/core/protocols/mcp/`
- `src/rex/a2a/protocols/*` → `cryptotrading/core/protocols/a2a/`

### Phase 3: Data Layer Consolidation  
**Problem**: Database models mixed with business logic
**Solution**: Clean data access layer

#### Move:
- `src/rex/database/*` → `cryptotrading/data/database/`
- `src/rex/storage/*` → `cryptotrading/data/storage/`
- `src/rex/historical_data/*` → `cryptotrading/data/historical/`

### Phase 4: Infrastructure Consolidation
**Problem**: Infrastructure scattered across modules
**Solution**: Dedicated infrastructure package

#### Move:
- `src/rex/logging/*` → `cryptotrading/infrastructure/logging/`
- `src/rex/observability/*` → `cryptotrading/infrastructure/monitoring/`
- `src/rex/security/*` → `cryptotrading/infrastructure/security/`
- `src/rex/registry/*` → `cryptotrading/infrastructure/registry/`

### Phase 5: Remove Duplicates
**Problem**: Duplicate registries and utilities
**Solution**: Single source of truth

#### Eliminate:
- `src/rex/registry/` (duplicate of `src/rex/a2a/registry/`)
- `src/strands/` (duplicate agent framework)
- `src/data/` (minimal utility, merge into main package)

## Implementation Steps

### Step 1: Create New Structure
1. Create new package directories
2. Create `__init__.py` files
3. Set up proper imports

### Step 2: Agent Migration
1. Analyze agent dependencies
2. Create unified base classes
3. Migrate specialized agents
4. Update imports throughout codebase

### Step 3: Protocol Migration
1. Move MCP modules
2. Move A2A protocol modules  
3. Update imports and references

### Step 4: Data Layer Migration
1. Move database models
2. Move storage abstractions
3. Update database imports

### Step 5: Infrastructure Migration
1. Move logging, monitoring, security
2. Consolidate registries
3. Update configuration

### Step 6: Cleanup
1. Remove duplicate directories
2. Update all imports
3. Run tests to verify functionality
4. Update documentation

## Benefits of Restructure

### Technical Benefits
- ✅ **Single Source of Truth**: No duplicate agent classes
- ✅ **Clear Separation**: Business logic vs infrastructure vs data
- ✅ **Standard Layout**: Follows Python packaging best practices
- ✅ **Better Imports**: `from cryptotrading.core.agents import BaseAgent`
- ✅ **IDE Support**: Better navigation and autocomplete

### Maintenance Benefits  
- ✅ **Easier Testing**: Clear package boundaries
- ✅ **Simpler Dependencies**: No circular imports
- ✅ **Better Documentation**: Logical module organization
- ✅ **Scalable**: Easy to add new features in right place

## Risk Mitigation

### Potential Risks
- Import breakage across codebase
- Circular dependency issues during migration
- Test failures during transition

### Mitigation Strategies
- Create migration script to update imports automatically
- Migrate in phases with testing at each step
- Keep backup of current structure
- Use git branches for safe migration

## Next Steps

1. **Approve this plan** and migration strategy
2. **Create backup branch** for current structure
3. **Start with Phase 1** (Agent consolidation) - highest impact
4. **Implement migration script** for automated import updates
5. **Test thoroughly** at each phase

This restructure will transform the codebase from a maintenance nightmare into a professional, maintainable Python package following industry best practices.
