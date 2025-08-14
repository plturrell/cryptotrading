# A2A Use Case Testing Framework

This directory contains SAP-standard ISO-compliant use case documentation and test implementations for the rex Trading Platform A2A agent system.

## Structure

```
use_cases/
├── README.md (this file)
├── cryptodatadownload/
│   ├── UC001_CryptoDataDownload_Schema_Discovery.md
│   ├── test_uc001_schema_discovery.py
│   ├── run_uc001_tests.py
│   └── test_results/
│       ├── UC001_Traceability_Report.md
│       ├── UC001_SAP_Test_Report.json
│       └── UC001_Test_Certificate.txt
└── [future use cases]/
```

## Standards Compliance

### ISO Standards
- **ISO/IEC/IEEE 29148:2018** - Systems and software engineering — Life cycle processes — Requirements engineering
- **ISO/IEC 25010:2011** - System and software quality models
- **ISO/IEC/IEEE 29119-3:2013** - Software testing — Test documentation

### SAP Standards
- **SAP TDD-UC-001** - Test-Driven Development for Use Cases
- **SAP CAP Core Schema** - For data product definitions
- **SAP Object Resource Discovery** - For metadata management

## Use Case Documentation Format

Each use case follows this structure:

1. **Use Case Identification** - Unique ID, name, priority
2. **Description** - Business context and value
3. **Actors** - Primary, secondary, and supporting actors
4. **Preconditions** - Required state before execution
5. **Basic Flow** - Step-by-step normal execution
6. **Alternative Flows** - Error and exception handling
7. **Postconditions** - Expected state after execution
8. **Data Elements** - Input/output specifications
9. **Business Rules** - Constraints and validations
10. **Non-Functional Requirements** - Performance, security, etc.
11. **Test Scenarios** - Mapped test cases
12. **Traceability Matrix** - Requirements to code mapping
13. **Dependencies** - External and internal dependencies
14. **Open Issues** - Known limitations
15. **Sign-off** - Approval tracking

## Running Tests

### Individual Use Case Tests

```bash
cd tests/a2a/use_cases/cryptodatadownload
python run_uc001_tests.py
```

### All Use Case Tests

```bash
cd tests/a2a/use_cases
python -m pytest . -v --tb=short
```

### Generate Test Reports

Each test runner generates:
- **Traceability Report** (Markdown) - Requirements mapping
- **SAP Test Report** (JSON) - Detailed test metrics
- **Test Certificate** (Text) - Compliance certification

## Code Linkage

All use cases include direct links to implementation code:
- Format: `src/path/to/file.py#L100-200`
- Links are validated by test suite
- Enables traceability from requirements to code

## Adding New Use Cases

1. Create directory: `mkdir use_case_name`
2. Copy template: `cp template/UC_TEMPLATE.md use_case_name/UCXXX_Name.md`
3. Implement tests: `use_case_name/test_ucxxx_name.py`
4. Create runner: `use_case_name/run_ucxxx_tests.py`
5. Update this README

## Quality Gates

Before marking a use case as complete:

1. ✅ All test scenarios pass (100%)
2. ✅ Code coverage > 80%
3. ✅ Performance requirements met
4. ✅ Traceability complete
5. ✅ SAP report generated
6. ✅ Test certificate issued
7. ✅ Sign-off obtained

## CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Run A2A Use Case Tests
  run: |
    cd tests/a2a/use_cases
    python -m pytest . --junit-xml=test-results.xml
    
- name: Upload Test Reports
  uses: actions/upload-artifact@v2
  with:
    name: a2a-test-reports
    path: tests/a2a/use_cases/**/UC*_Report.*
```

## Compliance Dashboard

Track use case test status:

| Use Case | Status | Pass Rate | Last Run | Certificate |
|----------|--------|-----------|----------|-------------|
| UC001 | ✅ Active | 100% | 2025-01-12 | [View](cryptodatadownload/UC001_Test_Certificate.txt) |
| UC002 | 🚧 In Progress | - | - | - |

## Contact

- **Product Owner:** rex Trading Platform Team
- **QA Lead:** quality@rex.com
- **Compliance:** compliance@rex.com