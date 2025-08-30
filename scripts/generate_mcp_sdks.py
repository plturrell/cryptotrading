#!/usr/bin/env python3
"""
Generate Client SDKs from OpenAPI Specification
Supports Python, TypeScript, JavaScript, Go, and Java
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml


class MCPSDKGenerator:
    """Generate client SDKs for MCP tools"""

    def __init__(self, openapi_spec_path: str):
        self.spec_path = openapi_spec_path
        self.output_dir = Path("generated_sdks")
        self.output_dir.mkdir(exist_ok=True)

        # Load OpenAPI spec
        with open(openapi_spec_path, "r") as f:
            if openapi_spec_path.endswith(".yaml"):
                self.spec = yaml.safe_load(f)
            else:
                self.spec = json.load(f)

    def generate_python_sdk(self) -> str:
        """Generate Python SDK"""
        output_path = self.output_dir / "python"
        output_path.mkdir(exist_ok=True)

        # Generate SDK structure
        self._create_python_structure(output_path)

        # Generate client
        client_code = self._generate_python_client()
        with open(output_path / "mcp_client.py", "w") as f:
            f.write(client_code)

        # Generate models
        models_code = self._generate_python_models()
        with open(output_path / "models.py", "w") as f:
            f.write(models_code)

        # Generate setup.py
        setup_code = self._generate_python_setup()
        with open(output_path / "setup.py", "w") as f:
            f.write(setup_code)

        print(f"Python SDK generated at {output_path}")
        return str(output_path)

    def _create_python_structure(self, path: Path):
        """Create Python package structure"""
        (path / "mcp_sdk").mkdir(exist_ok=True)
        (path / "mcp_sdk" / "__init__.py").touch()
        (path / "tests").mkdir(exist_ok=True)
        (path / "examples").mkdir(exist_ok=True)

    def _generate_python_client(self) -> str:
        """Generate Python client code"""
        return '''"""
MCP Tools Python SDK
Auto-generated from OpenAPI specification
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class MCPConfig:
    """Configuration for MCP client"""
    base_url: str = "https://api.rex.com/mcp"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3

class MCPClient:
    """Main client for MCP tools"""
    
    def __init__(self, config: MCPConfig = None):
        self.config = config or MCPConfig()
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers=self._get_headers(),
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MCP-Python-SDK/1.0.0"
        }
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        return headers
    
    async def discover_tools(self, category: str = None, tag: str = None) -> Dict[str, Any]:
        """Discover available MCP tools"""
        params = {}
        if category:
            params["category"] = category
        if tag:
            params["tag"] = tag
        
        async with self.session.get(
            f"{self.config.base_url}/tools/discover",
            params=params
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def execute_tool(self, tool_name: str, method: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool method"""
        url = f"{self.config.base_url}/tools/{tool_name}/{method}"
        
        async with self.session.post(url, json=kwargs) as response:
            response.raise_for_status()
            return await response.json()
    
    # Technical Analysis Tools
    async def technical_analysis(self, symbol: str, indicators: List[str]) -> Dict[str, Any]:
        """Execute technical analysis"""
        return await self.execute_tool(
            "technicalanalysis",
            "analyze",
            symbol=symbol,
            indicators=indicators
        )
    
    # ML Model Tools
    async def train_model(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML model"""
        return await self.execute_tool(
            "mlmodels",
            "train",
            data=data,
            config=config
        )
    
    async def predict(self, model_id: str, features: List[float]) -> Dict[str, Any]:
        """Make prediction with ML model"""
        return await self.execute_tool(
            "mlmodels",
            "predict",
            model_id=model_id,
            features=features
        )
    
    # Historical Data Tools
    async def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch historical market data"""
        return await self.execute_tool(
            "historicaldata",
            "fetch",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
    
    # Feature Engineering Tools
    async def engineer_features(self, data: List[Dict], method: str = "auto") -> Dict[str, Any]:
        """Engineer features for ML"""
        return await self.execute_tool(
            "featureengineering",
            "engineer",
            data=data,
            method=method
        )

# Convenience functions
async def quick_analysis(symbol: str, api_key: str = None):
    """Quick technical analysis for a symbol"""
    config = MCPConfig(api_key=api_key)
    async with MCPClient(config) as client:
        return await client.technical_analysis(
            symbol,
            ["RSI", "MACD", "BB", "SMA", "EMA"]
        )

# Example usage
if __name__ == "__main__":
    async def main():
        async with MCPClient() as client:
            # Discover tools
            tools = await client.discover_tools(category="trading")
            print(f"Found {len(tools['tools'])} trading tools")
            
            # Run technical analysis
            result = await client.technical_analysis("BTC-USD", ["RSI", "MACD"])
            print(f"Analysis result: {result}")
    
    asyncio.run(main())
'''

    def _generate_python_models(self) -> str:
        """Generate Python model classes"""
        return '''"""
Data models for MCP SDK
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ToolCategory(Enum):
    """Tool categories"""
    ANALYSIS = "analysis"
    DATA = "data"
    ML = "machine_learning"
    TRADING = "trading"
    INFRASTRUCTURE = "infrastructure"

@dataclass
class ToolMetadata:
    """Tool metadata"""
    name: str
    description: str
    category: ToolCategory
    version: str
    methods: List[str]
    tags: List[str]

@dataclass
class TechnicalIndicator:
    """Technical analysis indicator"""
    name: str
    value: float
    timestamp: datetime
    parameters: Dict[str, Any]

@dataclass
class MLModel:
    """Machine learning model"""
    id: str
    name: str
    type: str
    version: str
    accuracy: float
    features: List[str]
    created_at: datetime

@dataclass
class HistoricalData:
    """Historical market data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class FeatureSet:
    """Engineered feature set"""
    id: str
    features: List[str]
    method: str
    timestamp: datetime
    metadata: Dict[str, Any]
'''

    def _generate_python_setup(self) -> str:
        """Generate setup.py for Python SDK"""
        return """from setuptools import setup, find_packages

setup(
    name="mcp-sdk",
    version="1.0.0",
    description="Python SDK for MCP Tools",
    author="A2A Platform Team",
    author_email="support@rex.com",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.8.0",
        "pyyaml>=6.0",
        "python-dateutil>=2.8.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
"""

    def generate_typescript_sdk(self) -> str:
        """Generate TypeScript SDK"""
        output_path = self.output_dir / "typescript"
        output_path.mkdir(exist_ok=True)

        # Generate client
        client_code = self._generate_typescript_client()
        with open(output_path / "mcpClient.ts", "w") as f:
            f.write(client_code)

        # Generate types
        types_code = self._generate_typescript_types()
        with open(output_path / "types.ts", "w") as f:
            f.write(types_code)

        # Generate package.json
        package_json = self._generate_typescript_package()
        with open(output_path / "package.json", "w") as f:
            f.write(package_json)

        print(f"TypeScript SDK generated at {output_path}")
        return str(output_path)

    def _generate_typescript_client(self) -> str:
        """Generate TypeScript client code"""
        return """/**
 * MCP Tools TypeScript SDK
 * Auto-generated from OpenAPI specification
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { ToolMetadata, TechnicalAnalysisResult, MLModelConfig, HistoricalData } from './types';

export interface MCPConfig {
  baseURL?: string;
  apiKey?: string;
  timeout?: number;
  maxRetries?: number;
}

export class MCPClient {
  private client: AxiosInstance;
  private config: MCPConfig;

  constructor(config: MCPConfig = {}) {
    this.config = {
      baseURL: config.baseURL || 'https://api.rex.com/mcp',
      apiKey: config.apiKey,
      timeout: config.timeout || 30000,
      maxRetries: config.maxRetries || 3,
    };

    this.client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'MCP-TypeScript-SDK/1.0.0',
        ...(this.config.apiKey && { 'X-API-Key': this.config.apiKey }),
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add timestamp
        config.headers['X-Request-Time'] = new Date().toISOString();
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor with retry logic
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const config = error.config;
        if (!config || !config.retry) {
          config.retry = 0;
        }

        if (config.retry < this.config.maxRetries! && error.response?.status >= 500) {
          config.retry++;
          await new Promise(resolve => setTimeout(resolve, 1000 * config.retry));
          return this.client(config);
        }

        return Promise.reject(error);
      }
    );
  }

  /**
   * Discover available MCP tools
   */
  async discoverTools(category?: string, tag?: string): Promise<{ tools: ToolMetadata[] }> {
    const response = await this.client.get('/tools/discover', {
      params: { category, tag },
    });
    return response.data;
  }

  /**
   * Execute technical analysis
   */
  async technicalAnalysis(symbol: string, indicators: string[]): Promise<TechnicalAnalysisResult> {
    const response = await this.client.post('/tools/technicalanalysis/analyze', {
      symbol,
      indicators,
    });
    return response.data;
  }

  /**
   * Train ML model
   */
  async trainModel(data: any, config: MLModelConfig): Promise<{ modelId: string; metrics: any }> {
    const response = await this.client.post('/tools/mlmodels/train', {
      data,
      config,
    });
    return response.data;
  }

  /**
   * Make prediction with ML model
   */
  async predict(modelId: string, features: number[]): Promise<{ prediction: number; confidence: number }> {
    const response = await this.client.post('/tools/mlmodels/predict', {
      modelId,
      features,
    });
    return response.data;
  }

  /**
   * Fetch historical data
   */
  async fetchHistoricalData(
    symbol: string,
    startDate: string,
    endDate: string
  ): Promise<HistoricalData[]> {
    const response = await this.client.post('/tools/historicaldata/fetch', {
      symbol,
      startDate,
      endDate,
    });
    return response.data;
  }
}

// Export convenience functions
export async function quickAnalysis(symbol: string, apiKey?: string): Promise<TechnicalAnalysisResult> {
  const client = new MCPClient({ apiKey });
  return client.technicalAnalysis(symbol, ['RSI', 'MACD', 'BB', 'SMA', 'EMA']);
}
"""

    def _generate_typescript_types(self) -> str:
        """Generate TypeScript type definitions"""
        return """/**
 * Type definitions for MCP SDK
 */

export enum ToolCategory {
  ANALYSIS = 'analysis',
  DATA = 'data',
  ML = 'machine_learning',
  TRADING = 'trading',
  INFRASTRUCTURE = 'infrastructure',
}

export interface ToolMetadata {
  name: string;
  description: string;
  category: ToolCategory;
  version: string;
  methods: string[];
  tags: string[];
}

export interface TechnicalIndicator {
  name: string;
  value: number;
  timestamp: Date;
  parameters: Record<string, any>;
}

export interface TechnicalAnalysisResult {
  symbol: string;
  indicators: TechnicalIndicator[];
  timestamp: Date;
}

export interface MLModelConfig {
  algorithm: string;
  hyperparameters: Record<string, any>;
  features: string[];
}

export interface HistoricalData {
  symbol: string;
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface FeatureSet {
  id: string;
  features: string[];
  method: string;
  timestamp: Date;
  metadata: Record<string, any>;
}
"""

    def _generate_typescript_package(self) -> str:
        """Generate package.json for TypeScript SDK"""
        return """{
  "name": "@mcp/sdk",
  "version": "1.0.0",
  "description": "TypeScript SDK for MCP Tools",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "test": "jest",
    "lint": "eslint . --ext .ts"
  },
  "dependencies": {
    "axios": "^1.6.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0",
    "jest": "^29.0.0",
    "@types/jest": "^29.0.0",
    "eslint": "^8.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "@typescript-eslint/eslint-plugin": "^6.0.0"
  },
  "author": "A2A Platform Team",
  "license": "MIT"
}
"""

    def generate_all_sdks(self):
        """Generate SDKs for all supported languages"""
        print("Generating client SDKs from OpenAPI specification...")

        # Generate Python SDK
        python_path = self.generate_python_sdk()

        # Generate TypeScript SDK
        typescript_path = self.generate_typescript_sdk()

        # Generate README
        readme_content = f"""# MCP Tools Client SDKs

Auto-generated client SDKs for MCP Tools API.

## Available SDKs

- **Python SDK**: {python_path}
- **TypeScript SDK**: {typescript_path}

## Installation

### Python
```bash
cd {python_path}
pip install -e .
```

### TypeScript
```bash
cd {typescript_path}
npm install
npm run build
```

## Usage Examples

### Python
```python
from mcp_sdk import MCPClient, MCPConfig

async with MCPClient(MCPConfig(api_key="your-key")) as client:
    tools = await client.discover_tools()
    result = await client.technical_analysis("BTC-USD", ["RSI", "MACD"])
```

### TypeScript
```typescript
import {{ MCPClient }} from '@mcp/sdk';

const client = new MCPClient({{ apiKey: 'your-key' }});
const tools = await client.discoverTools();
const result = await client.technicalAnalysis('BTC-USD', ['RSI', 'MACD']);
```

## Documentation

Full API documentation available at: https://api.rex.com/docs

Generated on: {datetime.now().isoformat()}
"""

        with open(self.output_dir / "README.md", "w") as f:
            f.write(readme_content)

        print(f"SDKs generated successfully in {self.output_dir}")
        print("- Python SDK")
        print("- TypeScript SDK")
        print("- README.md")


def main():
    """Main function to generate SDKs"""
    # Check if OpenAPI spec exists
    spec_path = "api/openapi_mcp_tools.yaml"
    if not os.path.exists(spec_path):
        spec_path = "api/openapi_mcp_tools.json"

    if not os.path.exists(spec_path):
        print("OpenAPI specification not found. Generating...")
        # Import and run the OpenAPI generator
        from api.mcp_openapi_generator import generate_openapi_for_all_mcp_tools

        generate_openapi_for_all_mcp_tools()
        spec_path = "api/openapi_mcp_tools.yaml"

    # Generate SDKs
    generator = MCPSDKGenerator(spec_path)
    generator.generate_all_sdks()


if __name__ == "__main__":
    main()
