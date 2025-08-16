/**
 * Vercel Edge Function for MCTS Calculation Agent
 * Provides serverless endpoint for Monte Carlo Tree Search calculations
 */
import { NextRequest, NextResponse } from 'next/server';

export const config = {
  runtime: 'edge',
  // Increase timeout for complex calculations
  maxDuration: 30,
};

// Input validation schema
interface CalculationRequest {
  problem_type: 'trading' | 'portfolio' | 'optimization';
  parameters: {
    initial_portfolio: number;
    symbols: string[];
    max_depth?: number;
    [key: string]: any;
  };
  iterations?: number;
  timeout?: number;
}

// Response types
interface CalculationResponse {
  type: 'calculation_result' | 'error';
  best_action?: any;
  expected_value?: number;
  confidence?: number;
  exploration_stats?: any;
  error?: string;
  cached?: boolean;
  metrics?: {
    execution_time: number;
    agent_id: string;
    calculation_count: number;
  };
}

// Edge function handler
export default async function handler(req: NextRequest) {
  // CORS headers for browser compatibility
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Content-Type': 'application/json',
  };

  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    return new NextResponse(null, { status: 200, headers });
  }

  // Only allow POST
  if (req.method !== 'POST') {
    return NextResponse.json(
      { error: 'Method not allowed' },
      { status: 405, headers }
    );
  }

  try {
    // Parse request body
    const body: CalculationRequest = await req.json();

    // Validate required fields
    if (!body.problem_type || !body.parameters) {
      return NextResponse.json(
        { error: 'Missing required fields: problem_type, parameters' },
        { status: 400, headers }
      );
    }

    // Validate problem type
    if (!['trading', 'portfolio', 'optimization'].includes(body.problem_type)) {
      return NextResponse.json(
        { error: 'Invalid problem_type. Must be: trading, portfolio, or optimization' },
        { status: 400, headers }
      );
    }

    // Validate parameters
    const { initial_portfolio, symbols } = body.parameters;
    
    if (typeof initial_portfolio !== 'number' || initial_portfolio <= 0) {
      return NextResponse.json(
        { error: 'initial_portfolio must be a positive number' },
        { status: 400, headers }
      );
    }

    if (!Array.isArray(symbols) || symbols.length === 0) {
      return NextResponse.json(
        { error: 'symbols must be a non-empty array' },
        { status: 400, headers }
      );
    }

    // Set defaults
    const iterations = Math.min(body.iterations || 1000, 10000);
    const timeout = Math.min(body.timeout || 30, 30); // Max 30s for Edge Functions

    // Create cache key for response caching
    const cacheKey = `mcts:${body.problem_type}:${JSON.stringify(body.parameters)}:${iterations}`;
    
    // Check KV cache if available
    // @ts-ignore - KV namespace injected by Vercel
    if (typeof MCTS_CACHE !== 'undefined') {
      const cached = await MCTS_CACHE.get(cacheKey, 'json');
      if (cached) {
        return NextResponse.json(
          { ...cached, cached: true },
          { status: 200, headers }
        );
      }
    }

    // Execute MCTS calculation
    // In production, this would call the Python agent via internal API
    const result = await performMCTSCalculation(body, iterations, timeout);

    // Cache successful results
    // @ts-ignore
    if (typeof MCTS_CACHE !== 'undefined' && !result.error) {
      await MCTS_CACHE.put(cacheKey, JSON.stringify(result), {
        expirationTtl: 300, // 5 minutes
      });
    }

    // Return response
    return NextResponse.json(result, { 
      status: result.error ? 400 : 200, 
      headers 
    });

  } catch (error) {
    console.error('MCTS calculation error:', error);
    return NextResponse.json(
      { 
        type: 'error',
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500, headers }
    );
  }
}

/**
 * Perform MCTS calculation
 * In production, this would communicate with the Python agent
 */
async function performMCTSCalculation(
  request: CalculationRequest,
  iterations: number,
  timeout: number
): Promise<CalculationResponse> {
  // For demo purposes, return a mock response
  // In production, this would call the Python MCTS agent via internal API
  
  const startTime = Date.now();
  
  // Simulate calculation delay
  await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));
  
  // Mock calculation result
  const mockResult: CalculationResponse = {
    type: 'calculation_result',
    best_action: {
      type: 'buy',
      symbol: request.parameters.symbols[0],
      percentage: 0.25,
      risk_score: 0.3
    },
    expected_value: 0.0875,
    confidence: 0.82,
    exploration_stats: {
      iterations: iterations,
      elapsed_time: (Date.now() - startTime) / 1000,
      iterations_per_second: iterations / ((Date.now() - startTime) / 1000),
      average_value: 0.0642,
      max_value: 0.1823,
      min_value: -0.0921,
      tree_size: Math.floor(iterations * 0.7)
    },
    metrics: {
      execution_time: (Date.now() - startTime) / 1000,
      agent_id: 'mcts-edge-001',
      calculation_count: 1
    }
  };
  
  return mockResult;
}

/**
 * Helper function to validate symbols
 */
function isValidSymbol(symbol: string): boolean {
  const validSymbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'MATIC', 'LINK', 'UNI', 'AAVE', 'SUSHI'];
  return validSymbols.includes(symbol.toUpperCase());
}