/**
 * Vercel Edge Function for MCTS Calculation Agent
 * Provides serverless endpoint for Monte Carlo Tree Search calculations
 */
import { kv } from '@vercel/kv';

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
export default async function handler(req: Request): Promise<Response> {
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Content-Type': 'application/json',
  };

  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 200, headers });
  }

  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers,
    });
  }

  try {
    const body: CalculationRequest = await req.json();

    if (!body.problem_type || !body.parameters) {
      return new Response(
        JSON.stringify({ error: 'Missing required fields: problem_type, parameters' }),
        { status: 400, headers }
      );
    }

    if (!['trading', 'portfolio', 'optimization'].includes(body.problem_type)) {
      return new Response(
        JSON.stringify({
          error: 'Invalid problem_type. Must be: trading, portfolio, or optimization',
        }),
        { status: 400, headers }
      );
    }

    const { initial_portfolio, symbols } = body.parameters;
    if (typeof initial_portfolio !== 'number' || initial_portfolio <= 0) {
      return new Response(JSON.stringify({ error: 'initial_portfolio must be a positive number' }), {
        status: 400,
        headers,
      });
    }

    if (!Array.isArray(symbols) || symbols.length === 0) {
      return new Response(JSON.stringify({ error: 'symbols must be a non-empty array' }), {
        status: 400,
        headers,
      });
    }

    const iterations = Math.min(body.iterations || 1000, 10000);
    const timeout = Math.min(body.timeout || 30, 30);

    const cacheKey = `mcts:${body.problem_type}:${JSON.stringify(body.parameters)}:${iterations}`;

    try {
      const cached: CalculationResponse | null = await kv.get(cacheKey);
      if (cached) {
        return new Response(JSON.stringify({ ...cached, cached: true }), {
          status: 200,
          headers,
        });
      }
    } catch (e) {
      console.error('Failed to access Vercel KV for GET', e);
    }

    const result = await performMCTSCalculation(body, iterations, timeout);

    try {
      if (!result.error) {
        await kv.set(cacheKey, result, { ex: 300 });
      }
    } catch (e) {
      console.error('Failed to access Vercel KV for SET', e);
    }

    return new Response(JSON.stringify(result), {
      status: result.error ? 400 : 200,
      headers,
    });

  } catch (error) {
    console.error('MCTS calculation error:', error);
    return new Response(
      JSON.stringify({
        type: 'error',
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error',
      }),
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