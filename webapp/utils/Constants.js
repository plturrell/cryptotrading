sap.ui.define([], function() {
    "use strict";

    /**
     * Application Constants
     * Central place for all magic numbers and configuration values
     */
    return {
        // Time Constants (in milliseconds)
        TIME: {
            REFRESH_INTERVAL: 30000, // 30 seconds
            WEBSOCKET_TIMEOUT: 5000, // 5 seconds
            SHORT_TIMEOUT: 1500, // 1.5 seconds
            TOAST_DURATION: 3000, // 3 seconds
            AUTO_REFRESH: 5000, // 5 seconds
            NEWS_REFRESH: 300000, // 5 minutes
            CACHE_CLEANUP: 3600000, // 1 hour
            SESSION_TIMEOUT: 3600000, // 1 hour
            MILLISECONDS_IN_SECOND: 1000, // Milliseconds in second
            SECONDS_IN_DAY: 86400, // Seconds in a day
            DAYS_30_MS: 2592000000, // 30 days in milliseconds
            DAYS_365_MS: 31536000000 // 365 days in milliseconds
        },

        // Numeric Constants
        NUMBERS: {
            ZERO: 0, // Zero constant
            ONE: 1, // One constant
            HEX_BASE: 16, // Hexadecimal base
            HEX_MASK_LOW: 0x3, // Hexadecimal mask low bits
            HEX_MASK_HIGH: 0x8, // Hexadecimal mask high bits
            DECIMAL_PLACES: 4, // Price precision
            PERCENTAGE_PLACES: 2, // Percentage precision
            CHART_POINTS: 288, // 5-minute intervals in 24h
            WEB3_DECIMALS: 18, // Ethereum decimals
            WEI_TO_ETH: 1e18, // Wei to ETH conversion
            CRYPTO_PRECISION: 8, // Crypto precision
            MAX_RETRIES: 10, // Max retry attempts
            DEFAULT_LIMIT: 50, // Default API limit
            MAX_LIMIT: 500, // Maximum API limit
            SMALL_LIMIT: 20, // Small data limit
            LARGE_LIMIT: 200, // Large data limit
            HOURS_IN_DAY: 24, // Hours per day
            MINUTES_IN_HOUR: 60, // Minutes per hour
            SECONDS_IN_MINUTE: 60, // Seconds per minute
            DAYS_IN_MONTH: 30, // Average days in month
            DAYS_IN_YEAR: 365, // Days in year
            MONTHS_IN_YEAR: 12, // Months in year
            WEEKS_IN_MONTH: 4, // Average weeks in month
            PERCENTAGE_MULTIPLIER: 100, // Convert decimal to percentage
            HALF_PERCENTAGE: 50, // 50%
            THREE_QUARTERS: 75, // 75%
            ONE_QUARTER: 25, // 25%
            BINARY_HALF: 2, // Half divisor
            BINARY_QUARTER: 4, // Quarter divisor
            SMALL_STEP: 5, // Small increment
            MEDIUM_STEP: 10, // Medium increment
            LARGE_STEP: 50, // Large increment
            MAX_FILE_SIZE: 10000, // Maximum file size
            COMPLEXITY_LIMIT: 10, // Cyclomatic complexity limit
            PERCENTAGE_70: 70, // 70%
            PERCENTAGE_85: 85, // 85%
            PERCENTAGE_90: 90, // 90%
            PERCENTAGE_95: 95, // 95%
            PERCENTAGE_98: 98, // 98%
            ADDRESS_LENGTH: 42, // Ethereum address length
            CHAIN_POINTS: 288, // 5-minute intervals in 24h
            NEGATIVE_CHAIN: -288, // Negative chain points
            PRECISION_2: 2, // 2 decimal places
            PRECISION_6: 6, // 6 decimal places
            TIME_CHUNKS: 36 // Time division chunks
        },

        // Technical Analysis Constants
        TECHNICAL: {
            RSI_OVERBOUGHT: 70, // RSI overbought level
            RSI_OVERSOLD: 30, // RSI oversold level
            MACD_PERIODS: {
                FAST: 12, // MACD fast period
                SLOW: 26, // MACD slow period
                SIGNAL: 9 // MACD signal period
            },
            BOLLINGER_DEVIATION: 2, // Bollinger bands deviation
            SMA_SHORT: 20, // Short SMA period
            SMA_LONG: 50, // Long SMA period
            VOLUME_THRESHOLD: 1.5 // Volume spike threshold
        },

        // Risk Management Constants
        RISK: {
            MAX_POSITION_PCT: 50, // Maximum position as % of portfolio
            DEFAULT_RISK_PCT: 2, // Default risk per trade
            STOP_LOSS_PCT: 5, // Default stop loss percentage
            MAX_DRAWDOWN: 20, // Maximum drawdown percentage
            LEVERAGE_LIMIT: 10, // Maximum leverage
            CORRELATION_LIMIT: 0.8, // Maximum correlation
            VAR_CONFIDENCE: 95, // VaR confidence level
            SHARP_THRESHOLD: 0.7, // Minimum Sharpe ratio
            MAX_ORDERS: 50 // Maximum open orders
        },

        // ML/AI Constants
        ML: {
            TRAIN_TEST_SPLIT: 0.8, // Training data split
            VALIDATION_SPLIT: 0.6, // Validation split
            PREDICTION_HORIZON: 24, // Hours to predict
            MODEL_ACCURACY_MIN: 0.7, // Minimum model accuracy
            CONFIDENCE_THRESHOLD: 0.9, // Prediction confidence
            FEATURE_IMPORTANCE: 0.4, // Feature importance threshold
            FEATURE_THRESHOLD_LOW: 0.3, // Low feature threshold
            EPOCHS: 100, // Training epochs
            BATCH_SIZE: 32, // Training batch size
            LEARNING_RATE: 0.001, // Learning rate
            SIGNAL_PERIOD: 9 // Signal period for analysis
        },

        // HTTP Status Codes
        HTTP: {
            OK: 200, // Success
            CREATED: 201, // Created
            NO_CONTENT: 204, // No content
            BAD_REQUEST: 400, // Bad request
            UNAUTHORIZED: 401, // Unauthorized
            FORBIDDEN: 403, // Forbidden
            NOT_FOUND: 404, // Not found
            SERVER_ERROR: 500 // Internal server error
        },

        // Crypto Constants
        CRYPTO: {
            GWEI: 1e9, // Gwei to Wei conversion
            SATOSHI: 1e8, // Satoshi to BTC conversion
            GAS_LIMIT: 21000, // Standard gas limit
            MAX_GAS_PRICE: 100, // Maximum gas price in Gwei
            BLOCK_TIME: 15 // Average block time in seconds
        },

        // UI Constants
        UI: {
            GRID_COLUMNS: 12, // Grid system columns
            MOBILE_BREAKPOINT: 768, // Mobile breakpoint
            TABLET_BREAKPOINT: 1024, // Tablet breakpoint
            ANIMATION_DURATION: 300, // Animation duration
            DEBOUNCE_DELAY: 250, // Input debounce delay
            INFINITE_SCROLL_THRESHOLD: 200 // Infinite scroll threshold
        },

        // String Constants
        STRINGS: {
            EMPTY: "", // Empty string
            SPACE: " ", // Space character
            COMMA: ",", // Comma separator
            DOT: ".", // Dot separator
            DASH: "-", // Dash separator
            UNDERSCORE: "_", // Underscore
            PIPE: "|", // Pipe separator
            NEWLINE: "\n", // New line
            TAB: "\t" // Tab character
        },

        // Date Format Constants
        DATE_FORMATS: {
            ISO: "YYYY-MM-DDTHH:mm:ss.sssZ", // ISO format
            DATE_ONLY: "YYYY-MM-DD", // Date only
            TIME_ONLY: "HH:mm:ss", // Time only
            DISPLAY: "DD/MM/YYYY HH:mm", // Display format
            SHORT: "DD/MM/YY", // Short date
            LONG: "MMMM DD, YYYY" // Long date
        },

        // Color Constants (for charts and UI)
        COLORS: {
            POSITIVE: "#28a745", // Green for positive values
            NEGATIVE: "#dc3545", // Red for negative values
            NEUTRAL: "#6c757d", // Gray for neutral
            WARNING: "#ffc107", // Yellow for warnings
            INFO: "#007bff", // Blue for info
            SUCCESS: "#28a745", // Green for success
            PRIMARY: "#007bff" // Primary brand color
        }
    };
});
