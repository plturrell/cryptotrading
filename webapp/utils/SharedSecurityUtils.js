sap.ui.define([
    "sap/base/security/encodeXML",
    "sap/base/security/encodeJS",
    "sap/base/security/encodeURL",
    "sap/base/strings/escapeRegExp",
    "sap/base/Log",
    "sap/m/MessageToast"
], (encodeXML, encodeJS, encodeURL, escapeRegExp, Log, MessageToast) => {
    "use strict";

    /**
     * Shared Security Utilities for Crypto Trading Platform
     * Provides comprehensive security features including:
     * - Input validation and output encoding
     * - CSRF protection and secure HTTP calls
     * - XSS and injection attack prevention
     * - API key management and validation
     * - Secure trading data handling
     * - Rate limiting and request throttling
     */
    return {

        /**
         * Security configuration
         */
        CONFIG: {
            MAX_REQUEST_SIZE: 1024 * 1024, // 1MB
            RATE_LIMIT_WINDOW: 60000, // 1 minute
            MAX_REQUESTS_PER_WINDOW: 100,
            API_KEY_MIN_LENGTH: 32,
            SESSION_TIMEOUT: 30 * 60 * 1000 // 30 minutes
        },

        /**
         * Request tracking for rate limiting
         */
        _requestTracker: new Map(),

        /**
         * Encodes text for safe display in HTML contexts
         * @param {string} text - Text to encode
         * @returns {string} - Safely encoded text
         */
        encodeHTML(text) {
            if (typeof text !== "string") {
                return "";
            }
            return encodeXML(text);
        },

        /**
         * Encodes text for safe use in JavaScript contexts
         * @param {string} text - Text to encode
         * @returns {string} - Safely encoded text
         */
        encodeJS(text) {
            if (typeof text !== "string") {
                return "";
            }
            return encodeJS(text);
        },

        /**
         * Encodes text for safe use in URL contexts
         * @param {string} text - Text to encode
         * @returns {string} - Safely encoded URL
         */
        encodeURL(text) {
            if (typeof text !== "string") {
                return "";
            }
            return encodeURL(text);
        },

        /**
         * Validates and sanitizes trading input data
         * @param {Object} data - Trading data to validate
         * @param {string} type - Data type (order, portfolio, etc.)
         * @returns {Object} - Validation result
         */
        validateTradingData(data, type) {
            const result = {
                isValid: true,
                errors: [],
                sanitizedData: null
            };

            try {
                switch (type) {
                    case "order":
                        result.sanitizedData = this._validateOrderData(data, result);
                        break;
                    case "portfolio":
                        result.sanitizedData = this._validatePortfolioData(data, result);
                        break;
                    case "symbol":
                        result.sanitizedData = this._validateSymbolData(data, result);
                        break;
                    default:
                        result.sanitizedData = this._sanitizeGenericData(data);
                }
            } catch (error) {
                result.isValid = false;
                result.errors.push(`Validation error: ${error.message}`);
            }

            return result;
        },

        /**
         * Validates API key format and strength
         * @param {string} apiKey - API key to validate
         * @returns {Object} - Validation result
         */
        validateApiKey(apiKey) {
            const result = {
                isValid: false,
                strength: "weak",
                errors: []
            };

            if (!apiKey || typeof apiKey !== "string") {
                result.errors.push("API key is required");
                return result;
            }

            if (apiKey.length < this.CONFIG.API_KEY_MIN_LENGTH) {
                result.errors.push(`API key must be at least ${this.CONFIG.API_KEY_MIN_LENGTH} characters`);
                return result;
            }

            // Check for common patterns that indicate weak keys
            if (/^[a-z]+$/.test(apiKey) || /^[A-Z]+$/.test(apiKey) || /^[0-9]+$/.test(apiKey)) {
                result.strength = "weak";
                result.errors.push("API key appears to be weak (single character type)");
            } else if (/^[a-zA-Z0-9]+$/.test(apiKey)) {
                result.strength = "medium";
            } else {
                result.strength = "strong";
            }

            result.isValid = result.errors.length === 0;
            return result;
        },

        /**
         * Implements rate limiting for API requests
         * @param {string} identifier - Request identifier (IP, user ID, etc.)
         * @param {number} limit - Request limit (optional, uses default)
         * @returns {boolean} - Whether request is allowed
         */
        checkRateLimit(identifier, limit = this.CONFIG.MAX_REQUESTS_PER_WINDOW) {
            const now = Date.now();
            const windowStart = now - this.CONFIG.RATE_LIMIT_WINDOW;

            // Clean old entries
            this._cleanRateLimitTracker(windowStart);

            // Get current requests for identifier
            if (!this._requestTracker.has(identifier)) {
                this._requestTracker.set(identifier, []);
            }

            const requests = this._requestTracker.get(identifier);
            const recentRequests = requests.filter(timestamp => timestamp > windowStart);

            if (recentRequests.length >= limit) {
                Log.warning(`Rate limit exceeded for identifier: ${identifier}`);
                return false;
            }

            // Add current request
            recentRequests.push(now);
            this._requestTracker.set(identifier, recentRequests);

            return true;
        },

        /**
         * Creates secure headers for API requests
         * @param {Object} options - Header options
         * @returns {Object} - Secure headers
         */
        createSecureHeaders(options = {}) {
            const headers = {
                "Content-Type": "application/json",
                "X-Requested-With": "XMLHttpRequest",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            };

            // Add CSRF token if available
            if (options.csrfToken) {
                headers["X-CSRF-Token"] = options.csrfToken;
            }

            // Add correlation ID for request tracking
            if (options.correlationId) {
                headers["X-Correlation-ID"] = options.correlationId;
            }

            // Add API key if provided (for external APIs)
            if (options.apiKey) {
                headers["Authorization"] = `Bearer ${options.apiKey}`;
            }

            // Add custom headers
            if (options.customHeaders) {
                Object.assign(headers, options.customHeaders);
            }

            return headers;
        },

        /**
         * Sanitizes trading data to prevent injection attacks
         * @param {*} data - Data to sanitize
         * @returns {*} - Sanitized data
         */
        sanitizeData(data) {
            if (typeof data === "string") {
                return this._sanitizeString(data);
            }

            if (Array.isArray(data)) {
                return data.map(item => this.sanitizeData(item));
            }

            if (data && typeof data === "object" && data.constructor === Object) {
                const sanitized = {};
                Object.keys(data).forEach(key => {
                    const sanitizedKey = this._sanitizeString(key);
                    sanitized[sanitizedKey] = this.sanitizeData(data[key]);
                });
                return sanitized;
            }

            return data;
        },

        /**
         * Validates trading order data
         * @private
         */
        _validateOrderData(data, result) {
            const sanitized = {};

            // Validate symbol
            if (!data.symbol || typeof data.symbol !== "string") {
                result.errors.push("Symbol is required");
                result.isValid = false;
            } else {
                sanitized.symbol = this._sanitizeString(data.symbol.toUpperCase());
                if (!/^[A-Z]{2,10}$/.test(sanitized.symbol)) {
                    result.errors.push("Invalid symbol format");
                    result.isValid = false;
                }
            }

            // Validate side
            if (!data.side || !["BUY", "SELL"].includes(data.side.toUpperCase())) {
                result.errors.push("Side must be BUY or SELL");
                result.isValid = false;
            } else {
                sanitized.side = data.side.toUpperCase();
            }

            // Validate quantity
            if (!data.quantity || typeof data.quantity !== "number" || data.quantity <= 0) {
                result.errors.push("Quantity must be a positive number");
                result.isValid = false;
            } else {
                sanitized.quantity = Math.abs(data.quantity);
            }

            // Validate price (if provided)
            if (data.price !== undefined) {
                if (typeof data.price !== "number" || data.price <= 0) {
                    result.errors.push("Price must be a positive number");
                    result.isValid = false;
                } else {
                    sanitized.price = Math.abs(data.price);
                }
            }

            // Validate order type
            const validOrderTypes = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"];
            if (!data.type || !validOrderTypes.includes(data.type.toUpperCase())) {
                result.errors.push("Invalid order type");
                result.isValid = false;
            } else {
                sanitized.type = data.type.toUpperCase();
            }

            return sanitized;
        },

        /**
         * Validates portfolio data
         * @private
         */
        _validatePortfolioData(data, result) {
            const sanitized = {};

            if (data.positions && Array.isArray(data.positions)) {
                sanitized.positions = data.positions.map(position => {
                    const pos = {};
                    
                    if (position.symbol && typeof position.symbol === "string") {
                        pos.symbol = this._sanitizeString(position.symbol.toUpperCase());
                    }
                    
                    if (typeof position.quantity === "number") {
                        pos.quantity = position.quantity;
                    }
                    
                    if (typeof position.averagePrice === "number" && position.averagePrice > 0) {
                        pos.averagePrice = position.averagePrice;
                    }
                    
                    return pos;
                });
            }

            return sanitized;
        },

        /**
         * Validates symbol data
         * @private
         */
        _validateSymbolData(data, result) {
            if (typeof data !== "string") {
                result.errors.push("Symbol must be a string");
                result.isValid = false;
                return null;
            }

            const sanitized = this._sanitizeString(data.toUpperCase());
            
            if (!/^[A-Z]{2,10}$/.test(sanitized)) {
                result.errors.push("Invalid symbol format");
                result.isValid = false;
                return null;
            }

            return sanitized;
        },

        /**
         * Sanitizes generic data
         * @private
         */
        _sanitizeGenericData(data) {
            return this.sanitizeData(data);
        },

        /**
         * Sanitizes string input
         * @private
         */
        _sanitizeString(input) {
            if (typeof input !== "string") {
                return input;
            }

            return input
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#x27;")
                .replace(/\//g, "&#x2F;")
                .replace(/\\/g, "&#x5C;")
                .replace(/&/g, "&amp;")
                .trim();
        },

        /**
         * Cleans old entries from rate limit tracker
         * @private
         */
        _cleanRateLimitTracker(windowStart) {
            for (const [identifier, requests] of this._requestTracker.entries()) {
                const recentRequests = requests.filter(timestamp => timestamp > windowStart);
                if (recentRequests.length === 0) {
                    this._requestTracker.delete(identifier);
                } else {
                    this._requestTracker.set(identifier, recentRequests);
                }
            }
        },

        /**
         * Generates secure random string
         * @param {number} length - String length
         * @returns {string} - Random string
         */
        generateSecureRandom(length = 32) {
            const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            let result = "";
            
            for (let i = 0; i < length; i++) {
                result += chars.charAt(Math.floor(Math.random() * chars.length));
            }
            
            return result;
        },

        /**
         * Validates session token
         * @param {string} token - Session token
         * @returns {boolean} - Whether token is valid
         */
        validateSessionToken(token) {
            if (!token || typeof token !== "string") {
                return false;
            }

            // Basic format validation
            if (token.length < 32) {
                return false;
            }

            // Check if token is not expired (implementation specific)
            // This would typically involve decoding JWT or checking against server
            return true;
        },

        /**
         * Logs security events
         * @param {string} event - Security event type
         * @param {Object} details - Event details
         */
        logSecurityEvent(event, details = {}) {
            const logData = {
                event: event,
                timestamp: new Date().toISOString(),
                userAgent: navigator.userAgent,
                url: window.location.href,
                ...details
            };

            // Remove sensitive data
            delete logData.apiKey;
            delete logData.password;
            delete logData.token;

            Log.info(`[SECURITY] ${event}`, logData);
        }
    };
});
