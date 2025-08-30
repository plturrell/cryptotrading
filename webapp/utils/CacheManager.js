sap.ui.define([
    "sap/ui/base/Object"
], (BaseObject) => {
    "use strict";

    /**
     * Cache Manager for Crypto Trading Platform
     * Provides intelligent caching with trading-specific optimizations including:
     * - Multi-level caching (memory + session storage)
     * - Time-based and size-based eviction
     * - Cache warming for frequently accessed data
     * - Trading data prioritization
     * - Performance monitoring and analytics
     */
    return BaseObject.extend("com.rex.cryptotrading.utils.CacheManager", {

        constructor: function() {
            BaseObject.prototype.constructor.apply(this, arguments);
            this._memoryCache = new Map();
            this._cacheStats = {
                hits: 0,
                misses: 0,
                evictions: 0,
                totalRequests: 0
            };
            this._maxMemorySize = 50 * 1024 * 1024; // 50MB
            this._currentMemorySize = 0;
        },

        /**
         * Cache configuration
         */
        CONFIG: {
            DEFAULT_TTL: 5 * 60 * 1000, // 5 minutes
            MARKET_DATA_TTL: 30 * 1000, // 30 seconds
            PORTFOLIO_TTL: 60 * 1000, // 1 minute
            HISTORICAL_DATA_TTL: 15 * 60 * 1000, // 15 minutes
            SYMBOLS_TTL: 60 * 60 * 1000, // 1 hour
            MAX_MEMORY_SIZE: 50 * 1024 * 1024, // 50MB
            MAX_STORAGE_SIZE: 10 * 1024 * 1024, // 10MB for session storage
            CLEANUP_INTERVAL: 5 * 60 * 1000 // 5 minutes
        },

        /**
         * Cache priorities for eviction
         */
        PRIORITY: {
            CRITICAL: 1,    // Market data, portfolio
            HIGH: 2,        // Orders, positions
            MEDIUM: 3,      // Historical data
            LOW: 4          // Static data, symbols
        },

        /**
         * Initializes cache manager
         * @param {Object} config - Configuration options
         */
        initialize(config = {}) {
            this._config = { ...this.CONFIG, ...config };
            this._maxMemorySize = this._config.MAX_MEMORY_SIZE;
            
            // Start cleanup interval
            this._startCleanupInterval();
            
            // Warm up cache with essential data
            this._warmUpCache();
        },

        /**
         * Gets data from cache
         * @param {string} key - Cache key
         * @param {string} namespace - Cache namespace (optional)
         * @returns {*} - Cached data or null
         */
        get(key, namespace = "default") {
            const fullKey = this._buildKey(key, namespace);
            this._cacheStats.totalRequests++;

            // Check memory cache first
            const memoryItem = this._memoryCache.get(fullKey);
            if (memoryItem && this._isValid(memoryItem)) {
                this._cacheStats.hits++;
                memoryItem.lastAccessed = Date.now();
                memoryItem.accessCount++;
                return memoryItem.data;
            }

            // Check session storage
            const storageItem = this._getFromStorage(fullKey);
            if (storageItem && this._isValid(storageItem)) {
                this._cacheStats.hits++;
                // Promote to memory cache if frequently accessed
                if (storageItem.accessCount > 5) {
                    this._setInMemory(fullKey, storageItem);
                }
                return storageItem.data;
            }

            this._cacheStats.misses++;
            return null;
        },

        /**
         * Sets data in cache
         * @param {string} key - Cache key
         * @param {*} data - Data to cache
         * @param {Object} options - Cache options
         */
        set(key, data, options = {}) {
            const fullKey = this._buildKey(key, options.namespace || "default");
            const now = Date.now();
            
            const cacheItem = {
                data: data,
                createdAt: now,
                lastAccessed: now,
                accessCount: 1,
                ttl: options.ttl || this._getTTLForKey(key),
                priority: options.priority || this._getPriorityForKey(key),
                size: this._calculateSize(data)
            };

            // Set in memory cache
            this._setInMemory(fullKey, cacheItem);

            // Set in session storage for persistence
            if (cacheItem.priority <= this.PRIORITY.HIGH) {
                this._setInStorage(fullKey, cacheItem);
            }
        },

        /**
         * Removes data from cache
         * @param {string} key - Cache key
         * @param {string} namespace - Cache namespace (optional)
         */
        remove(key, namespace = "default") {
            const fullKey = this._buildKey(key, namespace);
            
            // Remove from memory
            const memoryItem = this._memoryCache.get(fullKey);
            if (memoryItem) {
                this._currentMemorySize -= memoryItem.size;
                this._memoryCache.delete(fullKey);
            }

            // Remove from storage
            this._removeFromStorage(fullKey);
        },

        /**
         * Clears all cache data
         * @param {string} namespace - Optional namespace to clear
         */
        clear(namespace = null) {
            if (namespace) {
                // Clear specific namespace
                const prefix = `${namespace}:`;
                for (const key of this._memoryCache.keys()) {
                    if (key.startsWith(prefix)) {
                        const item = this._memoryCache.get(key);
                        this._currentMemorySize -= item.size;
                        this._memoryCache.delete(key);
                    }
                }
                this._clearStorageNamespace(namespace);
            } else {
                // Clear all
                this._memoryCache.clear();
                this._currentMemorySize = 0;
                this._clearAllStorage();
            }
        },

        /**
         * Gets cache statistics
         * @returns {Object} - Cache statistics
         */
        getStats() {
            const hitRate = this._cacheStats.totalRequests > 0 ? 
                (this._cacheStats.hits / this._cacheStats.totalRequests * 100).toFixed(2) : 0;

            return {
                ...this._cacheStats,
                hitRate: `${hitRate}%`,
                memorySize: this._currentMemorySize,
                memoryItems: this._memoryCache.size,
                maxMemorySize: this._maxMemorySize,
                memoryUsage: `${((this._currentMemorySize / this._maxMemorySize) * 100).toFixed(2)}%`
            };
        },

        /**
         * Preloads frequently accessed data
         * @param {Array} keys - Keys to preload
         * @param {Function} dataLoader - Function to load data
         */
        async preload(keys, dataLoader) {
            const promises = keys.map(async (key) => {
                try {
                    const data = await dataLoader(key);
                    this.set(key, data, { priority: this.PRIORITY.HIGH });
                } catch (error) {
                    console.warn(`Failed to preload cache key: ${key}`, error);
                }
            });

            await Promise.allSettled(promises);
        },

        /**
         * Sets data in memory cache with size management
         * @private
         */
        _setInMemory(key, cacheItem) {
            // Check if we need to evict items
            while (this._currentMemorySize + cacheItem.size > this._maxMemorySize && this._memoryCache.size > 0) {
                this._evictLeastImportant();
            }

            // Remove existing item if present
            const existing = this._memoryCache.get(key);
            if (existing) {
                this._currentMemorySize -= existing.size;
            }

            // Add new item
            this._memoryCache.set(key, cacheItem);
            this._currentMemorySize += cacheItem.size;
        },

        /**
         * Evicts least important cache item
         * @private
         */
        _evictLeastImportant() {
            let evictKey = null;
            let lowestScore = Infinity;

            for (const [key, item] of this._memoryCache.entries()) {
                // Calculate importance score (lower = less important)
                const age = Date.now() - item.lastAccessed;
                const score = (item.priority * 1000) + (age / item.accessCount);

                if (score < lowestScore) {
                    lowestScore = score;
                    evictKey = key;
                }
            }

            if (evictKey) {
                const item = this._memoryCache.get(evictKey);
                this._currentMemorySize -= item.size;
                this._memoryCache.delete(evictKey);
                this._cacheStats.evictions++;
            }
        },

        /**
         * Gets data from session storage
         * @private
         */
        _getFromStorage(key) {
            try {
                const item = sessionStorage.getItem(`cache_${key}`);
                if (item) {
                    const parsed = JSON.parse(item);
                    parsed.accessCount++;
                    sessionStorage.setItem(`cache_${key}`, JSON.stringify(parsed));
                    return parsed;
                }
            } catch (error) {
                console.warn("Error reading from session storage:", error);
            }
            return null;
        },

        /**
         * Sets data in session storage
         * @private
         */
        _setInStorage(key, cacheItem) {
            try {
                // Check storage quota
                const serialized = JSON.stringify(cacheItem);
                if (serialized.length > this._config.MAX_STORAGE_SIZE) {
                    return; // Skip if too large
                }

                sessionStorage.setItem(`cache_${key}`, serialized);
            } catch (error) {
                if (error.name === 'QuotaExceededError') {
                    this._cleanupStorage();
                    // Try again after cleanup
                    try {
                        sessionStorage.setItem(`cache_${key}`, JSON.stringify(cacheItem));
                    } catch (retryError) {
                        console.warn("Failed to store in session storage after cleanup:", retryError);
                    }
                } else {
                    console.warn("Error writing to session storage:", error);
                }
            }
        },

        /**
         * Removes data from session storage
         * @private
         */
        _removeFromStorage(key) {
            try {
                sessionStorage.removeItem(`cache_${key}`);
            } catch (error) {
                console.warn("Error removing from session storage:", error);
            }
        },

        /**
         * Clears storage namespace
         * @private
         */
        _clearStorageNamespace(namespace) {
            try {
                const prefix = `cache_${namespace}:`;
                const keysToRemove = [];
                
                for (let i = 0; i < sessionStorage.length; i++) {
                    const key = sessionStorage.key(i);
                    if (key && key.startsWith(prefix)) {
                        keysToRemove.push(key);
                    }
                }
                
                keysToRemove.forEach(key => sessionStorage.removeItem(key));
            } catch (error) {
                console.warn("Error clearing storage namespace:", error);
            }
        },

        /**
         * Clears all storage
         * @private
         */
        _clearAllStorage() {
            try {
                const keysToRemove = [];
                
                for (let i = 0; i < sessionStorage.length; i++) {
                    const key = sessionStorage.key(i);
                    if (key && key.startsWith('cache_')) {
                        keysToRemove.push(key);
                    }
                }
                
                keysToRemove.forEach(key => sessionStorage.removeItem(key));
            } catch (error) {
                console.warn("Error clearing all storage:", error);
            }
        },

        /**
         * Cleans up expired storage items
         * @private
         */
        _cleanupStorage() {
            try {
                const now = Date.now();
                const keysToRemove = [];
                
                for (let i = 0; i < sessionStorage.length; i++) {
                    const key = sessionStorage.key(i);
                    if (key && key.startsWith('cache_')) {
                        try {
                            const item = JSON.parse(sessionStorage.getItem(key));
                            if (!this._isValid(item)) {
                                keysToRemove.push(key);
                            }
                        } catch (parseError) {
                            keysToRemove.push(key); // Remove corrupted items
                        }
                    }
                }
                
                keysToRemove.forEach(key => sessionStorage.removeItem(key));
            } catch (error) {
                console.warn("Error during storage cleanup:", error);
            }
        },

        /**
         * Builds full cache key
         * @private
         */
        _buildKey(key, namespace) {
            return `${namespace}:${key}`;
        },

        /**
         * Checks if cache item is valid
         * @private
         */
        _isValid(item) {
            if (!item || !item.createdAt) {
                return false;
            }
            
            const age = Date.now() - item.createdAt;
            return age < item.ttl;
        },

        /**
         * Gets TTL for specific key type
         * @private
         */
        _getTTLForKey(key) {
            if (key.includes('market') || key.includes('price')) {
                return this._config.MARKET_DATA_TTL;
            }
            if (key.includes('portfolio') || key.includes('balance')) {
                return this._config.PORTFOLIO_TTL;
            }
            if (key.includes('history') || key.includes('kline')) {
                return this._config.HISTORICAL_DATA_TTL;
            }
            if (key.includes('symbol')) {
                return this._config.SYMBOLS_TTL;
            }
            return this._config.DEFAULT_TTL;
        },

        /**
         * Gets priority for specific key type
         * @private
         */
        _getPriorityForKey(key) {
            if (key.includes('market') || key.includes('portfolio')) {
                return this.PRIORITY.CRITICAL;
            }
            if (key.includes('order') || key.includes('position')) {
                return this.PRIORITY.HIGH;
            }
            if (key.includes('history')) {
                return this.PRIORITY.MEDIUM;
            }
            return this.PRIORITY.LOW;
        },

        /**
         * Calculates approximate size of data
         * @private
         */
        _calculateSize(data) {
            try {
                return JSON.stringify(data).length * 2; // Rough estimate (UTF-16)
            } catch (error) {
                return 1000; // Default size estimate
            }
        },

        /**
         * Starts cleanup interval
         * @private
         */
        _startCleanupInterval() {
            setInterval(() => {
                this._cleanupExpiredItems();
                this._cleanupStorage();
            }, this._config.CLEANUP_INTERVAL);
        },

        /**
         * Cleans up expired memory items
         * @private
         */
        _cleanupExpiredItems() {
            const now = Date.now();
            const keysToRemove = [];

            for (const [key, item] of this._memoryCache.entries()) {
                if (!this._isValid(item)) {
                    keysToRemove.push(key);
                }
            }

            keysToRemove.forEach(key => {
                const item = this._memoryCache.get(key);
                this._currentMemorySize -= item.size;
                this._memoryCache.delete(key);
            });
        },

        /**
         * Warms up cache with essential data
         * @private
         */
        _warmUpCache() {
            // This would typically preload frequently accessed data
            // Implementation depends on specific trading platform needs
        },

        /**
         * Destroys cache manager and cleans up resources
         */
        destroy() {
            this.clear();
        }
    });
});
