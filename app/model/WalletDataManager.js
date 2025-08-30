sap.ui.define([
    "com/rex/cryptotrading/model/DataManager",
    "com/rex/cryptotrading/model/EventBusManager"
], function(DataManager, EventBusManager) {
    "use strict";

    /**
     * Wallet Data Manager - Specialized data manager for wallet operations
     * Follows SAP UI5 patterns with blockchain integration
     */
    return DataManager.extend("com.rex.cryptotrading.model.WalletDataManager", {
        
        constructor: function() {
            var oInitialData = this._getInitialWalletData();
            var oConfig = {
                enableChangeDetection: true,
                enableValidation: true,
                enableHistory: true, // Track wallet state changes
                maxHistorySize: 50
            };
            
            DataManager.prototype.constructor.call(this, "wallet", oInitialData, oConfig);
            
            this._oEventBusManager = new EventBusManager();
            this._initializeValidationSchema();
            this._setupEventHandlers();
        },
        
        /**
         * Connect wallet with provider
         * @param {string} sProvider - Wallet provider (metamask, walletconnect, etc.)
         * @param {string} sAddress - Wallet address
         * @param {string} sNetwork - Network name
         * @param {number} iChainId - Chain ID
         */
        connectWallet: function(sProvider, sAddress, sNetwork, iChainId) {
            if (!this._validateAddress(sAddress)) {
                throw new Error("Invalid wallet address");
            }
            
            var oConnectionData = {
                connected: true,
                provider: sProvider,
                address: sAddress,
                network: sNetwork || "mainnet",
                chainId: iChainId || 1,
                connectedAt: new Date().toISOString(),
                lastActivity: new Date().toISOString()
            };
            
            this.updateObject("/connection", oConnectionData);
            this.setProperty("/status", "connected");
            
            // Initialize empty balances for connected wallet
            this.setProperty("/balances", {});
            
            // Publish connection event
            this._oEventBusManager.publishWalletConnectionChanged(
                true, sAddress, sProvider
            );
        },
        
        /**
         * Disconnect wallet
         */
        disconnectWallet: function() {
            var sAddress = this.getProperty("/connection/address");
            var sProvider = this.getProperty("/connection/provider");
            
            this.setProperty("/connection", {
                connected: false,
                provider: null,
                address: null,
                network: null,
                chainId: null,
                connectedAt: null,
                disconnectedAt: new Date().toISOString()
            });
            
            this.setProperty("/status", "disconnected");
            this.setProperty("/balances", {});
            this.setProperty("/transactions", []);
            
            // Publish disconnection event
            this._oEventBusManager.publishWalletConnectionChanged(
                false, sAddress, sProvider
            );
        },
        
        /**
         * Update wallet balances
         * @param {Object} oBalances - Balance data by token/coin
         */
        updateBalances: function(oBalances) {
            if (!this.isConnected()) {
                console.warn("Cannot update balances: wallet not connected");
                return;
            }
            
            var oCurrentBalances = this.getProperty("/balances") || {};
            var oUpdatedBalances = Object.assign({}, oCurrentBalances);
            
            // Process each balance update
            Object.keys(oBalances).forEach(function(sToken) {
                var vBalance = oBalances[sToken];
                if (this._validateBalance(vBalance)) {
                    oUpdatedBalances[sToken] = {
                        amount: vBalance.amount || vBalance,
                        decimals: vBalance.decimals || 18,
                        symbol: vBalance.symbol || sToken,
                        lastUpdate: new Date().toISOString()
                    };
                }
            }.bind(this));
            
            this.setProperty("/balances", oUpdatedBalances);
            this.setProperty("/lastBalanceUpdate", new Date().toISOString());
            
            // Publish balance update event
            this._oEventBusManager.publishBalanceUpdated(oUpdatedBalances);
        },
        
        /**
         * Get wallet connection status
         * @returns {boolean} Connection status
         */
        isConnected: function() {
            return this.getProperty("/connection/connected") || false;
        },
        
        /**
         * Get wallet address
         * @returns {string} Wallet address
         */
        getAddress: function() {
            return this.getProperty("/connection/address");
        },
        
        /**
         * Get balance for specific token
         * @param {string} sToken - Token symbol
         * @returns {Object} Balance data
         */
        getBalance: function(sToken) {
            return this.getProperty("/balances/" + sToken) || { amount: 0 };
        },
        
        /**
         * Set wallet error
         * @param {Object|string} vError - Error data
         */
        setError: function(vError) {
            var oError = typeof vError === 'string' ? { message: vError } : vError;
            this.setProperty("/error", oError);
            
            if (oError) {
                this._oEventBusManager.publish(
                    this._oEventBusManager.CHANNELS.WALLET,
                    this._oEventBusManager.EVENTS.WALLET.ERROR_OCCURRED,
                    { error: oError }
                );
            }
        },
        
        /**
         * Initialize wallet data structure
         * @private
         */
        _getInitialWalletData: function() {
            return {
                connection: {
                    connected: false,
                    provider: null,
                    address: null,
                    network: null,
                    chainId: null,
                    connectedAt: null,
                    lastActivity: null
                },
                status: "disconnected",
                balances: {},
                lastBalanceUpdate: null,
                error: null
            };
        },
        
        /**
         * Validate wallet address
         * @private
         */
        _validateAddress: function(sAddress) {
            // Basic Ethereum address validation
            return sAddress && 
                   typeof sAddress === 'string' && 
                   sAddress.length === 42 && 
                   sAddress.startsWith('0x');
        },
        
        /**
         * Validate balance data
         * @private
         */
        _validateBalance: function(vBalance) {
            if (typeof vBalance === 'number') return vBalance >= 0;
            if (typeof vBalance === 'string') return !isNaN(parseFloat(vBalance));
            if (typeof vBalance === 'object') {
                return vBalance.amount !== undefined && !isNaN(parseFloat(vBalance.amount));
            }
            return false;
        },
        
        /**
         * Initialize validation schema
         * @private
         */
        _initializeValidationSchema: function() {
            this._oValidationSchema = {
                connection: {
                    type: 'object',
                    required: true
                },
                balances: {
                    type: 'object',
                    required: false
                }
            };
        },
        
        /**
         * Setup event handlers
         * @private
         */
        _setupEventHandlers: function() {
            // Listen for market data changes to update portfolio values
            this._oEventBusManager.subscribe(
                this._oEventBusManager.CHANNELS.MARKET,
                this._oEventBusManager.EVENTS.MARKET.DATA_UPDATED,
                this._onMarketDataUpdated.bind(this),
                this
            );
        },
        
        /**
         * Handle market data updates for portfolio calculation
         * @private
         */
        _onMarketDataUpdated: function(sChannel, sEvent, oData) {
            if (this.isConnected() && oData.marketData) {
                // Could update portfolio values here
            }
        },
        
        /**
         * Cleanup
         */
        destroy: function() {
            if (this._oEventBusManager) {
                this._oEventBusManager.destroy();
            }
            
            DataManager.prototype.destroy.apply(this, arguments);
        }
    });
});