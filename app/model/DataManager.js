sap.ui.define([
    "sap/ui/base/Object",
    "sap/ui/model/json/JSONModel",
    "sap/ui/core/EventBus"
], function(BaseObject, JSONModel, EventBus) {
    "use strict";

    /**
     * Enhanced Data Manager following SAP UI5 patterns
     * Manages structured data with change events and validation
     */
    return BaseObject.extend("com.rex.cryptotrading.model.DataManager", {
        
        constructor: function(sModelName, oInitialData, oConfig) {
            BaseObject.prototype.constructor.apply(this, arguments);
            
            this._sModelName = sModelName;
            this._oEventBus = EventBus.getInstance();
            this._oConfig = Object.assign({
                enableChangeDetection: true,
                enableValidation: false,
                enableHistory: false,
                maxHistorySize: 10
            }, oConfig || {});
            
            // Initialize model with structured data
            this._oModel = new JSONModel(oInitialData || {});
            this._oModel.setDefaultBindingMode("TwoWay");
            this._oModel.setSizeLimit(10000);
            
            // History tracking
            if (this._oConfig.enableHistory) {
                this._aHistory = [];
                this._iCurrentHistoryIndex = -1;
            }
            
            // Change detection
            if (this._oConfig.enableChangeDetection) {
                this._setupChangeDetection();
            }
        },
        
        /**
         * Get the JSONModel instance
         * @returns {sap.ui.model.json.JSONModel}
         */
        getModel: function() {
            return this._oModel;
        },
        
        /**
         * Get model name
         * @returns {string}
         */
        getModelName: function() {
            return this._sModelName;
        },
        
        /**
         * Set data with change notification
         * @param {Object} oData - Data to set
         * @param {boolean} bMerge - Whether to merge with existing data
         */
        setData: function(oData, bMerge) {
            var oOldData = this._oModel.getData();
            
            if (bMerge) {
                var oMergedData = Object.assign({}, oOldData, oData);
                this._oModel.setData(oMergedData);
            } else {
                this._oModel.setData(oData);
            }
            
            // Save to history
            if (this._oConfig.enableHistory) {
                this._saveToHistory(oOldData);
            }
            
            // Fire change event
            this._fireDataChanged("dataSet", {
                oldData: oOldData,
                newData: this._oModel.getData(),
                merged: bMerge
            });
        },
        
        /**
         * Update property with change notification
         * @param {string} sPath - Property path
         * @param {any} vValue - New value
         */
        setProperty: function(sPath, vValue) {
            var vOldValue = this._oModel.getProperty(sPath);
            
            if (vOldValue !== vValue) {
                this._oModel.setProperty(sPath, vValue);
                
                // Fire property change event
                this._firePropertyChanged(sPath, vOldValue, vValue);
            }
        },
        
        /**
         * Get property value
         * @param {string} sPath - Property path
         * @returns {any}
         */
        getProperty: function(sPath) {
            return this._oModel.getProperty(sPath);
        },
        
        /**
         * Update nested object with merge
         * @param {string} sPath - Object path
         * @param {Object} oUpdates - Updates to apply
         */
        updateObject: function(sPath, oUpdates) {
            var oCurrentObject = this.getProperty(sPath) || {};
            var oUpdatedObject = Object.assign({}, oCurrentObject, oUpdates);
            this.setProperty(sPath, oUpdatedObject);
        },
        
        /**
         * Add item to array
         * @param {string} sArrayPath - Array path
         * @param {any} vItem - Item to add
         * @param {number} iIndex - Optional index
         */
        addToArray: function(sArrayPath, vItem, iIndex) {
            var aArray = this.getProperty(sArrayPath) || [];
            var aNewArray = aArray.slice();
            
            if (typeof iIndex === 'number') {
                aNewArray.splice(iIndex, 0, vItem);
            } else {
                aNewArray.push(vItem);
            }
            
            this.setProperty(sArrayPath, aNewArray);
        },
        
        /**
         * Remove item from array
         * @param {string} sArrayPath - Array path
         * @param {number|function} vIdentifier - Index or predicate function
         */
        removeFromArray: function(sArrayPath, vIdentifier) {
            var aArray = this.getProperty(sArrayPath) || [];
            var aNewArray;
            
            if (typeof vIdentifier === 'number') {
                aNewArray = aArray.filter(function(item, index) { 
                    return index !== vIdentifier; 
                });
            } else if (typeof vIdentifier === 'function') {
                aNewArray = aArray.filter(function(item, index) { 
                    return !vIdentifier(item, index); 
                });
            }
            
            this.setProperty(sArrayPath, aNewArray);
        },
        
        /**
         * Refresh model
         */
        refresh: function() {
            this._oModel.refresh(true);
            this._fireDataChanged("dataRefreshed", {
                data: this._oModel.getData()
            });
        },
        
        /**
         * Setup change detection
         * @private
         */
        _setupChangeDetection: function() {
            // Attach to model's checkUpdate event
            this._oModel.attachEvent("checkUpdate", this._onModelUpdate.bind(this));
        },
        
        /**
         * Handle model update
         * @private
         */
        _onModelUpdate: function() {
            this._fireDataChanged("modelUpdated", {
                data: this._oModel.getData()
            });
        },
        
        /**
         * Fire data changed event
         * @private
         */
        _fireDataChanged: function(sReason, oData) {
            this._oEventBus.publish("DataManager", this._sModelName + ".dataChanged", Object.assign({
                modelName: this._sModelName,
                reason: sReason,
                timestamp: new Date()
            }, oData));
        },
        
        /**
         * Fire property changed event
         * @private
         */
        _firePropertyChanged: function(sPath, vOldValue, vNewValue) {
            this._oEventBus.publish("DataManager", this._sModelName + ".propertyChanged", {
                modelName: this._sModelName,
                path: sPath,
                oldValue: vOldValue,
                newValue: vNewValue,
                timestamp: new Date()
            });
        },
        
        /**
         * Save state to history
         * @private
         */
        _saveToHistory: function(oData) {
            if (!this._oConfig.enableHistory) return;
            
            // Remove future history if we're not at the end
            if (this._iCurrentHistoryIndex < this._aHistory.length - 1) {
                this._aHistory.splice(this._iCurrentHistoryIndex + 1);
            }
            
            // Add new entry
            this._aHistory.push({
                data: JSON.parse(JSON.stringify(oData)), // Deep clone
                timestamp: new Date()
            });
            
            // Limit history size
            if (this._aHistory.length > this._oConfig.maxHistorySize) {
                this._aHistory.shift();
            } else {
                this._iCurrentHistoryIndex++;
            }
        },
        
        /**
         * Cleanup
         */
        destroy: function() {
            if (this._oModel) {
                this._oModel.destroy();
            }
            BaseObject.prototype.destroy.apply(this, arguments);
        }
    });
});