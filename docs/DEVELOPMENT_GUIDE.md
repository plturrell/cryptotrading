# Development Guide - Crypto Trading Application

## Code Standards and Best Practices

### 1. Controller Development

#### BaseController Usage
All controllers MUST extend from BaseController to ensure consistent behavior:

```javascript
sap.ui.define([
    "./BaseController",
    // other dependencies
], function (BaseController, ...) {
    "use strict";

    return BaseController.extend("com.rex.cryptotrading.controller.MyController", {
        
        onInit: function () {
            // ALWAYS call parent onInit first
            BaseController.prototype.onInit.apply(this, arguments);
            
            // Controller-specific initialization
            this._initializeLocalModels();
            this._setupEventHandlers();
        },

        onExit: function () {
            // Cleanup logic
            this._cleanupResources();
            
            // Call parent onExit
            BaseController.prototype.onExit.apply(this, arguments);
        }
    });
});
```

#### Navigation Standards
Use BaseController navigation methods instead of direct router calls:

```javascript
// ✅ Correct - Use BaseController methods
this.navigateToRoute("marketOverview", { symbol: "BTC" });

// ❌ Incorrect - Direct router usage
this.getRouter().navTo("marketOverview", { symbol: "BTC" });
```

#### Error Handling Standards
Always use SharedErrorHandlingUtils for consistent error handling:

```javascript
// ✅ Correct - Centralized error handling
try {
    var result = await this.performOperation();
} catch (error) {
    this.getSharedErrorHandlingUtils().handleError(
        error, 
        "Operation failed", 
        this.getView()
    );
}

// ❌ Incorrect - Direct error handling
catch (error) {
    MessageBox.error("Something went wrong");
}
```

### 2. Service Integration

#### Using Shared Services
Access services through the component, not direct instantiation:

```javascript
// ✅ Correct - Use component services
var oMarketDataService = this.getOwnerComponent().getMarketDataService();
var oTradingService = this.getOwnerComponent().getTradingService();

// ❌ Incorrect - Direct instantiation
var oService = new MarketDataService();
```

#### Secure API Calls
Always use BaseController secure request methods:

```javascript
// ✅ Correct - Secure requests
this.makeSecureRequest("/api/market-data", {
    method: "GET",
    headers: { "Accept": "application/json" }
}).then(function(data) {
    // Handle response
});

// ❌ Incorrect - Direct fetch/jQuery
fetch("/api/market-data").then(response => response.json());
```

### 3. Model Management

#### Model Naming Conventions
- **Global Models**: Use descriptive names (e.g., "marketData", "portfolio", "userSettings")
- **Local Models**: Use "local" prefix (e.g., "localUI", "localFilters")
- **UI State Models**: Use "ui" suffix (e.g., "dashboardUI", "tradingUI")

```javascript
// ✅ Correct model setup
onInit: function () {
    BaseController.prototype.onInit.apply(this, arguments);
    
    // Local UI model for view state
    var oUIModel = new JSONModel({
        isLoading: false,
        hasError: false,
        selectedTab: "overview"
    });
    this.setModel(oUIModel, "localUI");
    
    // Get global models from component
    var oMarketModel = this.getOwnerComponent().getModel("marketData");
}
```

### 4. Extension Development

#### Creating Extensions
Follow the established plugin pattern:

```javascript
sap.ui.define([
    "sap/ui/base/Object",
    "sap/base/Log"
], function (BaseObject, Log) {
    "use strict";

    return BaseObject.extend("com.rex.cryptotrading.extensions.plugins.MyPlugin", {
        
        constructor: function (oConfig) {
            BaseObject.prototype.constructor.apply(this, arguments);
            this._oConfig = oConfig || {};
            this._bInitialized = false;
        },

        init: function () {
            if (this._bInitialized) {
                return;
            }
            
            Log.info("Initializing MyPlugin");
            this._setupPlugin();
            this._bInitialized = true;
        },

        // Plugin methods...

        destroy: function () {
            this._cleanup();
            this._bInitialized = false;
            BaseObject.prototype.destroy.apply(this, arguments);
        }
    });
});
```

#### Extension Registration
Register extensions in Component.js initialization:

```javascript
_initExtensions: function () {
    var oExtensionManager = this.getExtensionManager();
    
    // Register custom extension
    oExtensionManager.registerExtension("myPlugin", new MyPlugin({
        apiEndpoint: "/api/my-service"
    }), {
        dependencies: ["marketData"],
        autoInit: true
    });
}
```

### 5. Performance Guidelines

#### Memory Management
- Always clean up intervals and timeouts in onExit
- Use BaseController interval tracking for automatic cleanup
- Destroy models and bindings properly

```javascript
onInit: function () {
    BaseController.prototype.onInit.apply(this, arguments);
    
    // ✅ Correct - Use BaseController interval tracking
    this.setManagedInterval(function() {
        this._refreshData();
    }.bind(this), 30000);
},

onExit: function () {
    // Cleanup is handled automatically by BaseController
    BaseController.prototype.onExit.apply(this, arguments);
}
```

#### Data Binding Optimization
- Use one-way binding when data doesn't change
- Implement lazy loading for large datasets
- Use aggregation binding for lists

```javascript
// ✅ Correct - Optimized binding
var oTable = this.byId("marketTable");
oTable.bindItems({
    path: "/marketData",
    template: oTemplate,
    growing: true,
    growingThreshold: 50
});
```

### 6. Testing Standards

#### Unit Test Structure
```javascript
QUnit.module("MyController", {
    beforeEach: function () {
        this.oController = new MyController();
        this.oView = sap.ui.xmlview({
            viewName: "com.rex.cryptotrading.view.MyView"
        });
        this.oController.setView(this.oView);
    },
    
    afterEach: function () {
        this.oView.destroy();
        this.oController.destroy();
    }
});

QUnit.test("Should initialize correctly", function (assert) {
    // Arrange
    var done = assert.async();
    
    // Act
    this.oController.onInit();
    
    // Assert
    assert.ok(this.oController.getModel("localUI"), "Local UI model created");
    done();
});
```

#### Integration Test Guidelines
- Test complete user workflows
- Mock external services
- Verify error handling paths
- Test responsive behavior

### 7. Security Best Practices

#### Input Validation
```javascript
// ✅ Correct - Always validate and sanitize
onSubmitOrder: function (oEvent) {
    var sSymbol = oEvent.getParameter("symbol");
    var fQuantity = oEvent.getParameter("quantity");
    
    // Validate inputs
    if (!sSymbol || !this._isValidSymbol(sSymbol)) {
        this.showErrorMessage("Invalid trading symbol");
        return;
    }
    
    // Sanitize inputs
    var oSecurityUtils = this.getSharedSecurityUtils();
    sSymbol = oSecurityUtils.sanitizeInput(sSymbol);
    
    // Proceed with order
    this._executeOrder(sSymbol, fQuantity);
}
```

#### API Security
- Always use HTTPS endpoints
- Include CSRF tokens for state-changing operations
- Validate server responses
- Handle authentication errors gracefully

### 8. Documentation Standards

#### JSDoc Comments
All public methods must have JSDoc comments:

```javascript
/**
 * Executes a trading order with risk validation
 * @public
 * @param {Object} oOrder Order details
 * @param {string} oOrder.symbol Trading symbol (e.g., "BTC/USD")
 * @param {number} oOrder.quantity Order quantity
 * @param {string} oOrder.type Order type ("buy" or "sell")
 * @returns {Promise<Object>} Promise resolving to order result
 * @throws {Error} When order validation fails
 */
executeOrder: function (oOrder) {
    // Implementation
}
```

#### Code Comments
- Explain complex business logic
- Document workarounds and technical debt
- Include references to external documentation
- Use TODO comments for future improvements

### 9. Build and Deployment

#### Build Configuration
The ui5.yaml file is configured for enterprise builds:
- Minification enabled for production
- Source maps for debugging
- Resource bundling for performance
- Theme compilation

#### Deployment Checklist
- [ ] All tests passing
- [ ] Code linting clean
- [ ] Performance benchmarks met
- [ ] Security scan completed
- [ ] Documentation updated
- [ ] Version number incremented

### 10. Troubleshooting

#### Common Issues and Solutions

**Issue**: Controller not extending BaseController properly
```javascript
// ❌ Problem
return Controller.extend("...", {

// ✅ Solution
return BaseController.extend("...", {
```

**Issue**: Memory leaks from intervals
```javascript
// ❌ Problem
setInterval(function() { ... }, 1000);

// ✅ Solution
this.setManagedInterval(function() { ... }, 1000);
```

**Issue**: Inconsistent error handling
```javascript
// ❌ Problem
MessageBox.error("Error occurred");

// ✅ Solution
this.getSharedErrorHandlingUtils().handleError(error, "Operation failed", this.getView());
```

### 11. Code Review Checklist

Before submitting code for review, ensure:
- [ ] Extends BaseController (for controllers)
- [ ] Uses shared utilities for error handling and security
- [ ] Includes proper JSDoc documentation
- [ ] Has corresponding unit tests
- [ ] Follows naming conventions
- [ ] Implements proper cleanup in onExit
- [ ] Uses secure API methods
- [ ] Validates user inputs
- [ ] Handles error cases gracefully
- [ ] Optimized for performance

This development guide ensures consistency and quality across the entire Crypto Trading application codebase.
