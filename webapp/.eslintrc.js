/**
 * ESLint configuration for Crypto Trading SAP UI5 application
 * Extends base configuration with UI5-specific rules and crypto trading patterns
 */

module.exports = {
    extends: ["eslint:recommended"],
    env: {
        browser: true,
        jquery: true,
        es6: true
    },
    parserOptions: {
        ecmaVersion: 2020,
        sourceType: "module"
    },
    globals: {
        sap: "readonly",
        jQuery: "readonly",
        QUnit: "readonly",
        sinon: "readonly",
        module: "readonly",
        require: "readonly",
        io: "readonly",
        WebSocket: "readonly",
        fetch: "readonly",
        Promise: "readonly"
    },
    rules: {
        // UI5 specific relaxations
        "no-unused-vars": ["error", {
            "args": "none",
            "varsIgnorePattern": "^_",
            "argsIgnorePattern": "^_|oEvent|oResponse|oData|oModel|oView|oController"
        }],
        "max-len": ["error", {
            "code": 120,
            "ignoreStrings": true,
            "ignoreTemplateLiterals": true,
            "ignoreRegExpLiterals": true,
            "ignoreUrls": true
        }],

        // Code quality rules
        "no-undef": "off", // UI5 global namespaces
        "radix": "error",
        "no-var": "error",
        "prefer-const": "error",
        "no-console": ["warn", { "allow": ["warn", "error"] }],
        "no-debugger": "error",
        "no-alert": "error",

        // Security rules for crypto trading
        "no-eval": "error",
        "no-implied-eval": "error",
        "no-new-func": "error",
        "no-script-url": "error",

        // Performance rules
        "no-loop-func": "error",
        "no-inner-declarations": "off", // UI5 patterns sometimes need this

        // Allow some UI5 specific patterns
        "no-new": "off", // UI5 sometimes uses new for side effects
        "no-case-declarations": "off", // Common in switch statements
        "no-prototype-builtins": "off", // hasOwnProperty is common in UI5
        "no-useless-escape": "off", // RegExp patterns may need escapes
        "no-shadow": "off", // UI5 variable shadowing is common
        "brace-style": "off", // UI5 formatting style

        // Keep important errors
        "no-unreachable": "error",
        "no-async-promise-executor": "error",
        "no-constant-condition": "error",
        "no-useless-catch": "error",
        "no-duplicate-case": "error",
        "no-empty": "error",
        "no-extra-boolean-cast": "error",
        "no-irregular-whitespace": "error",
        "no-sparse-arrays": "error",
        "use-isnan": "error",
        "valid-typeof": "error",

        // Best practices for crypto trading
        "eqeqeq": ["error", "always"],
        "no-floating-decimal": "error",
        "no-implicit-coercion": "error",
        "no-magic-numbers": "off", // Disabled - constants are defined in Constants.js
        "no-multi-spaces": "error",
        "no-redeclare": "error",
        "no-self-assign": "error",
        "no-self-compare": "error",
        "no-throw-literal": "error",
        "no-unused-expressions": "error",
        "no-useless-call": "error",
        "no-useless-concat": "error",
        "no-useless-return": "error",
        "prefer-promise-reject-errors": "error"
    },
    overrides: [
        {
            files: ["test/**/*.js", "**/*.qunit.js", "**/*test*.js"],
            env: {
                qunit: true,
                mocha: true,
                jest: true
            },
            globals: {
                before: "readonly",
                after: "readonly",
                beforeEach: "readonly",
                afterEach: "readonly",
                describe: "readonly",
                it: "readonly",
                expect: "readonly",
                assert: "readonly",
                chai: "readonly",
                should: "readonly"
            },
            rules: {
                "no-unused-vars": "off", // Test files often have unused vars
                "no-undef": "off", // Test globals
                "no-magic-numbers": "off", // Test data often uses magic numbers
                "max-len": "off" // Test descriptions can be long
            }
        },
        {
            files: ["utils/**/*.js", "extensions/**/*.js"],
            rules: {
                "no-console": "off", // Utilities may need console for debugging
                "no-magic-numbers": "off" // Utilities may have configuration constants
            }
        },
        {
            files: ["controller/**/*.js"],
            rules: {
                "max-lines-per-function": ["warn", 100], // Keep controller methods manageable
                "complexity": ["warn", 10] // Limit cyclomatic complexity
            }
        }
    ]
};
