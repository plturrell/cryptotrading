module.exports = function(config) {
    config.set({
        frameworks: ["ui5"],
        ui5: {
            type: "application",
            config: {
                async: true,
                resourceroots: {
                    "com.rex.trading": "./webapp",
                    "com.rex.trading.test": "./webapp/test"
                }
            },
            tests: [
                "com/rex/trading/test/unit/AllTests",
                "com/rex/trading/test/integration/AllJourneys"
            ]
        },
        browsers: ["Chrome"],
        browserConsoleLogOptions: {
            level: "error"
        },
        coverageReporter: {
            includeAllSources: true,
            reporters: [
                {
                    type: "html",
                    dir: "coverage/"
                },
                {
                    type: "text"
                },
                {
                    type: "lcov",
                    dir: "coverage/"
                }
            ],
            check: {
                each: {
                    statements: 80,
                    branches: 75,
                    functions: 80,
                    lines: 80
                }
            }
        },
        junitReporter: {
            outputDir: "test-results/",
            outputFile: "karma-results.xml",
            useBrowserName: false
        },
        reporters: ["progress", "coverage", "junit"],
        preprocessors: {
            "webapp/!(test)/**/*.js": ["coverage"]
        },
        proxies: {
            "/resources/": "/base/resources/",
            "/test-resources/": "/base/test-resources/"
        },
        urlRoot: "/",
        autoWatch: true,
        singleRun: true,
        client: {
            clearContext: false,
            qunit: {
                showUI: true,
                testTimeout: 90000
            }
        }
    });
};
