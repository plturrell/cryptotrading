sap.ui.define([
    "./BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (BaseController, JSONModel, MessageToast, MessageBox) {
    "use strict";

    return BaseController.extend("com.rex.cryptotrading.controller.Login", {
        
        /**
         * Controller initialization
         */
        onInit: function () {
            // Call parent onInit
            BaseController.prototype.onInit.apply(this, arguments);
            // Initialize models
            this._initModels();
            
            // Check for saved credentials
            this._checkSavedCredentials();
            
            // Set demo mode
            this.getView().getModel("settings").setProperty("/demoMode", true);
        },
        
        /**
         * Initialize data models
         */
        _initModels: function () {
            // Login model
            var oLoginModel = new JSONModel({
                username: "",
                password: "",
                rememberMe: false,
                showMessage: false,
                message: "",
                messageType: "None",
                loginEnabled: false
            });
            this.getView().setModel(oLoginModel, "login");
            
            // Settings model
            var oSettingsModel = new JSONModel({
                language: localStorage.getItem("app-language") || "en",
                demoMode: false
            });
            this.getView().setModel(oSettingsModel, "settings");
            
            // Demo users
            this.demoUsers = [
                { username: "craig", password: "Craig2024!", name: "Craig Wright", role: "Admin" },
                { username: "irina", password: "Irina2024!", name: "Irina Petrova", role: "Trader" },
                { username: "dasha", password: "Dasha2024!", name: "Dasha Ivanova", role: "Analyst" },
                { username: "dany", password: "Dany2024!", name: "Dany Chen", role: "Trader" }
            ];
        },
        
        /**
         * Check for saved credentials
         */
        _checkSavedCredentials: function () {
            var savedUsername = localStorage.getItem("savedUsername");
            if (savedUsername) {
                this.getView().getModel("login").setProperty("/username", savedUsername);
                this.getView().getModel("login").setProperty("/rememberMe", true);
            }
        },
        
        /**
         * Handle demo user selection
         */
        onUserSelect: function (oEvent) {
            var iIndex = oEvent.getParameter("selectedIndex");
            var oUser = this.demoUsers[iIndex];
            
            if (oUser) {
                var oModel = this.getView().getModel("login");
                oModel.setProperty("/username", oUser.username);
                oModel.setProperty("/password", oUser.password);
                oModel.setProperty("/loginEnabled", true);
                
                MessageToast.show("Selected: " + oUser.name + " (" + oUser.role + ")");
            }
        },
        
        /**
         * Handle input change
         */
        onInputChange: function () {
            var oModel = this.getView().getModel("login");
            var sUsername = oModel.getProperty("/username");
            var sPassword = oModel.getProperty("/password");
            
            // Enable login button if both fields have values
            oModel.setProperty("/loginEnabled", sUsername.length > 0 && sPassword.length > 0);
            
            // Clear any error messages
            if (oModel.getProperty("/showMessage")) {
                oModel.setProperty("/showMessage", false);
            }
        },
        
        /**
         * Handle login
         */
        onLogin: async function () {
            var oModel = this.getView().getModel("login");
            var sUsername = oModel.getProperty("/username");
            var sPassword = oModel.getProperty("/password");
            var bRememberMe = oModel.getProperty("/rememberMe");
            
            // Show loading
            sap.ui.core.BusyIndicator.show();
            
            try {
                // Call authentication service
                const response = await fetch("http://localhost:8001/api/auth/login", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        username: sUsername,
                        password: sPassword
                    })
                });
                
                const data = await response.json();
                
                if (response.ok && data.success && data.token) {
                    // Save credentials if remember me is checked
                    if (bRememberMe) {
                        localStorage.setItem("savedUsername", sUsername);
                    } else {
                        localStorage.removeItem("savedUsername");
                    }
                    
                    // Store token and user info
                    localStorage.setItem("authToken", data.token);
                    localStorage.setItem("sessionToken", data.session);
                    localStorage.setItem("currentUser", JSON.stringify(data.user));
                    
                    // Set user language preference
                    if (data.user.language) {
                        localStorage.setItem("app-language", data.user.language);
                    }
                    
                    // Show success message
                    MessageToast.show("Welcome, " + data.user.first_name + "!");
                    
                    // Navigate to main app
                    setTimeout(() => {
                        window.location.href = "index.html";
                    }, 1000);
                    
                } else {
                    // Show error
                    oModel.setProperty("/showMessage", true);
                    oModel.setProperty("/message", data.message || "Invalid username or password");
                    oModel.setProperty("/messageType", "Error");
                }
                
            } catch (error) {
                // For demo mode, check against demo users
                var oUser = this.demoUsers.find(u => 
                    u.username === sUsername && u.password === sPassword
                );
                
                if (oUser) {
                    // Demo login successful
                    if (bRememberMe) {
                        localStorage.setItem("savedUsername", sUsername);
                    }
                    
                    // Store demo user info
                    localStorage.setItem("currentUser", JSON.stringify({
                        username: oUser.username,
                        first_name: oUser.name.split(" ")[0],
                        last_name: oUser.name.split(" ")[1],
                        role: oUser.role.toLowerCase(),
                        language: oUser.username === "irina" || oUser.username === "dasha" ? "ru" : "en"
                    }));
                    
                    MessageToast.show("Welcome, " + oUser.name + "!");
                    
                    setTimeout(() => {
                        window.location.href = "index.html";
                    }, 1000);
                } else {
                    oModel.setProperty("/showMessage", true);
                    oModel.setProperty("/message", "Invalid username or password");
                    oModel.setProperty("/messageType", "Error");
                }
            } finally {
                sap.ui.core.BusyIndicator.hide();
            }
        },
        
        /**
         * Handle forgot password
         */
        onForgotPassword: function () {
            var oModel = this.getView().getModel("login");
            var sUsername = oModel.getProperty("/username");
            
            if (!sUsername) {
                MessageBox.information("Please enter your username first");
                return;
            }
            
            MessageBox.confirm(
                "Send password reset link to email associated with username '" + sUsername + "'?",
                {
                    title: "Reset Password",
                    onClose: function (oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            MessageToast.show("Password reset link sent to your email");
                        }
                    }
                }
            );
        },
        
        /**
         * Handle create account
         */
        onCreateAccount: function () {
            MessageBox.information("Please contact your administrator to create an account");
        },
        
        /**
         * Handle language change
         */
        onLanguageChange: function (oEvent) {
            var sLanguage = oEvent.getParameter("selectedItem").getKey();
            localStorage.setItem("app-language", sLanguage);
            
            // Reload to apply language
            window.location.reload();
        },
        
        /**
         * Handle OAuth logins
         */
        onGoogleLogin: function () {
            MessageToast.show("Google login not yet implemented");
        },
        
        onGithubLogin: function () {
            MessageToast.show("GitHub login not yet implemented");
        },
        
        onMicrosoftLogin: function () {
            MessageToast.show("Microsoft login not yet implemented");
        },
        
        /**
         * Handle footer links
         */
        onPrivacy: function () {
            window.open("https://rex.com/privacy", "_blank");
        },
        
        onTerms: function () {
            window.open("https://rex.com/terms", "_blank");
        }
    });
});