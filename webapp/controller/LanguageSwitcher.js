sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/Fragment",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function (Controller, Fragment, MessageToast, JSONModel) {
    "use strict";

    return Controller.extend("com.rex.trading.controller.LanguageSwitcher", {
        
        /**
         * Initialize language switcher
         */
        onInit: function () {
            // Available languages
            var oLanguages = {
                languages: [
                    { key: "en", text: "English", flag: "🇬🇧" },
                    { key: "ru", text: "Русский", flag: "🇷🇺" }
                ],
                currentLanguage: this._getCurrentLanguage()
            };
            
            var oModel = new JSONModel(oLanguages);
            this.getView().setModel(oModel, "languages");
            
            // Load appropriate i18n model
            this._loadI18nModel();
        },
        
        /**
         * Handle language change
         */
        onLanguageChange: function (oEvent) {
            var sSelectedLanguage = oEvent.getParameter("selectedItem").getKey();
            this._setLanguage(sSelectedLanguage);
            this._loadI18nModel();
            
            // Show confirmation
            var sMessage = sSelectedLanguage === "ru" ? 
                "Язык изменен на русский" : 
                "Language changed to English";
            MessageToast.show(sMessage);
            
            // Reload app to apply language change
            setTimeout(function() {
                window.location.reload();
            }, 1500);
        },
        
        /**
         * Get current language from localStorage or browser
         */
        _getCurrentLanguage: function () {
            var sStoredLang = localStorage.getItem("app-language");
            if (sStoredLang) {
                return sStoredLang;
            }
            
            // Detect browser language
            var sBrowserLang = navigator.language || navigator.userLanguage;
            if (sBrowserLang.startsWith("ru")) {
                return "ru";
            }
            return "en";
        },
        
        /**
         * Set language in localStorage
         */
        _setLanguage: function (sLanguage) {
            localStorage.setItem("app-language", sLanguage);
            
            // Update model
            var oModel = this.getView().getModel("languages");
            oModel.setProperty("/currentLanguage", sLanguage);
        },
        
        /**
         * Load i18n model based on current language
         */
        _loadI18nModel: function () {
            var sCurrentLang = this._getCurrentLanguage();
            var sI18nPath = "i18n/i18n";
            
            if (sCurrentLang === "ru") {
                sI18nPath = "i18n/i18n_ru";
            }
            
            var oI18nModel = new sap.ui.model.resource.ResourceModel({
                bundleName: "com.rex.trading." + sI18nPath,
                supportedLocales: ["en", "ru"],
                fallbackLocale: "en"
            });
            
            this.getView().setModel(oI18nModel, "i18n");
            sap.ui.getCore().setModel(oI18nModel, "i18n");
        },
        
        /**
         * Open language selector dialog
         */
        onOpenLanguageDialog: function () {
            var oView = this.getView();
            
            if (!this._pDialog) {
                this._pDialog = Fragment.load({
                    id: oView.getId(),
                    name: "com.rex.trading.fragment.LanguageDialog",
                    controller: this
                }).then(function (oDialog) {
                    oView.addDependent(oDialog);
                    return oDialog;
                });
            }
            
            this._pDialog.then(function(oDialog) {
                oDialog.open();
            });
        },
        
        /**
         * Close language dialog
         */
        onCloseLanguageDialog: function () {
            this.byId("languageDialog").close();
        },
        
        /**
         * Get translated text with AI service
         */
        getAITranslation: async function(sText, sTargetLang) {
            try {
                const response = await fetch('/api/translate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: sText,
                        source_lang: 'en',
                        target_lang: sTargetLang || this._getCurrentLanguage()
                    })
                });
                
                const data = await response.json();
                return data.translation;
            } catch (error) {
                console.error('Translation error:', error);
                return sText; // Return original text on error
            }
        },
        
        /**
         * Translate entire page dynamically
         */
        translatePage: async function() {
            var sTargetLang = this._getCurrentLanguage();
            if (sTargetLang === 'en') return; // No translation needed
            
            // Get all text elements
            var aElements = document.querySelectorAll('[data-translate="true"]');
            
            for (let element of aElements) {
                var sOriginalText = element.getAttribute('data-original-text') || element.textContent;
                element.setAttribute('data-original-text', sOriginalText);
                
                var sTranslated = await this.getAITranslation(sOriginalText, sTargetLang);
                element.textContent = sTranslated;
            }
        }
    });
});