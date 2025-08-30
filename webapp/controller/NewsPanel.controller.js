sap.ui.define([
    "./BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment"
], function (BaseController, JSONModel, MessageToast, MessageBox, Fragment) {
    "use strict";

    return BaseController.extend("com.rex.cryptotrading.controller.NewsPanel", {

        onInit: function () {
            // Call parent onInit
            BaseController.prototype.onInit.apply(this, arguments);
            // Initialize model
            var oModel = new JSONModel({
                articles: [],
                loading: false,
                hasError: false,
                errorMessage: "",
                selectedLanguage: "en",
                selectedCategory: "all",
                newsCount: 0,
                lastRefresh: null
            });
            this.getView().setModel(oModel);

            // Load initial news
            this._loadNews();

            // Set up auto-refresh every 5 minutes
            this._setupAutoRefresh();
        },

        onLanguageChange: function (oEvent) {
            var sSelectedLanguage = oEvent.getParameter("key");
            this.getView().getModel().setProperty("/selectedLanguage", sSelectedLanguage);
            this._loadNews();
            
            MessageToast.show(sSelectedLanguage === "ru" ? 
                "Переключено на русский язык" : 
                "Switched to English");
        },

        onCategoryChange: function (oEvent) {
            var sSelectedCategory = oEvent.getParameter("key");
            this.getView().getModel().setProperty("/selectedCategory", sSelectedCategory);
            this._loadNews();
        },

        onRefreshNews: function () {
            this._loadNews(true);
            MessageToast.show("Refreshing news...");
        },

        onFetchFreshNews: function () {
            this._fetchFreshNews();
        },

        onArticlePress: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext();
            var oArticle = oBindingContext.getObject();
            
            // Increment view count
            this._incrementViewCount(oArticle.id);
            
            // Open article in new tab
            if (oArticle.url) {
                window.open(oArticle.url, '_blank');
            }
        },

        onReadMore: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext();
            var oArticle = oBindingContext.getObject();
            
            if (oArticle.url) {
                window.open(oArticle.url, '_blank');
            } else {
                this._showArticleDetails(oArticle);
            }
        },

        onTranslateArticle: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext();
            var oArticle = oBindingContext.getObject();
            
            this._translateArticle(oArticle.id);
        },

        onShareArticle: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext();
            var oArticle = oBindingContext.getObject();
            
            if (navigator.share) {
                navigator.share({
                    title: oArticle.title,
                    text: oArticle.summary,
                    url: oArticle.url
                });
            } else {
                // Fallback: copy to clipboard
                navigator.clipboard.writeText(oArticle.url).then(function() {
                    MessageToast.show("Article URL copied to clipboard");
                });
            }
        },

        onBookmarkArticle: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext();
            var oArticle = oBindingContext.getObject();
            
            // Store bookmark in localStorage
            var aBookmarks = JSON.parse(localStorage.getItem("newsBookmarks") || "[]");
            var bAlreadyBookmarked = aBookmarks.some(function(bookmark) {
                return bookmark.id === oArticle.id;
            });
            
            if (!bAlreadyBookmarked) {
                aBookmarks.push({
                    id: oArticle.id,
                    title: oArticle.title,
                    url: oArticle.url,
                    bookmarkedAt: new Date().toISOString()
                });
                localStorage.setItem("newsBookmarks", JSON.stringify(aBookmarks));
                MessageToast.show("Article bookmarked");
            } else {
                MessageToast.show("Article already bookmarked");
            }
        },

        onLoadMore: function () {
            this._loadMoreNews();
        },

        onOpenSettings: function () {
            this._openSettingsDialog();
        },

        // Private methods

        _loadNews: function (bForceRefresh) {
            var oModel = this.getView().getModel();
            var sLanguage = oModel.getProperty("/selectedLanguage");
            var sCategory = oModel.getProperty("/selectedCategory");
            
            oModel.setProperty("/loading", true);
            oModel.setProperty("/hasError", false);
            
            var sUrl = "/api/news/ui/latest";
            var oParams = {
                limit: 20,
                language: sLanguage
            };
            
            if (sCategory !== "all") {
                if (sCategory === "russian_crypto") {
                    sUrl = "/api/news/ui/russian";
                } else {
                    oParams.category = sCategory;
                }
            }
            
            // Build query string
            var sQueryString = Object.keys(oParams).map(function(key) {
                return key + '=' + encodeURIComponent(oParams[key]);
            }).join('&');
            
            jQuery.ajax({
                url: sUrl + "?" + sQueryString,
                method: "GET",
                success: function (oData) {
                    oModel.setProperty("/articles", oData.articles || []);
                    oModel.setProperty("/newsCount", oData.count || 0);
                    oModel.setProperty("/lastRefresh", new Date());
                    oModel.setProperty("/loading", false);
                    
                    if (bForceRefresh) {
                        MessageToast.show("News refreshed successfully");
                    }
                }.bind(this),
                error: function (oError) {
                    oModel.setProperty("/loading", false);
                    oModel.setProperty("/hasError", true);
                    oModel.setProperty("/errorMessage", "Failed to load news: " + 
                        (oError.responseJSON?.error || oError.statusText));
                    MessageToast.show("Failed to load news");
                }.bind(this)
            });
        },

        _fetchFreshNews: function () {
            var oModel = this.getView().getModel();
            var sLanguage = oModel.getProperty("/selectedLanguage");
            
            MessageToast.show("Fetching fresh news from API...");
            
            jQuery.ajax({
                url: "/api/news/fetch/fresh",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    limit: 20,
                    language: sLanguage
                }),
                success: function (oData) {
                    if (oData.success) {
                        MessageToast.show(`Fetched ${oData.fetched_count} new articles`);
                        this._loadNews(); // Reload the news list
                    } else {
                        MessageToast.show("Failed to fetch fresh news: " + oData.error);
                    }
                }.bind(this),
                error: function (oError) {
                    MessageToast.show("Error fetching fresh news");
                }
            });
        },

        _translateArticle: function (sArticleId) {
            MessageToast.show("Translating article to Russian...");
            
            jQuery.ajax({
                url: "/api/news/translate/" + sArticleId,
                method: "POST",
                success: function (oData) {
                    if (oData.success) {
                        MessageToast.show("Article translated successfully");
                        this._loadNews(); // Reload to show translated version
                    } else {
                        MessageToast.show("Translation failed: " + oData.error);
                    }
                }.bind(this),
                error: function (oError) {
                    MessageToast.show("Translation error");
                }
            });
        },

        _incrementViewCount: function (sArticleId) {
            // Track article view
            jQuery.ajax({
                url: "/api/news/track/view",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    articleId: sArticleId,
                    timestamp: new Date().toISOString()
                }),
                success: function () {
                    // Silently increment view count
                },
                error: function () {
                    // Ignore tracking errors
                }
            });
        },

        _loadMoreNews: function () {
            var oModel = this.getView().getModel();
            var aCurrentArticles = oModel.getProperty("/articles");
            var sLanguage = oModel.getProperty("/selectedLanguage");
            var sCategory = oModel.getProperty("/selectedCategory");
            
            var sUrl = "/api/news/ui/latest";
            var oParams = {
                limit: 10,
                offset: aCurrentArticles.length,
                language: sLanguage
            };
            
            if (sCategory !== "all") {
                if (sCategory === "russian_crypto") {
                    sUrl = "/api/news/ui/russian";
                } else {
                    oParams.category = sCategory;
                }
            }
            
            var sQueryString = Object.keys(oParams).map(function(key) {
                return key + '=' + encodeURIComponent(oParams[key]);
            }).join('&');
            
            jQuery.ajax({
                url: sUrl + "?" + sQueryString,
                method: "GET",
                success: function (oData) {
                    var aNewArticles = aCurrentArticles.concat(oData.articles || []);
                    oModel.setProperty("/articles", aNewArticles);
                    oModel.setProperty("/newsCount", aNewArticles.length);
                    MessageToast.show(`Loaded ${oData.articles?.length || 0} more articles`);
                },
                error: function () {
                    MessageToast.show("Failed to load more articles");
                }
            });
        },

        _showArticleDetails: function (oArticle) {
            if (!this._oArticleDialog) {
                Fragment.load({
                    name: "com.rex.cryptotrading.view.fragments.ArticleDetails",
                    controller: this
                }).then(function (oDialog) {
                    this._oArticleDialog = oDialog;
                    this.getView().addDependent(this._oArticleDialog);
                    this._oArticleDialog.setModel(new JSONModel(oArticle));
                    this._oArticleDialog.open();
                }.bind(this));
            } else {
                this._oArticleDialog.setModel(new JSONModel(oArticle));
                this._oArticleDialog.open();
            }
        },

        _openSettingsDialog: function () {
            if (!this._oSettingsDialog) {
                Fragment.load({
                    name: "com.rex.cryptotrading.view.fragments.NewsSettings",
                    controller: this
                }).then(function (oDialog) {
                    this._oSettingsDialog = oDialog;
                    this.getView().addDependent(this._oSettingsDialog);
                    this._oSettingsDialog.open();
                }.bind(this));
            } else {
                this._oSettingsDialog.open();
            }
        },

        _setupAutoRefresh: function () {
            // Auto-refresh every 5 minutes
            setInterval(function () {
                var oModel = this.getView().getModel();
                if (!oModel.getProperty("/loading")) {
                    this._loadNews();
                }
            }.bind(this), 300000); // 5 minutes
        },

        // Dialog event handlers

        onCloseArticleDialog: function () {
            this._oArticleDialog.close();
        },

        onCloseSettingsDialog: function () {
            this._oSettingsDialog.close();
        }
    });
});
