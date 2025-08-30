sap.ui.predefine("com/rex/cryptotrading/Component", [
    "sap/ui/core/UIComponent",
    "sap/ui/Device"
], function (UIComponent, Device) {
    "use strict";
    
    return UIComponent.extend("com.rex.cryptotrading.Component", {
        metadata: {
            manifest: "json"
        },
        
        init: function () {
            UIComponent.prototype.init.apply(this, arguments);
            this.getRouter().initialize();
        }
    });
});