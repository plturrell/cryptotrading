"""
Shared routes for Flask applications
Consolidates common routes used by both app.py and app_vercel.py
"""

import logging

from flask import Blueprint, jsonify, send_from_directory

logger = logging.getLogger(__name__)

# Create blueprints for different route groups
ui5_bp = Blueprint("ui5", __name__)
health_bp = Blueprint("health", __name__)


# SAP UI5 routes
@ui5_bp.route("/")
def index():
    """Serve SAP UI5 application"""
    return send_from_directory("webapp", "index.html")


@ui5_bp.route("/manifest.json")
def manifest():
    """Serve SAP UI5 manifest"""
    return send_from_directory(".", "manifest.json")


@ui5_bp.route("/webapp/<path:filename>")
def webapp_files(filename):
    """Serve SAP UI5 webapp files"""
    return send_from_directory("webapp", filename)


# Health check route
@health_bp.route("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "platform": "cryptotrading.com", "version": "0.1.0"}


# Common error handlers
def register_error_handlers(app):
    """Register common error handlers on the Flask app"""

    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Internal server error"}, 500


def register_shared_routes(app, include_ui5=True, include_health=True):
    """
    Register shared routes on a Flask application

    Args:
        app: Flask application instance
        include_ui5: Whether to include SAP UI5 routes
        include_health: Whether to include health check route
    """
    if include_ui5:
        app.register_blueprint(ui5_bp)
        logger.info("Registered SAP UI5 routes")

    if include_health:
        app.register_blueprint(health_bp)
        logger.info("Registered health check route")

    # Always register error handlers
    register_error_handlers(app)
    logger.info("Registered error handlers")
