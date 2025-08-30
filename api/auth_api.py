#!/usr/bin/env python3
"""
Authentication API for Fiori app
"""

import sys
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.cryptotrading.infrastructure.auth.user_management import UserManagementService

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize user service
user_service = UserManagementService()


@app.route("/api/auth/login", methods=["POST"])
def login():
    """Authenticate user and return JWT token"""
    try:
        data = request.get_json()

        if not data or not data.get("username") or not data.get("password"):
            return jsonify({"success": False, "message": "Username and password are required"}), 400

        username = data["username"]
        password = data["password"]

        # Authenticate user
        auth_result = user_service.authenticate(username, password)

        if auth_result and auth_result.get("token"):
            # Return successful login
            return jsonify(
                {
                    "success": True,
                    "token": auth_result["token"],
                    "session": auth_result["session"],
                    "user": {
                        "id": auth_result["user"]["id"],
                        "username": auth_result["user"]["username"],
                        "first_name": auth_result["user"]["first_name"],
                        "last_name": auth_result["user"]["last_name"],
                        "email": auth_result["user"]["email"],
                        "role": auth_result["user"]["role"],
                        "language": auth_result["user"]["language"],
                        "avatar_url": auth_result["user"].get("avatar_url", ""),
                    },
                }
            )
        else:
            return jsonify({"success": False, "message": "Invalid username or password"}), 401

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"success": False, "message": "Internal server error"}), 500


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    """Logout user and invalidate session"""
    try:
        data = request.get_json()
        session_token = data.get("session_token") if data else None

        if session_token:
            # Invalidate session
            user_service.logout_user(session_token)

        return jsonify({"success": True, "message": "Logged out successfully"})

    except Exception as e:
        print(f"Logout error: {e}")
        return jsonify({"success": False, "message": "Logout failed"}), 500


@app.route("/api/auth/validate", methods=["POST"])
def validate_token():
    """Validate JWT token"""
    try:
        data = request.get_json()
        token = data.get("token") if data else None

        if not token:
            return jsonify({"valid": False, "message": "Token required"}), 400

        # Validate token
        user_data = user_service.validate_token(token)

        if user_data:
            return jsonify({"valid": True, "user": user_data})
        else:
            return jsonify({"valid": False, "message": "Invalid or expired token"}), 401

    except Exception as e:
        print(f"Token validation error: {e}")
        return jsonify({"valid": False, "message": "Token validation failed"}), 500


@app.route("/api/users/profile/<int:user_id>", methods=["GET"])
def get_user_profile(user_id):
    """Get user profile data"""
    try:
        # Get user by ID
        user = user_service.get_user_by_id(user_id)

        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404

        # Get user preferences
        preferences = user_service.get_user_preferences(user_id)

        return jsonify(
            {
                "success": True,
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "first_name": user["first_name"],
                    "last_name": user["last_name"],
                    "email": user["email"],
                    "role": user["role"],
                    "language": user["language"],
                    "timezone": user["timezone"],
                    "phone": user.get("phone", ""),
                    "avatar_url": user.get("avatar_url", ""),
                    "two_factor_enabled": bool(user.get("two_factor_enabled", 0)),
                    "status": user["status"],
                    "last_login": user.get("last_login"),
                    "created_at": user.get("created_at"),
                },
                "preferences": preferences or {},
            }
        )

    except Exception as e:
        print(f"Profile error: {e}")
        return jsonify({"success": False, "message": "Failed to get user profile"}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Authentication API",
        }
    )


if __name__ == "__main__":
    print("🚀 Starting Authentication API server...")
    print(f"📊 Database: {user_service.db_path}")

    # Check if users exist
    craig = user_service.get_user_by_username("craig")
    print(f"👤 Users initialized: {'✅' if craig else '❌'}")

    app.run(host="0.0.0.0", port=8001, debug=True)
