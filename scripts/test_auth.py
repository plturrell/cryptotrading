#!/usr/bin/env python3
"""
Test authentication for all users
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cryptotrading.infrastructure.auth.user_management import UserManagementService


def test_authentication():
    """Test authentication for all users"""

    service = UserManagementService()

    users = [
        {"username": "craig", "password": "Craig2024!", "name": "Craig Wright"},
        {"username": "irina", "password": "Irina2024!", "name": "Irina Petrova"},
        {"username": "dasha", "password": "Dasha2024!", "name": "Dasha Ivanova"},
        {"username": "dany", "password": "Dany2024!", "name": "Dany Chen"},
    ]

    print("🔐 Testing Authentication for All Users")
    print("=" * 50)

    for user in users:
        print(f"\nTesting {user['name']} ({user['username']})...")

        try:
            auth_result = service.authenticate(user["username"], user["password"])

            if auth_result and auth_result.get("token"):
                print(f"✅ Authentication successful!")
                print(f"   Token: {auth_result['token'][:20]}...")
                print(f"   Session: {auth_result['session'][:20]}...")
                print(f"   User ID: {auth_result['user']['id']}")
                print(f"   Role: {auth_result['user']['role']}")
            else:
                print(f"❌ Authentication failed: {auth_result}")

        except Exception as e:
            print(f"❌ Error during authentication: {e}")

    print("\n" + "=" * 50)
    print("🏁 Authentication tests completed!")


if __name__ == "__main__":
    test_authentication()
