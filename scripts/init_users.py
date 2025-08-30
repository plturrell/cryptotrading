#!/usr/bin/env python3
"""
Initialize the four initial users in the database
Run this script to set up Craig, Irina, Dasha, and Dany
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cryptotrading.infrastructure.auth.user_management import UserManagementService, UserRole

def init_users():
    """Initialize the four users with their profiles"""
    
    print("Initializing User Management System...")
    service = UserManagementService()
    
    print("\n✅ Database tables created successfully")
    print("\n👥 Creating initial users...")
    
    # The users are automatically created by the service
    # Let's verify they exist
    users = ["craig", "irina", "dasha", "dany"]
    
    for username in users:
        user = service.get_user_by_username(username)
        if user:
            print(f"\n✅ User: {user['first_name']} {user['last_name']}")
            print(f"   Username: {user['username']}")
            print(f"   Email: {user['email']}")
            print(f"   Role: {user['role']}")
            print(f"   Language: {user['language']}")
            print(f"   Status: {user['status']}")
            
            # Get preferences
            prefs = service.get_user_preferences(user['id'])
            if prefs:
                print(f"   Theme: {prefs['theme']}")
                print(f"   Trading View: {prefs['trading_view']}")
                if prefs.get('favorite_pairs'):
                    print(f"   Favorite Pairs: {', '.join(prefs['favorite_pairs'])}")
    
    print("\n" + "="*50)
    print("✅ User Management System Initialized Successfully!")
    print("="*50)
    
    print("\n📝 Login Credentials:")
    print("-"*30)
    print("Craig Wright (Admin):")
    print("  Username: craig")
    print("  Password: Craig2024!")
    print("")
    print("Irina Petrova (Trader):")
    print("  Username: irina")
    print("  Password: Irina2024!")
    print("")
    print("Dasha Ivanova (Analyst):")
    print("  Username: dasha")
    print("  Password: Dasha2024!")
    print("")
    print("Dany Chen (Trader):")
    print("  Username: dany")
    print("  Password: Dany2024!")
    print("-"*30)
    
    print("\n🚀 You can now login at: http://localhost:8080/login.html")
    print("   Or use the demo mode selector in the login screen")
    
    # Test authentication
    print("\n🔐 Testing authentication...")
    auth_result = service.authenticate("craig", "Craig2024!")
    if auth_result and auth_result.get("token"):
        print("✅ Authentication test successful!")
        print(f"   Token generated for: {auth_result['user']['first_name']}")
    else:
        print("❌ Authentication test failed")
    
    return True

if __name__ == "__main__":
    try:
        init_users()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)