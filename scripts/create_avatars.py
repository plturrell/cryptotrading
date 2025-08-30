#!/usr/bin/env python3
"""
Create avatar images for the four users
"""

import os
from PIL import Image, ImageDraw, ImageFont
import random

# User data
users = [
    {"username": "craig", "name": "Craig Wright", "initials": "CW", "color": "#2196F3"},
    {"username": "irina", "name": "Irina Petrova", "initials": "IP", "color": "#E91E63"},
    {"username": "dasha", "name": "Dasha Ivanova", "initials": "DI", "color": "#9C27B0"},
    {"username": "dany", "name": "Dany Chen", "initials": "DC", "color": "#4CAF50"}
]

def create_avatar(username, initials, color, size=200):
    """Create a circular avatar with initials"""
    
    # Create image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw circle background
    margin = 5
    draw.ellipse([margin, margin, size-margin, size-margin], fill=color)
    
    # Try to load a font, fallback to default
    try:
        # Try different font paths
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/Windows/Fonts/arial.ttf"
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, int(size * 0.4))
                break
        
        if font is None:
            font = ImageFont.load_default()
            
    except:
        font = ImageFont.load_default()
    
    # Get text size and position
    bbox = draw.textbbox((0, 0), initials, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    text_x = (size - text_width) // 2
    text_y = (size - text_height) // 2 - bbox[1]
    
    # Draw white text
    draw.text((text_x, text_y), initials, fill='white', font=font)
    
    return img

def main():
    """Create all user avatars"""
    
    avatars_dir = "webapp/images/avatars"
    os.makedirs(avatars_dir, exist_ok=True)
    
    print("🎨 Creating user avatars...")
    
    for user in users:
        print(f"Creating avatar for {user['name']} ({user['username']})...")
        
        # Create avatar image
        avatar = create_avatar(user['username'], user['initials'], user['color'])
        
        # Save as PNG (better quality) and JPG (smaller size)
        png_path = os.path.join(avatars_dir, f"{user['username']}.png")
        jpg_path = os.path.join(avatars_dir, f"{user['username']}.jpg")
        
        avatar.save(png_path, 'PNG', quality=95)
        
        # Convert RGBA to RGB for JPEG
        rgb_avatar = Image.new('RGB', avatar.size, (255, 255, 255))
        rgb_avatar.paste(avatar, mask=avatar.split()[-1])
        rgb_avatar.save(jpg_path, 'JPEG', quality=90)
        
        print(f"  ✅ Saved: {png_path}")
        print(f"  ✅ Saved: {jpg_path}")
    
    # Create a generic default avatar
    print("Creating default avatar...")
    default_avatar = create_avatar("default", "?", "#757575")
    default_avatar.save(os.path.join(avatars_dir, "default.png"), 'PNG')
    
    # Convert to RGB for JPEG
    rgb_default = Image.new('RGB', default_avatar.size, (255, 255, 255))
    rgb_default.paste(default_avatar, mask=default_avatar.split()[-1])
    rgb_default.save(os.path.join(avatars_dir, "default.jpg"), 'JPEG')
    
    print("\n🎯 Avatar creation completed!")
    print(f"📂 Avatars saved to: {avatars_dir}")
    
    # List created files
    print("\n📋 Created files:")
    for file in os.listdir(avatars_dir):
        file_path = os.path.join(avatars_dir, file)
        size = os.path.getsize(file_path)
        print(f"   {file} ({size:,} bytes)")

if __name__ == "__main__":
    main()