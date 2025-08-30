#!/usr/bin/env python3
"""
Create logo for the cryptocurrency trading platform
"""

import os
from PIL import Image, ImageDraw, ImageFont

def create_logo(size=200):
    """Create a cryptocurrency trading platform logo"""
    
    # Create image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw main circle (background)
    margin = 10
    draw.ellipse([margin, margin, size-margin, size-margin], 
                fill='#FF6B00', outline='#E55A00', width=3)
    
    # Draw inner design elements (cryptocurrency-like)
    center = size // 2
    
    # Draw Bitcoin-like symbol
    # Vertical lines
    line_width = int(size * 0.08)
    draw.rectangle([center - line_width//2 - size//6, center - size//3, 
                   center - line_width//2 - size//6 + line_width, center + size//3], 
                   fill='white')
    draw.rectangle([center - line_width//2 + size//6, center - size//3, 
                   center - line_width//2 + size//6 + line_width, center + size//3], 
                   fill='white')
    
    # Horizontal curves (simplified S shape)
    # Top curve
    draw.arc([center - size//4, center - size//4, center + size//4, center], 
             start=180, end=0, fill='white', width=line_width)
    # Bottom curve  
    draw.arc([center - size//4, center, center + size//4, center + size//4], 
             start=0, end=180, fill='white', width=line_width)
    
    # Add small "REX" text at bottom
    try:
        font = ImageFont.load_default()
        text = "РЕX"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (size - text_width) // 2
        text_y = size - margin - 20
        draw.text((text_x, text_y), text, fill='white', font=font)
    except:
        pass
    
    return img

def main():
    """Create logo image"""
    
    images_dir = "webapp/images"
    os.makedirs(images_dir, exist_ok=True)
    
    print("🎨 Creating platform logo...")
    
    # Create logo
    logo = create_logo(200)
    
    # Save as PNG and smaller favicon
    logo_path = os.path.join(images_dir, "logo.png")
    favicon_path = os.path.join(images_dir, "favicon.png")
    
    logo.save(logo_path, 'PNG')
    
    # Create smaller favicon
    favicon = logo.resize((32, 32), Image.Resampling.LANCZOS)
    favicon.save(favicon_path, 'PNG')
    
    print(f"  ✅ Saved: {logo_path}")
    print(f"  ✅ Saved: {favicon_path}")
    
    print("\n🎯 Logo creation completed!")

if __name__ == "__main__":
    main()