from PIL import Image, ImageDraw, ImageFont
import os

def create_icon():
    # 256x256 icon
    size = 256
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Draw a rounded rectangle (Blue Gradient effect)
    margin = 20
    draw.rounded_rectangle([margin, margin, size-margin, size-margin], radius=50, fill=(30, 136, 229))
    
    # Draw a simple face/mask representation
    # Using simple shapes as a placeholder
    eye_size = 30
    draw.ellipse([80, 90, 80+eye_size, 90+eye_size], fill='white')
    draw.ellipse([146, 90, 146+eye_size, 90+eye_size], fill='white')
    
    # Smile
    draw.arc([80, 120, 176, 200], start=0, end=180, fill='white', width=10)

    # Save as ICO
    icon_path = "app_icon.ico"
    # ICO supports multiple sizes, PIL can handle this if we pass a list of images
    # but for a simple start, we just save the 256x256 one
    image.save(icon_path, format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])
    print(f"✅ Icon created: {icon_path}")

if __name__ == "__main__":
    create_icon()
