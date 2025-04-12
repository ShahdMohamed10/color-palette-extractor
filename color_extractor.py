import numpy as np
from PIL import Image
from collections import Counter
import os

def load_image(image_file):
    """Load an image from a file object and return it as a PIL Image."""
    try:
        img = Image.open(image_file)
        
        # Ensure the image is in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image for processing
        img = img.resize((100, 100))
        
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        # Create a fallback image (2x2 color grid)
        fallback = Image.new('RGB', (2, 2))
        fallback.putpixel((0, 0), (255, 0, 0))  # Red
        fallback.putpixel((1, 0), (0, 255, 0))  # Green
        fallback.putpixel((0, 1), (0, 0, 255))  # Blue
        fallback.putpixel((1, 1), (255, 255, 0))  # Yellow
        return fallback

def extract_colors(img, num_colors=5):
    """
    Extract dominant colors using PIL's quantize method instead of K-means.
    Much simpler and more reliable than scikit-learn.
    
    Args:
        img: PIL Image
        num_colors: Number of colors to extract
        
    Returns:
        List of colors in RGB format
    """
    try:
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Use PIL's quantize to reduce colors (much simpler than K-means)
        img_quantized = img.quantize(colors=num_colors * 2, method=2)
        img_palette = img_quantized.convert('RGB')
        
        # Get all pixels
        pixels = list(img_palette.getdata())
        
        # Count occurrences of each color
        color_count = Counter(pixels)
        
        # Get the most common colors
        most_common = color_count.most_common(num_colors)
        colors = [np.array(color) for color, _ in most_common]
        
        return colors
    except Exception as e:
        print(f"Error extracting colors: {e}")
        # Return some default colors as fallback
        return [
            np.array([255, 0, 0]),   # Red
            np.array([0, 255, 0]),   # Green
            np.array([0, 0, 255]),   # Blue
            np.array([255, 255, 0]), # Yellow
            np.array([255, 0, 255])  # Magenta
        ][:num_colors]

def generate_palette_image(colors):
    """
    Generate a simple HTML representation of colors.
    
    Args:
        colors: List of RGB colors
        
    Returns:
        HTML string representation of the color palette
    """
    # Generate HTML color boxes
    html = '<div style="display: flex; width: 100%;">'
    
    for color in colors:
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        text_color = 'white' if sum(color) < 380 else 'black'
        
        html += f'''
        <div style="flex: 1; height: 100px; background-color: {hex_color}; 
                  display: flex; justify-content: center; align-items: center;">
            <span style="color: {text_color}; font-family: monospace;">{hex_color}</span>
        </div>
        '''
    
    html += '</div>'
    return html

def get_color_data(colors):
    """
    Convert colors to a list of dictionaries with RGB and hex values.
    
    Args:
        colors: List of RGB colors
        
    Returns:
        List of color dictionaries
    """
    color_data = []
    for color in colors:
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        color_data.append({
            'rgb': color.tolist(),
            'hex': hex_color
        })
    return color_data