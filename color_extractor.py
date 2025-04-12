import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import os

def load_image(image_file):
    """Load an image from a file object and return it as a numpy array."""
    try:
        img = Image.open(image_file)
        
        # Ensure the image is in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image for processing (smaller for efficiency)
        img_for_processing = img.resize((100, 100))
        
        # Convert to array with explicit dtype
        return np.array(img_for_processing, dtype=np.float64)
    except Exception as e:
        print(f"Error loading image: {e}")
        # Return a small colored image as fallback
        return np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.float64)

def extract_colors(img_array, num_colors=5):
    """
    Extract dominant colors from an image using K-means clustering.
    
    Args:
        img_array: NumPy array representation of the image
        num_colors: Number of colors to extract
        
    Returns:
        List of colors in RGB format
    """
    try:
        # Reshape the image to be a list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Handle images with very few unique colors
        unique_pixels = np.unique(pixels, axis=0)
        if len(unique_pixels) < num_colors:
            # If we have fewer unique colors than requested, just return those
            return unique_pixels.astype(int)
        
        # Perform k-means clustering with explicit parameters for compatibility
        kmeans = KMeans(
            n_clusters=num_colors,
            random_state=42,
            n_init=10 if hasattr(KMeans, 'n_init') else None,  # Handle different sklearn versions
            max_iter=300
        )
        kmeans.fit(pixels)
        
        # Get the colors as RGB values in the range [0, 255]
        colors = kmeans.cluster_centers_.astype(int)
        
        # Ensure colors are in valid range
        colors = np.clip(colors, 0, 255)
        
        # Sort colors by frequency (number of pixels assigned to each cluster)
        labels = kmeans.labels_
        counts = Counter(labels)
        colors = [colors[i] for i in sorted(counts.keys(), key=lambda x: -counts[x])]
        
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