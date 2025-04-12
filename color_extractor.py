import numpy as np
from PIL import Image
from collections import Counter
import os
import warnings

# Set environment variable to avoid the CPU core detection issue
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# Filter the specific warning about CPU cores
warnings.filterwarnings("ignore", category=UserWarning, 
                      message="Could not find the number of physical cores")

def load_image(image_file):
    """Load an image from a file object and return it as a PIL Image."""
    try:
        img = Image.open(image_file)
        
        # Ensure the image is in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image for processing, but keep it larger than before for better color representation
        img = img.resize((200, 200))
        
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
    Extract dominant colors using K-means clustering.
    More accurate for finding representative colors than quantization.
    
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
            
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Reshape the array to a list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Take a sample of pixels to speed up processing for large images
        pixels = shuffle(pixels, random_state=0)[:10000]
        
        # Perform k-means clustering with explicit single-thread configuration
        # Avoiding parallel processing for PythonAnywhere compatibility
        kmeans = KMeans(
            n_clusters=num_colors, 
            random_state=0, 
            n_init=10,
            n_jobs=1,           # Force single-thread processing
            algorithm='auto',
            max_iter=100        # Reduce iterations for faster processing
        )
        kmeans.fit(pixels)
        
        # Get cluster centers (these are our colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Count occurrences of each label
        labels = kmeans.predict(pixels)
        counts = Counter(labels)
        
        # Sort colors by frequency (most common first)
        sorted_colors = [colors[i] for i in sorted(counts.keys(), key=lambda x: -counts[x])]
        
        return [np.array(color) for color in sorted_colors]
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