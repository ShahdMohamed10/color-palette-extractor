import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from PIL import Image
from collections import Counter
import os
import warnings

# Set environment variable to avoid the CPU core detection issue
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Make scikit-learn optional to avoid dependency issues
try:
    from sklearn.cluster import KMeans
    from sklearn.utils import shuffle
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, falling back to simpler methods")
    SKLEARN_AVAILABLE = False

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
        logger.error(f"Error loading image: {e}")
        # Create a fallback image (2x2 color grid)
        fallback = Image.new('RGB', (2, 2))
        fallback.putpixel((0, 0), (255, 0, 0))  # Red
        fallback.putpixel((1, 0), (0, 255, 0))  # Green
        fallback.putpixel((0, 1), (0, 0, 255))  # Blue
        fallback.putpixel((1, 1), (255, 255, 0))  # Yellow
        return fallback

def extract_colors_quantize(img, num_colors=5):
    """Extract colors using PIL's quantize method (simple and reliable)"""
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Use PIL's quantization - much more reliable
        img_quantized = img.quantize(colors=num_colors * 2, method=2)
        
        # Get the palette
        palette = img_quantized.getpalette()
        color_counts = Counter(img_quantized.getdata())
        
        # Convert palette to RGB colors, ordered by frequency
        colors = []
        for i, count in color_counts.most_common(num_colors):
            r = palette[i*3]
            g = palette[i*3+1]
            b = palette[i*3+2]
            colors.append(np.array([r, g, b]))
        
        return colors
    except Exception as e:
        logger.error(f"Quantize extraction failed: {e}")
        return None

def extract_colors_kmeans(img, num_colors=5):
    """Extract colors using k-means clustering if scikit-learn is available"""
    if not SKLEARN_AVAILABLE:
        return None
        
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
        kmeans = KMeans(
            n_clusters=num_colors, 
            random_state=0, 
            n_init=10,
            n_jobs=1,
            algorithm='auto',
            max_iter=100
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
        logger.error(f"K-means extraction failed: {e}")
        return None

def extract_colors_binning(img, num_colors=5):
    """Extract colors using color binning - simple and reliable approach"""
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Reshape the array to a list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Quantize the colors (divide by 16 and multiply to get values 0, 16, 32, ..., 240)
        # This reduces the 16.7 million colors to a more manageable 4096
        bins = 16
        quantized = (pixels // bins * bins).astype(np.uint8)
        
        # Count frequency of each quantized color
        color_counts = Counter([tuple(color) for color in quantized])
        
        # Get the most common colors
        common_colors = [np.array(color) for color, count in color_counts.most_common(num_colors)]
        
        return common_colors
    except Exception as e:
        logger.error(f"Binning extraction failed: {e}")
        return None

def extract_colors(img, num_colors=5):
    """
    Extract dominant colors using multiple methods and pick the best result.
    
    Args:
        img: PIL Image
        num_colors: Number of colors to extract
        
    Returns:
        List of colors in RGB format
    """
    logger.info("Starting color extraction with multiple methods")
    
    # Try multiple methods in order of visual quality
    methods = [
        ("quantize", extract_colors_quantize),
        ("kmeans", extract_colors_kmeans),
        ("binning", extract_colors_binning)
    ]
    
    for method_name, method_func in methods:
        try:
            colors = method_func(img, num_colors)
            if colors and len(colors) >= num_colors:
                logger.info(f"Successfully extracted colors using {method_name} method")
                # Return exactly the number of colors requested
                return colors[:num_colors]
        except Exception as e:
            logger.warning(f"Method {method_name} failed: {e}")
    
    # If all methods fail, return default colors
    logger.warning("All extraction methods failed, returning default colors")
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