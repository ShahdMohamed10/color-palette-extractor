import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from PIL import Image
from collections import Counter
import os
import warnings
import colorsys

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

# Try to import scikit-image for LAB color space conversion
try:
    from skimage import color
    SKIMAGE_AVAILABLE = True
except ImportError:
    logger.warning("scikit-image not available, falling back to RGB color space")
    SKIMAGE_AVAILABLE = False

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

def extract_colors_lab(img, num_colors=5):
    """Extract colors using k-means in LAB color space for perceptually better results"""
    if not SKLEARN_AVAILABLE or not SKIMAGE_AVAILABLE:
        return None
        
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert PIL image to numpy array in RGB
        img_array = np.array(img)
        
        # Reshape for conversion
        pixels_rgb = img_array.reshape(-1, 3)
        
        # Convert RGB to LAB color space
        pixels_lab = color.rgb2lab(pixels_rgb / 255.0)
        
        # Sample pixels for faster processing
        pixels_lab = shuffle(pixels_lab, random_state=0)[:10000]
        
        # Perform k-means in LAB space
        kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10, n_jobs=1)
        kmeans.fit(pixels_lab)
        
        # Get cluster centers
        centers_lab = kmeans.cluster_centers_
        
        # Convert centers back to RGB
        centers_rgb = color.lab2rgb(centers_lab) * 255
        centers_rgb = centers_rgb.astype(int)
        
        # Return colors as numpy arrays
        return [np.array(color) for color in centers_rgb]
    except Exception as e:
        logger.error(f"LAB color space extraction failed: {e}")
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

def extract_colors_fg_bg(img, num_colors=5):
    """Extract colors with foreground/background separation"""
    if not SKLEARN_AVAILABLE:
        return None
        
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert PIL image to numpy array
        img_array = np.array(img)
        width, height, _ = img_array.shape
        
        # Create weight map - edge pixels are likely background, center foreground
        y, x = np.mgrid[0:height, 0:width]
        center_y, center_x = height // 2, width // 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 1 at center, 0 at edges
        weight = 1 - (distance / max_distance)
        
        # Flatten arrays
        pixels = img_array.reshape(-1, 3)
        weights = weight.flatten()
        
        # Get "foreground" colors (closer to center)
        fg_threshold = 0.6
        fg_indices = np.where(weights > fg_threshold)[0]
        fg_pixels = pixels[fg_indices]
        
        # Get "background" colors (closer to edges)
        bg_threshold = 0.3
        bg_indices = np.where(weights < bg_threshold)[0]
        bg_pixels = pixels[bg_indices]
        
        # Get foreground colors (more of them)
        fg_count = max(1, int(num_colors * 0.7))  # At least 1, up to 70% of colors
        fg_kmeans = KMeans(n_clusters=fg_count, random_state=0, n_init=10, n_jobs=1)
        fg_kmeans.fit(shuffle(fg_pixels, random_state=0)[:5000])
        fg_colors = fg_kmeans.cluster_centers_.astype(int)
        
        # Get background colors
        bg_count = num_colors - fg_count
        if bg_count > 0 and len(bg_pixels) > 0:
            bg_kmeans = KMeans(n_clusters=bg_count, random_state=0, n_init=10, n_jobs=1)
            bg_kmeans.fit(shuffle(bg_pixels, random_state=0)[:5000])
            bg_colors = bg_kmeans.cluster_centers_.astype(int)
            
            # Combine foreground and background colors
            all_colors = np.vstack([fg_colors, bg_colors])
        else:
            all_colors = fg_colors
            
        return [np.array(color) for color in all_colors]
    except Exception as e:
        logger.error(f"Foreground/background separation failed: {e}")
        return None

def enhance_color_variety(colors, min_distance=30):
    """Ensure colors in the palette have sufficient variety"""
    try:
        result = [colors[0]]  # Start with first color
        
        # For each color, keep it only if it's different enough from existing colors
        for color in colors[1:]:
            # Calculate Euclidean distance to all existing colors
            distances = [np.sqrt(np.sum((color - existing)**2)) for existing in result]
            
            # If it's far enough from all existing colors, add it
            if all(d > min_distance for d in distances):
                result.append(color)
        
        # If we filtered too many, add back most different ones
        while len(result) < len(colors):
            # Find the color not in result that's most different from existing colors
            best_color = None
            best_min_distance = -1
            
            for color in colors:
                # Skip if color is already in result
                if any(np.array_equal(color, c) for c in result):
                    continue
                    
                # Find minimum distance to any color in result
                min_dist = min(np.sqrt(np.sum((color - c)**2)) for c in result)
                
                # Update if this is the most different color found so far
                if min_dist > best_min_distance:
                    best_min_distance = min_dist
                    best_color = color
            
            # Add the most different color if found
            if best_color is not None:
                result.append(best_color)
            else:
                break  # No more colors to add
        
        return result
    except Exception as e:
        logger.error(f"Color variety enhancement failed: {e}")
        return colors  # Return original colors if enhancement fails

def harmonize_colors(colors):
    """Adjust colors to create a more harmonious palette"""
    try:
        harmonized = []
        
        # Convert RGB to HSV for easier manipulation
        hsv_colors = []
        for color in colors:
            r, g, b = [c/255.0 for c in color]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsv_colors.append((h, s, v))
        
        # Slightly boost saturation for more vibrant colors
        for h, s, v in hsv_colors:
            new_s = min(1.0, s * 1.2)  # Boost saturation by 20%
            
            # Ensure decent saturation minimum
            new_s = max(0.2, new_s)
            
            # Convert back to RGB
            r, g, b = colorsys.hsv_to_rgb(h, new_s, v)
            
            # Convert to 0-255 range
            harmonized.append(np.array([int(r*255), int(g*255), int(b*255)]))
        
        # Make sure colors are in valid range
        harmonized = [np.clip(color, 0, 255) for color in harmonized]
        
        return harmonized
    except Exception as e:
        logger.error(f"Color harmonization failed: {e}")
        return colors  # Return original colors if harmonization fails

def extract_colors(img, num_colors=5):
    """
    Extract dominant colors using multiple methods and pick the best result.
    Applies post-processing to improve visual quality.
    
    Args:
        img: PIL Image
        num_colors: Number of colors to extract
        
    Returns:
        List of colors in RGB format
    """
    logger.info(f"Starting color extraction with {num_colors} colors")
    
    # Try multiple methods in order of visual quality
    methods = [
        ("lab", extract_colors_lab),
        ("foreground_background", extract_colors_fg_bg),
        ("quantize", extract_colors_quantize),
        ("kmeans", extract_colors_kmeans),
        ("binning", extract_colors_binning)
    ]
    
    for method_name, method_func in methods:
        try:
            colors = method_func(img, num_colors)
            if colors and len(colors) >= num_colors:
                logger.info(f"Successfully extracted colors using {method_name} method")
                
                # Post-process colors for better results
                colors = enhance_color_variety(colors)
                colors = harmonize_colors(colors)
                
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