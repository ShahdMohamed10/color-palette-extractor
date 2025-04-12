import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import os

def load_image(image_file):
    """Load an image from a file object and return it as a numpy array."""
    img = Image.open(image_file)
    
    # Resize image for processing
    img_for_processing = img.resize((150, 150))
    
    return np.array(img_for_processing)

def extract_colors(img_array, num_colors=5):
    """
    Extract dominant colors from an image using K-means clustering.
    
    Args:
        img_array: NumPy array representation of the image
        num_colors: Number of colors to extract
        
    Returns:
        List of colors in RGB format
    """
    # Reshape the image to be a list of pixels
    pixels = img_array.reshape(-1, 3)
    
    # Perform k-means clustering to find the most dominant colors
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get the colors as RGB values in the range [0, 255]
    colors = kmeans.cluster_centers_.astype(int)
    
    # Sort colors by frequency (number of pixels assigned to each cluster)
    labels = kmeans.labels_
    counts = Counter(labels)
    colors = [colors[i] for i in sorted(counts.keys(), key=lambda x: -counts[x])]
    
    return colors

def generate_palette_image(colors):
    """
    Generate an image of the color palette.
    
    Args:
        colors: List of RGB colors
        
    Returns:
        Base64 encoded string of the palette image
    """
    # Create a figure for the palette
    fig = Figure(figsize=(10, 2))
    canvas = FigureCanvas(fig)
    
    # Plot each color as a rectangle
    for i, color in enumerate(colors):
        ax = fig.add_subplot(1, len(colors), i+1)
        ax.set_axis_off()
        
        # Create a solid color rectangle
        rgb = color / 255.0  # Convert to 0-1 range for matplotlib
        ax.fill([0, 1, 1, 0], [0, 0, 1, 1], color=rgb)
        
        # Add RGB values as text
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        ax.text(0.5, 0.5, hex_color, 
                horizontalalignment='center',
                verticalalignment='center',
                color='white' if sum(color) < 380 else 'black')
    
    fig.tight_layout()
    
    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the image to base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return img_base64

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