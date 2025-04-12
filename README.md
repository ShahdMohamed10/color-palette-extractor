# Color Palette Extractor

A web application that extracts dominant color palettes from images.

## Features

- Upload any image to extract its dominant colors
- Choose the number of colors to extract (3-8)
- Displays color palette with RGB and HEX values
- API endpoint for integration with other applications

## Technologies Used

- Python
- Flask
- Scikit-learn (K-means clustering)
- Matplotlib
- NumPy
- PIL (Python Imaging Library)

## How to Use

### Web Interface
1. Visit the site
2. Upload an image
3. Select the number of colors to extract
4. Get your color palette!

### API
Send a POST request to `/api/extract-colors` with:
- An image file in the form field "image"
- Optionally specify "num_colors" (default is 5)

## Installation

```
pip install -r requirements.txt
python app.py
```

## License

MIT 