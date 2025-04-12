from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import time
from werkzeug.utils import secure_filename
from color_extractor import load_image, extract_colors, generate_palette_image, get_color_data

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/extract-colors', methods=['POST'])
def api_extract_colors():
    # Check if image file is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400
    
    file = request.files['image']
    
    # Check if the file is valid
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Get number of colors from request, default to 5
        num_colors = int(request.form.get('num_colors', 5))
        
        # Process the image
        img_array = load_image(file)
        colors = extract_colors(img_array, num_colors)
        
        # Generate palette HTML
        palette_html = generate_palette_image(colors)
        
        # Convert colors to a more API-friendly format
        color_data = get_color_data(colors)
        
        # Return the results
        return jsonify({
            'colors': color_data,
            'palette_html': palette_html
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        # Check if image file is in the request
        if 'image' not in request.files:
            return render_template('index.html', error='No image file in request')
        
        file = request.files['image']
        
        # Check if the file is valid
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if not allowed_file(file.filename):
            return render_template('index.html', error='File type not allowed')
        
        try:
            # Save the file temporarily
            filename = secure_filename(f"{int(time.time())}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Get number of colors from request, default to 5
            num_colors = int(request.form.get('num_colors', 5))
            
            # Process the image
            img_array = load_image(open(file_path, 'rb'))
            colors = extract_colors(img_array, num_colors)
            
            # Generate palette HTML
            palette_html = generate_palette_image(colors)
            
            # Convert colors to a more template-friendly format
            color_data = get_color_data(colors)
            
            # Return the results
            return render_template('result.html', 
                                  colors=color_data,
                                  palette_html=palette_html,
                                  image_src=url_for('static', filename=f'uploads/{filename}'))
        
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)