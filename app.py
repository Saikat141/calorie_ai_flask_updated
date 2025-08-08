"""
Flask Food Calorie Estimation API
"""

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image
import logging
import time

from food_calorie_estimator import FoodCalorieEstimator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the food calorie estimator
try:
    estimator = FoodCalorieEstimator()
    logger.info("Food calorie estimator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize estimator: {e}")
    estimator = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    try:
        # Convert BGR to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_array

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

def clean_results_for_json(results):
    """Clean results to make them JSON serializable"""
    if not isinstance(results, dict):
        return results

    cleaned = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            # Convert numpy arrays to lists
            cleaned[key] = value.tolist()
        elif isinstance(value, np.integer):
            # Convert numpy integers to Python int
            cleaned[key] = int(value)
        elif isinstance(value, np.floating):
            # Convert numpy floats to Python float
            cleaned[key] = float(value)
        elif isinstance(value, dict):
            # Recursively clean nested dictionaries
            cleaned[key] = clean_results_for_json(value)
        elif isinstance(value, list):
            # Clean lists that might contain numpy objects
            cleaned[key] = [clean_results_for_json(item) if isinstance(item, dict) else
                           (float(item) if isinstance(item, (np.integer, np.floating)) else item)
                           for item in value]
        else:
            cleaned[key] = value

    return cleaned

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/estimate_calories', methods=['POST'])
def estimate_calories():
    """API endpoint for calorie estimation"""
    if estimator is None:
        return jsonify({
            'success': False,
            'error': 'Calorie estimator not initialized'
        }), 500
    
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'File type not allowed'
            }), 400
        
        # Read and process image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image file'
            }), 400
        
        # Get confidence threshold from request
        confidence = float(request.form.get('confidence', 0.5))
        
        # Run calorie estimation
        results = estimator.estimate_calories(image, confidence)

        # Convert visualization images to base64
        if 'visualizations' in results:
            for key, img_array in results['visualizations'].items():
                if img_array is not None:
                    results['visualizations'][key] = image_to_base64(img_array)

        # Clean results for JSON serialization
        cleaned_results = clean_results_for_json(results)

        return jsonify(cleaned_results)
        
    except Exception as e:
        logger.error(f"Error in calorie estimation: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/api/supported_foods', methods=['GET'])
def get_supported_foods():
    """Get list of supported food items"""
    if estimator is None:
        return jsonify({
            'success': False,
            'error': 'Estimator not initialized'
        }), 500

    try:
        foods = estimator.get_supported_foods()
        return jsonify({
            'success': True,
            'supported_foods': foods
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test', methods=['GET'])
def test_api():
    """Test endpoint to verify API is working"""
    return jsonify({
        'success': True,
        'message': 'API is working!',
        'estimator_status': 'initialized' if estimator is not None else 'not_initialized',
        'timestamp': str(time.time())
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test basic functionality
        test_result = {
            'api_status': 'healthy',
            'estimator_initialized': estimator is not None,
            'supported_foods_count': len(estimator.get_supported_foods()) if estimator else 0
        }

        return jsonify({
            'success': True,
            'health': test_result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# main driver function
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
