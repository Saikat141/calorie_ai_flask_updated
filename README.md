# Food Calorie Estimation System

A comprehensive AI-powered food calorie estimation system that combines YOLOv8n-seg for food detection and segmentation with Depth Pro for volume estimation to calculate accurate calorie content.

## Features

- **Food Detection & Segmentation**: Uses YOLOv11n-seg to detect and segment food items
- **Depth Estimation**: Integrates Depth Pro for accurate volume calculation
- **Calorie Calculation**: Estimates calories based on food type, volume, and nutritional density
- **Web Interface**: User-friendly web application for image upload and analysis
- **REST API**: RESTful endpoints for programmatic access
- **Visualization**: Shows detection results with calorie information overlay

## Supported Food Items

The system currently supports the following South Asian/Bengali cuisine items:

- Beef curry
- Biriyani
- Chicken curry
- Egg
- Egg curry
- Eggplants
- Fish
- Khichuri
- Potato mash
- Rice

## System Architecture

```
Image Input → YOLOv8n-seg → Food Detection & Segmentation
                ↓
            Depth Pro → Volume Estimation
                ↓
        Calorie Calculator → Nutritional Analysis
                ↓
            Results & Visualization
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd flask_calorie_ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Depth Pro** (if not already installed)
   ```bash
   git clone https://github.com/apple/ml-depth-pro
   cd ml-depth-pro
   pip install -e .
   # Download pretrained models
   source get_pretrained_models.sh
   cd ..
   ```

4. **Ensure YOLOv8n-seg model is available**
   - Place your trained YOLOv8n-seg model in `models/yolov8n-seg.pt`
   - Or the system will download the default COCO-trained model

5. **Create necessary directories**
   ```bash
   mkdir uploads
   ```

## Usage

### Web Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Upload a food image** and click "Analyze Food"

4. **View results** including:
   - Total calories and nutritional breakdown
   - Individual food item analysis
   - Detection visualization

### API Usage

#### Estimate Calories

```bash
curl -X POST -F "image=@your_food_image.jpg" -F "confidence=0.5" \
     http://localhost:5000/api/estimate_calories
```

#### Get Supported Foods

```bash
curl http://localhost:5000/api/supported_foods
```

### Python API

```python
from food_calorie_estimator import FoodCalorieEstimator
import cv2

# Initialize the estimator
estimator = FoodCalorieEstimator()

# Load an image
image = cv2.imread('path/to/your/food_image.jpg')

# Estimate calories
results = estimator.estimate_calories(image, confidence_threshold=0.5)

# Print results
print(f"Total calories: {results['nutrition_summary']['total_calories']:.1f}")
print(f"Food items detected: {results['food_items_detected']}")
```

## Configuration

### Pixel Size Calibration

For accurate volume estimation, calibrate the pixel size:

```python
# If you know the real-world size of an object in the image
estimator.calibrate_pixel_size(
    reference_object_pixels=100,  # Size in pixels
    reference_object_mm=50        # Known size in mm
)
```

### Custom Food Items

Add custom food nutritional data:

```python
estimator.add_custom_food_nutrition(
    food_name="Custom Food",
    calories_per_100g=200,
    density_g_per_ml=0.8,
    protein_per_100g=15,
    carbs_per_100g=25,
    fat_per_100g=8
)
```

## File Structure

```
flask_calorie_ai/
├── app.py                      # Flask web application
├── food_detection.py           # YOLOv11n-seg food detection
├── depth_estimation.py         # Depth Pro integration
├── calorie_calculator.py       # Calorie calculation logic
├── food_calorie_estimator.py   # Main integration module
├── labels.txt                  # Food class labels
├── requirements.txt            # Python dependencies
├── models/
│   └── best.pt                 # YOLOv11n-seg segmentation model
        depth_pro.pt
├── templates/
│   └── index.html             # Web interface template
├── static/
│   └── style.css              # CSS styling
└── uploads/                   # Uploaded images directory
```

## API Response Format

```json
{
  "success": true,
  "processing_time_seconds": 2.34,
  "food_items_detected": 3,
  "nutrition_summary": {
    "total_calories": 450.2,
    "total_protein_g": 25.1,
    "total_carbohydrates_g": 45.3,
    "total_fat_g": 18.7,
    "total_weight_g": 320.5,
    "total_volume_ml": 280.0,
    "food_count": 3
  },
  "detailed_food_items": [
    {
      "class_name": "Rice",
      "confidence": 0.89,
      "calories": 195.0,
      "volume_ml": 150.0,
      "weight_g": 112.5,
      "protein_g": 3.4,
      "carbohydrates_g": 42.0,
      "fat_g": 0.3
    }
  ],
  "visualizations": {
    "detection_with_calories": "data:image/png;base64,..."
  }
}
```

## Troubleshooting

### Common Issues

1. **Model loading errors**
   - Ensure YOLOv11n-seg model is in the correct path
   - Check if the model is compatible with your ultralytics version

2. **Depth Pro installation issues**
   - Make sure you have the correct PyTorch version
   - Check CUDA compatibility

3. **Memory errors**
   - Reduce image size before processing
   - Use CPU instead of GPU if memory is limited

4. **No food detected**
   - Lower the confidence threshold
   - Ensure good image quality and lighting
   - Check if the food items are in the supported list

### Performance Optimization

- Use GPU acceleration for faster processing
- Resize large images before analysis
- Adjust confidence threshold based on your needs
- Consider batch processing for multiple images

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments


- [Apple Depth Pro](https://github.com/apple/ml-depth-pro) for depth estimation
- Flask for the web framework
- Bootstrap for UI components
