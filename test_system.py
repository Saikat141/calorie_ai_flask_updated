"""
Test script for the Food Calorie Estimation System
"""

import cv2
import numpy as np
import os
import logging
from food_calorie_estimator import FoodCalorieEstimator
from food_detection import FoodDetector
from depth_estimation import DepthEstimator
from calorie_calculator import CalorieCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_individual_components():
    """Test each component individually"""
    print("=" * 60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 60)
    
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test Food Detector
    print("\n1. Testing Food Detector...")
    try:
        detector = FoodDetector()
        detections = detector.detect_food(test_image)
        print(f"✓ Food Detector initialized successfully")
        print(f"✓ Detected {len(detections)} food items (expected 0 for random image)")
        
        # Test visualization
        vis_image = detector.visualize_detections(test_image, detections)
        print(f"✓ Visualization created: {vis_image.shape}")
        
    except Exception as e:
        print(f"✗ Food Detector failed: {e}")
    
    # Test Depth Estimator
    print("\n2. Testing Depth Estimator...")
    try:
        depth_estimator = DepthEstimator()
        depth_map = depth_estimator.estimate_depth(test_image)
        
        if depth_map is not None:
            print(f"✓ Depth estimation successful: {depth_map.shape}")
            print(f"✓ Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
            
            # Test volume calculation
            test_mask = np.zeros((480, 640), dtype=np.uint8)
            test_mask[200:300, 250:400] = 1
            
            volume_info = depth_estimator.calculate_volume(depth_map, test_mask)
            print(f"✓ Volume calculation: {volume_info['volume_ml']:.2f} ml")
        else:
            print("✗ Depth estimation returned None")
            
    except Exception as e:
        print(f"✗ Depth Estimator failed: {e}")
    
    # Test Calorie Calculator
    print("\n3. Testing Calorie Calculator...")
    try:
        calculator = CalorieCalculator()
        
        # Test individual food calculation
        nutrition = calculator.calculate_calories('Rice', 150)
        print(f"✓ Rice nutrition (150ml): {nutrition['calories']:.1f} calories")
        
        # Test multiple foods
        food_items = [
            {'class_name': 'Rice', 'volume_ml': 150},
            {'class_name': 'chicken curry', 'volume_ml': 100}
        ]
        
        total_nutrition = calculator.calculate_total_nutrition(food_items)
        print(f"✓ Total nutrition: {total_nutrition['summary']['total_calories']:.1f} calories")
        
        # Test supported foods
        supported_foods = list(calculator.nutrition_data.keys())
        print(f"✓ Supports {len(supported_foods)} food types")
        
    except Exception as e:
        print(f"✗ Calorie Calculator failed: {e}")

def test_integrated_system():
    """Test the complete integrated system"""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATED SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize the complete system
        print("\nInitializing Food Calorie Estimator...")
        estimator = FoodCalorieEstimator()
        print("✓ System initialized successfully")
        
        # Test with a synthetic image
        test_image = create_test_food_image()
        print("✓ Test image created")
        
        # Run complete analysis
        print("\nRunning complete calorie estimation...")
        results = estimator.estimate_calories(test_image, confidence_threshold=0.3)
        
        if results['success']:
            print("✓ Analysis completed successfully")
            print(f"✓ Processing time: {results['processing_time_seconds']:.2f} seconds")
            print(f"✓ Food items detected: {results['food_items_detected']}")
            print(f"✓ Total calories: {results['nutrition_summary']['total_calories']:.1f}")
            
            # Test visualization
            if 'visualizations' in results:
                print(f"✓ Visualizations created: {len(results['visualizations'])} types")
            
        else:
            print(f"✗ Analysis failed: {results.get('error', 'Unknown error')}")
        
        # Test supported foods API
        supported_foods = estimator.get_supported_foods()
        print(f"✓ Supported foods: {len(supported_foods)} items")
        
        # Test custom food addition
        estimator.add_custom_food_nutrition(
            "Test Food", 
            calories_per_100g=200, 
            density_g_per_ml=0.8
        )
        print("✓ Custom food addition successful")
        
    except Exception as e:
        print(f"✗ Integrated system test failed: {e}")
        import traceback
        traceback.print_exc()

def create_test_food_image():
    """Create a synthetic test image with food-like shapes"""
    # Create a more realistic test image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light background
    
    # Add some food-like colored regions
    # Rice-like region (white/beige)
    cv2.rectangle(image, (100, 150), (250, 250), (245, 245, 220), -1)
    
    # Curry-like region (brown/orange)
    cv2.circle(image, (400, 200), 60, (139, 69, 19), -1)
    
    # Add some texture
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def test_file_operations():
    """Test file loading and saving operations"""
    print("\n" + "=" * 60)
    print("TESTING FILE OPERATIONS")
    print("=" * 60)
    
    # Test labels file
    print("\n1. Testing labels file...")
    try:
        if os.path.exists('labels.txt'):
            with open('labels.txt', 'r') as f:
                labels = f.readlines()
            print(f"✓ Labels file loaded: {len(labels)} labels")
            for i, label in enumerate(labels[:5]):  # Show first 5
                print(f"   {i}: {label.strip()}")
        else:
            print("✗ Labels file not found")
    except Exception as e:
        print(f"✗ Labels file error: {e}")
    
    # Test model file
    print("\n2. Testing model file...")
    try:
        model_path = "models/yolov8n-seg.pt"
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"✓ Model file found: {file_size:.1f} MB")
        else:
            print("✗ Model file not found - will download default YOLO model")
    except Exception as e:
        print(f"✗ Model file error: {e}")
    
    # Test directories
    print("\n3. Testing directories...")
    required_dirs = ['models', 'templates', 'static']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory exists: {dir_name}")
        else:
            print(f"✗ Directory missing: {dir_name}")

def run_performance_test():
    """Run basic performance tests"""
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)
    
    try:
        estimator = FoodCalorieEstimator()
        test_image = create_test_food_image()
        
        # Run multiple iterations
        import time
        times = []
        
        print("\nRunning 3 iterations...")
        for i in range(3):
            start_time = time.time()
            results = estimator.estimate_calories(test_image, confidence_threshold=0.3)
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
            print(f"Iteration {i+1}: {processing_time:.2f} seconds")
        
        avg_time = sum(times) / len(times)
        print(f"\nAverage processing time: {avg_time:.2f} seconds")
        
        if avg_time < 10:
            print("✓ Performance: Good (< 10 seconds)")
        elif avg_time < 30:
            print("⚠ Performance: Acceptable (< 30 seconds)")
        else:
            print("✗ Performance: Slow (> 30 seconds)")
            
    except Exception as e:
        print(f"✗ Performance test failed: {e}")

def main():
    """Run all tests"""
    print("FOOD CALORIE ESTIMATION SYSTEM - TEST SUITE")
    print("=" * 60)
    
    # Run all test suites
    test_file_operations()
    test_individual_components()
    test_integrated_system()
    run_performance_test()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)
    print("\nIf all tests passed, you can start the web application with:")
    print("python app.py")
    print("\nThen open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()
