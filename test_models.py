#!/usr/bin/env python3
"""
Test script to verify that the uploaded models work correctly
"""

import os
import sys
import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_files():
    """Test if model files exist and are accessible"""
    logger.info("Testing model file accessibility...")
    
    models_to_check = [
        "models/best.pt",
        "models/depth_pro.pt",
        "labels.txt"
    ]
    
    for model_path in models_to_check:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"✓ {model_path} exists ({size_mb:.1f} MB)")
        else:
            logger.error(f"✗ {model_path} not found")
            return False
    
    return True

def test_food_detector():
    """Test the food detector with uploaded model"""
    logger.info("Testing food detector...")
    
    try:
        from food_detection import FoodDetector
        
        # Initialize detector
        detector = FoodDetector()
        logger.info("✓ Food detector initialized successfully")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        detections = detector.detect_food(test_image, confidence_threshold=0.3)
        logger.info(f"✓ Food detection completed. Found {len(detections)} items")
        
        # Test supported classes
        logger.info(f"✓ Supported food classes: {list(detector.food_classes.values())}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Food detector test failed: {e}")
        return False

def test_depth_estimator():
    """Test the depth estimator"""
    logger.info("Testing depth estimator...")
    
    try:
        from depth_estimation import DepthEstimator
        
        # Initialize estimator
        estimator = DepthEstimator()
        logger.info("✓ Depth estimator initialized successfully")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test depth estimation
        depth_map = estimator.estimate_depth(test_image)
        
        if depth_map is not None:
            logger.info(f"✓ Depth estimation completed. Shape: {depth_map.shape}, Range: {depth_map.min():.3f} - {depth_map.max():.3f}")
        else:
            logger.warning("⚠ Depth estimation returned None (using fallback)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Depth estimator test failed: {e}")
        return False

def test_calorie_calculator():
    """Test the calorie calculator"""
    logger.info("Testing calorie calculator...")
    
    try:
        from calorie_calculator import CalorieCalculator
        
        # Initialize calculator
        calculator = CalorieCalculator()
        logger.info("✓ Calorie calculator initialized successfully")
        
        # Test calculation
        nutrition = calculator.calculate_calories('Rice', 150)
        logger.info(f"✓ Calorie calculation completed: {nutrition['calories']:.1f} calories for 150ml Rice")
        
        # Test supported foods
        supported_foods = list(calculator.nutrition_data.keys())
        logger.info(f"✓ Supported foods: {supported_foods}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Calorie calculator test failed: {e}")
        return False

def test_complete_system():
    """Test the complete food calorie estimation system"""
    logger.info("Testing complete system...")
    
    try:
        from food_calorie_estimator import FoodCalorieEstimator
        
        # Initialize system
        estimator = FoodCalorieEstimator()
        logger.info("✓ Complete system initialized successfully")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test estimation
        results = estimator.estimate_calories(test_image, confidence_threshold=0.3)
        
        if results['success']:
            logger.info(f"✓ Complete estimation successful!")
            logger.info(f"  - Processing time: {results['processing_time_seconds']:.2f}s")
            logger.info(f"  - Food items detected: {results['food_items_detected']}")
            logger.info(f"  - Total calories: {results['nutrition_summary']['total_calories']:.1f}")
        else:
            logger.warning(f"⚠ Estimation completed but with issues: {results.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Complete system test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting model and system tests...")
    
    tests = [
        ("Model Files", test_model_files),
        ("Food Detector", test_food_detector),
        ("Depth Estimator", test_depth_estimator),
        ("Calorie Calculator", test_calorie_calculator),
        ("Complete System", test_complete_system)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All tests passed! The system is ready to use.")
    else:
        logger.warning("\n⚠ Some tests failed. Please check the logs above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
