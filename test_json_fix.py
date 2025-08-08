"""
Test script to verify JSON serialization fix
"""

import json
import numpy as np
import cv2
from food_calorie_estimator import FoodCalorieEstimator

def test_json_serialization():
    """Test that the system returns JSON-serializable results"""
    print("Testing JSON serialization fix...")
    
    try:
        # Initialize estimator
        estimator = FoodCalorieEstimator()
        print("✓ Estimator initialized")
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("✓ Test image created")
        
        # Run estimation
        results = estimator.estimate_calories(test_image, confidence_threshold=0.3)
        print("✓ Estimation completed")
        
        # Test JSON serialization
        try:
            json_str = json.dumps(results, indent=2)
            print("✓ JSON serialization successful!")
            
            # Parse back to verify
            parsed_results = json.loads(json_str)
            print("✓ JSON parsing successful!")
            
            print(f"Results summary:")
            print(f"  - Success: {parsed_results.get('success', False)}")
            print(f"  - Food items: {parsed_results.get('food_items_detected', 0)}")
            print(f"  - Total calories: {parsed_results.get('nutrition_summary', {}).get('total_calories', 0)}")
            
            return True
            
        except TypeError as e:
            print(f"✗ JSON serialization failed: {e}")
            print("Checking for non-serializable objects...")
            
            def find_non_serializable(obj, path=""):
                """Recursively find non-serializable objects"""
                try:
                    json.dumps(obj)
                except TypeError:
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            find_non_serializable(value, f"{path}.{key}")
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            find_non_serializable(item, f"{path}[{i}]")
                    else:
                        print(f"Non-serializable object at {path}: {type(obj)} - {obj}")
            
            find_non_serializable(results)
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_json_cleaning():
    """Test the JSON cleaning function from app.py"""
    print("\nTesting JSON cleaning function...")
    
    # Import the cleaning function
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    try:
        from app import clean_results_for_json
        
        # Test data with numpy objects
        test_data = {
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_array': np.array([1, 2, 3]),
            'normal_int': 10,
            'normal_float': 2.5,
            'normal_list': [1, 2, 3],
            'nested_dict': {
                'numpy_value': np.int32(100),
                'normal_value': 'test'
            },
            'list_with_numpy': [np.float32(1.1), np.int16(2), 'normal']
        }
        
        print("✓ Test data created with numpy objects")
        
        # Clean the data
        cleaned_data = clean_results_for_json(test_data)
        print("✓ Data cleaning completed")
        
        # Test JSON serialization
        json_str = json.dumps(cleaned_data, indent=2)
        print("✓ Cleaned data is JSON serializable!")
        
        # Verify types
        parsed = json.loads(json_str)
        print("Type conversions:")
        print(f"  numpy_int: {type(parsed['numpy_int'])} = {parsed['numpy_int']}")
        print(f"  numpy_float: {type(parsed['numpy_float'])} = {parsed['numpy_float']}")
        print(f"  numpy_array: {type(parsed['numpy_array'])} = {parsed['numpy_array']}")
        print(f"  nested numpy: {type(parsed['nested_dict']['numpy_value'])} = {parsed['nested_dict']['numpy_value']}")
        
        return True
        
    except Exception as e:
        print(f"✗ JSON cleaning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("JSON SERIALIZATION FIX TEST")
    print("=" * 60)
    
    # Test the cleaning function
    cleaning_success = test_app_json_cleaning()
    
    # Test the full system
    system_success = test_json_serialization()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"JSON Cleaning Function: {'✓ PASS' if cleaning_success else '✗ FAIL'}")
    print(f"Full System Test: {'✓ PASS' if system_success else '✗ FAIL'}")
    
    if cleaning_success and system_success:
        print("\n🎉 All tests passed! The JSON serialization issue should be fixed.")
        print("You can now run the Flask app without the 'Object of type ndarray is not JSON serializable' error.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
