"""
Main Food Calorie Estimation System
Integrates YOLOv8n-seg, Depth Pro, and calorie calculation
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time

from food_detection import FoodDetector
from depth_estimation import DepthEstimator
from calorie_calculator import CalorieCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodCalorieEstimator:
    """Complete food calorie estimation system"""
    
    def __init__(self, model_path: str = "models/best.pt",
                 labels_path: str = "labels.txt",
                 depth_model_path: str = "models/depth_pro.pt",
                 pixel_size_mm: float = 0.5):
        """
        Initialize the food calorie estimation system

        Args:
            model_path: Path to YOLOv8 detection/segmentation model
            labels_path: Path to labels.txt file
            depth_model_path: Path to depth estimation model
            pixel_size_mm: Camera calibration - size of each pixel in mm
        """
        self.pixel_size_mm = pixel_size_mm

        # Initialize components
        logger.info("Initializing Food Calorie Estimation System...")

        self.food_detector = FoodDetector(model_path, labels_path)
        self.depth_estimator = DepthEstimator(depth_model_path)
        self.calorie_calculator = CalorieCalculator()

        logger.info("System initialization complete!")
    
    def estimate_calories(self, image: np.ndarray, 
                         confidence_threshold: float = 0.5) -> Dict:
        """
        Complete calorie estimation pipeline
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence_threshold: Minimum confidence for food detection
            
        Returns:
            Complete analysis results including calories, nutrition, and visualizations
        """
        start_time = time.time()
        
        try:
            # Step 1: Detect and segment food items
            logger.info("Step 1: Detecting food items...")
            food_detections = self.food_detector.detect_food(image, confidence_threshold)
            
            if not food_detections:
                logger.warning("No food items detected in the image")
                return self._empty_result()
            
            # Step 2: Estimate depth and calculate volumes
            logger.info("Step 2: Estimating depth and calculating volumes...")
            food_with_volumes = self.depth_estimator.process_food_regions(
                image, food_detections, self.pixel_size_mm
            )
            
            # Step 3: Calculate calories and nutrition
            logger.info("Step 3: Calculating calories and nutrition...")
            nutrition_results = self.calorie_calculator.calculate_total_nutrition(food_with_volumes)
            
            # Step 4: Create visualizations
            logger.info("Step 4: Creating visualizations...")
            visualizations = self._create_visualizations(image, food_with_volumes)
            
            # Compile final results
            processing_time = time.time() - start_time
            
            results = {
                'success': True,
                'processing_time_seconds': float(processing_time),
                'food_items_detected': int(len(food_detections)),
                'nutrition_summary': nutrition_results['summary'],
                'detailed_food_items': nutrition_results['detailed_items'],
                'visualizations': visualizations,
                'metadata': {
                    'image_shape': list(image.shape),  # Convert numpy array to list
                    'pixel_size_mm': float(self.pixel_size_mm),
                    'confidence_threshold': float(confidence_threshold)
                }
            }
            
            logger.info(f"Analysis complete! Found {len(food_detections)} food items, "
                       f"Total calories: {nutrition_results['summary']['total_calories']:.1f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during calorie estimation: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time_seconds': time.time() - start_time
            }
    
    def _empty_result(self) -> Dict:
        """Return empty result when no food is detected"""
        return {
            'success': True,
            'processing_time_seconds': 0.0,
            'food_items_detected': 0,
            'nutrition_summary': {
                'total_calories': 0.0,
                'total_protein_g': 0.0,
                'total_carbohydrates_g': 0.0,
                'total_fat_g': 0.0,
                'total_weight_g': 0.0,
                'total_volume_ml': 0.0,
                'food_count': 0
            },
            'detailed_food_items': [],
            'visualizations': {},
            'metadata': {}
        }
    
    def _create_visualizations(self, image: np.ndarray, 
                             food_detections: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Create visualization images
        
        Args:
            image: Original image
            food_detections: Food detections with volume and nutrition data
            
        Returns:
            Dictionary of visualization images
        """
        visualizations = {}
        
        # Detection visualization
        vis_image = self.food_detector.visualize_detections(image, food_detections)
        
        # Add nutrition information to visualization
        y_offset = 30
        for i, detection in enumerate(food_detections):
            if 'calories' in detection:
                text = f"{detection['class_name']}: {detection['calories']:.0f} cal"
                cv2.putText(vis_image, text, (10, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        visualizations['detection_with_calories'] = vis_image
        
        # Create depth visualization if available
        depth_map = self.depth_estimator.estimate_depth(image)
        if depth_map is not None:
            # Normalize depth map for visualization
            depth_normalized = ((depth_map - depth_map.min()) / 
                              (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            visualizations['depth_map'] = depth_colored
        
        return visualizations
    
    def analyze_single_food(self, image: np.ndarray, food_name: str, 
                           mask: np.ndarray) -> Dict:
        """
        Analyze a single food item with known mask
        
        Args:
            image: Input image
            food_name: Name of the food item
            mask: Binary segmentation mask
            
        Returns:
            Analysis results for single food item
        """
        try:
            # Calculate volume using depth estimation
            depth_map = self.depth_estimator.estimate_depth(image)
            if depth_map is None:
                return {'success': False, 'error': 'Depth estimation failed'}
            
            volume_info = self.depth_estimator.calculate_volume(
                depth_map, mask, self.pixel_size_mm
            )
            
            # Calculate nutrition
            nutrition = self.calorie_calculator.calculate_calories(
                food_name, volume_info['volume_ml']
            )
            
            result = {
                'success': True,
                'food_name': food_name,
                'volume_info': volume_info,
                'nutrition': nutrition
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing single food item: {e}")
            return {'success': False, 'error': str(e)}
    
    def calibrate_pixel_size(self, reference_object_pixels: int, 
                           reference_object_mm: float):
        """
        Calibrate pixel size using a reference object
        
        Args:
            reference_object_pixels: Size of reference object in pixels
            reference_object_mm: Known size of reference object in mm
        """
        self.pixel_size_mm = reference_object_mm / reference_object_pixels
        logger.info(f"Pixel size calibrated to: {self.pixel_size_mm:.4f} mm/pixel")
    
    def get_supported_foods(self) -> List[str]:
        """Get list of supported food items"""
        return list(self.calorie_calculator.nutrition_data.keys())
    
    def add_custom_food_nutrition(self, food_name: str, calories_per_100g: float,
                                 density_g_per_ml: float, protein_per_100g: float = 0,
                                 carbs_per_100g: float = 0, fat_per_100g: float = 0):
        """Add custom food nutrition data"""
        self.calorie_calculator.add_custom_food(
            food_name, calories_per_100g, density_g_per_ml,
            protein_per_100g, carbs_per_100g, fat_per_100g
        )

def test_food_calorie_estimator():
    """Test function for the complete system"""
    estimator = FoodCalorieEstimator()
    
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run estimation
    results = estimator.estimate_calories(test_image)
    
    print(f"Estimation results: {results}")
    print(f"Supported foods: {estimator.get_supported_foods()}")

if __name__ == "__main__":
    test_food_calorie_estimator()
