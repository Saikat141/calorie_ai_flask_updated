"""
Depth Estimation Module using Depth Pro for volume calculation
"""

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthEstimator:
    """Depth estimation using Depth Pro model for food volume calculation"""
    
    def __init__(self, model_path: str = "models/depth_pro.pt"):
        """Initialize the depth estimator"""
        self.model = None
        self.transform = None
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_depth_pro()
    
    def load_depth_pro(self):
        """Load Depth Pro model and transforms"""
        try:
            # Check if local model file exists
            import os
            if not os.path.exists(self.model_path):
                logger.warning(f"⚠ Local depth model not found at {self.model_path}. Using fallback depth estimation.")
                self.model = None
                self.transform = None
                return

            # Try to load the model using PyTorch directly if depth_pro package is not available
            try:
                import depth_pro
                logger.info(f"Depth Pro package found, loading local model from {self.model_path}...")

                # Load model from local file
                self.model, self.transform = depth_pro.create_model_and_transforms(
                    device=self.device,
                    precision=torch.float32
                )

                # Load the state dict from the local file
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.eval()
                self.model.to(self.device)
                logger.info(f"✓ Local Depth Pro model loaded successfully on {self.device}")

            except ImportError as e:
                logger.warning("⚠ Depth Pro package not available. Trying to load model directly with PyTorch...")
                try:
                    # Try to load as a standard PyTorch model
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    logger.info(f"✓ Model checkpoint loaded from {self.model_path}")
                    logger.info("⚠ Using fallback depth estimation since depth_pro package is not available.")
                    self.model = None
                    self.transform = None
                except Exception as torch_error:
                    logger.error(f"⚠ Failed to load model with PyTorch: {torch_error}")
                    logger.info("Using improved fallback depth estimation.")
                    self.model = None
                    self.transform = None

            except Exception as model_error:
                logger.error(f"⚠ Local Depth Pro model loading failed: {model_error}")
                logger.info("Using fallback depth estimation instead.")
                self.model = None
                self.transform = None
        except Exception as e:
            logger.error(f"Error during Depth Pro initialization: {e}")
            self.model = None
            self.transform = None
    
    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map for the input image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Depth map as numpy array or None if estimation fails
        """
        if self.model is None or self.transform is None:
            return self._fallback_depth_estimation(image)
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Apply transforms
            image_tensor = self.transform(pil_image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Estimate depth
            with torch.no_grad():
                prediction = self.model.infer(image_tensor)
                depth_map = prediction["depth"].cpu().numpy().squeeze()
            
            logger.info(f"Depth estimation completed. Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
            return depth_map
            
        except Exception as e:
            logger.error(f"Error during depth estimation: {e}")
            return self._fallback_depth_estimation(image)
    
    def _fallback_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        """
        Improved fallback depth estimation using multiple cues

        Args:
            image: Input image as numpy array

        Returns:
            Estimated depth map
        """
        logger.info("Using improved fallback depth estimation")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Brightness-based depth (brighter = closer for food images)
        brightness_depth = 1.0 - (gray.astype(np.float32) / 255.0)

        # 2. Edge-based depth (sharper edges = closer)
        edges = cv2.Canny(gray, 50, 150)
        edge_depth = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0) / 255.0

        # 3. Gradient-based depth
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_depth = gradient_magnitude / np.max(gradient_magnitude)

        # 4. Position-based depth (center is typically closer for food photos)
        y_coords, x_coords = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        position_depth = 1.0 - (distance_from_center / max_distance)

        # 5. Color saturation depth (more saturated = closer for food)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0

        # Combine all depth cues with weights
        depth_map = (
            0.3 * brightness_depth +
            0.2 * edge_depth +
            0.2 * gradient_depth +
            0.15 * position_depth +
            0.15 * saturation
        )

        # Smooth the depth map
        depth_map = cv2.GaussianBlur(depth_map, (7, 7), 0)

        # Scale to reasonable depth range (0.5 to 2.5 meters)
        depth_map = depth_map * 2.0 + 0.5

        return depth_map
    
    def calculate_volume(self, depth_map: np.ndarray, mask: np.ndarray, 
                        pixel_size_mm: float = 0.5) -> Dict[str, float]:
        """
        Calculate volume of food item using depth map and segmentation mask
        
        Args:
            depth_map: Depth map from depth estimation
            mask: Binary segmentation mask for the food item
            pixel_size_mm: Size of each pixel in millimeters (camera calibration dependent)
            
        Returns:
            Dictionary containing volume calculations
        """
        try:
            # Ensure mask is binary
            mask_binary = (mask > 0).astype(np.uint8)
            
            # Resize depth map to match mask dimensions if needed
            if depth_map.shape != mask.shape:
                depth_map = cv2.resize(depth_map, (mask.shape[1], mask.shape[0]))
            
            # Extract depth values within the mask
            masked_depth = depth_map * mask_binary
            valid_depths = masked_depth[mask_binary > 0]
            
            if len(valid_depths) == 0:
                return {'volume_ml': 0.0, 'area_mm2': 0.0, 'avg_height_mm': 0.0}
            
            # Calculate statistics
            area_pixels = np.sum(mask_binary)
            area_mm2 = area_pixels * (pixel_size_mm ** 2)
            
            # Estimate height from depth variation
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            avg_height_mm = (max_depth - min_depth) * 1000  # Convert to mm
            
            # Simple volume estimation (assuming roughly cylindrical/ellipsoidal shape)
            # Volume = base_area * average_height * shape_factor
            shape_factor = 0.6  # Empirical factor for food items (between cylinder=1.0 and hemisphere=0.67)
            volume_mm3 = area_mm2 * avg_height_mm * shape_factor
            volume_ml = volume_mm3 / 1000.0  # Convert mm³ to ml
            
            result = {
                'volume_ml': float(volume_ml),
                'area_mm2': float(area_mm2),
                'avg_height_mm': float(avg_height_mm),
                'area_pixels': int(area_pixels),
                'depth_range': (float(min_depth), float(max_depth))
            }
            
            logger.info(f"Volume calculation: {volume_ml:.2f} ml, Area: {area_mm2:.2f} mm²")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating volume: {e}")
            return {'volume_ml': 0.0, 'area_mm2': 0.0, 'avg_height_mm': 0.0}
    
    def process_food_regions(self, image: np.ndarray, food_detections: List[Dict], 
                           pixel_size_mm: float = 0.5) -> List[Dict]:
        """
        Process multiple food regions to estimate volumes
        
        Args:
            image: Original image
            food_detections: List of food detections with masks
            pixel_size_mm: Pixel size calibration
            
        Returns:
            List of food detections with volume information added
        """
        # Estimate depth for the entire image
        depth_map = self.estimate_depth(image)
        
        if depth_map is None:
            logger.error("Failed to estimate depth")
            return food_detections
        
        # Process each food detection
        enhanced_detections = []
        for detection in food_detections:
            enhanced_detection = detection.copy()
            
            # Calculate volume for this food item
            volume_info = self.calculate_volume(
                depth_map, 
                detection['mask'], 
                pixel_size_mm
            )
            
            # Add volume information to detection
            enhanced_detection.update(volume_info)
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections

def test_depth_estimator():
    """Test function for the depth estimator"""
    estimator = DepthEstimator()
    
    # Create a dummy image for testing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create a dummy mask
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    test_mask[200:300, 250:400] = 1  # Rectangular region
    
    # Estimate depth
    depth_map = estimator.estimate_depth(test_image)
    
    if depth_map is not None:
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
        
        # Calculate volume
        volume_info = estimator.calculate_volume(depth_map, test_mask)
        print(f"Volume estimation: {volume_info}")
    else:
        print("Depth estimation failed")

if __name__ == "__main__":
    test_depth_estimator()
