"""
Food Detection and Segmentation Module using YOLOv8n-seg
"""

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodDetector:
    """Food detection and segmentation using YOLOv8n-seg model"""

    def __init__(self, model_path: str = "models/best.pt", labels_path: str = "labels.txt"):
        """
        Initialize the food detector

        Args:
            model_path: Path to YOLOv8n-seg model file
            labels_path: Path to labels.txt file containing food class names
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.food_classes = self.load_labels()
        self.load_model()

    def load_labels(self) -> Dict[int, str]:
        """
        Load food class labels from labels.txt file

        Returns:
            Dictionary mapping class indices to class names
        """
        try:
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]

            # Create mapping from index to class name (0-based indexing)
            food_classes = {i: label for i, label in enumerate(labels)}
            logger.info(f"Loaded {len(food_classes)} food classes from {self.labels_path}")
            return food_classes

        except FileNotFoundError:
            logger.error(f"Labels file not found: {self.labels_path}")
            # Fallback to default classes if labels.txt not found
            return {
                0: 'Beef curry', 1: 'Biriyani', 2: 'chicken curry', 3: 'Egg', 4: 'egg curry',
                5: 'Eggplants', 6: 'Fish', 7: 'Khichuri', 8: 'Potato mash', 9: 'Rice'
            }
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return {}

    def load_model(self):
        """Load the YOLOv8n-seg model"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect_food(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect and segment food items in the image
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected food items with segmentation masks
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Run inference
            results = self.model(image, conf=confidence_threshold)
            
            detected_foods = []
            
            for result in results:
                if result.masks is not None and result.boxes is not None:
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.data.cpu().numpy()
                    
                    for box, mask in zip(boxes, masks):
                        class_id = int(box[5])
                        confidence = float(box[4])
                        
                        # Check if detected class is a food item
                        if class_id in self.food_classes:
                            x1, y1, x2, y2 = map(int, box[:4])
                            
                            # Resize mask to match image dimensions
                            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                            mask_binary = (mask_resized > 0.5).astype(np.uint8)
                            
                            food_item = {
                                'class_id': int(class_id),
                                'class_name': self.food_classes[class_id],
                                'confidence': float(confidence),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'mask': mask_binary,  # Keep mask for internal processing
                                'area_pixels': int(np.sum(mask_binary))
                            }
                            
                            detected_foods.append(food_item)
            
            logger.info(f"Detected {len(detected_foods)} food items")
            return detected_foods
            
        except Exception as e:
            logger.error(f"Error during food detection: {e}")
            return []
    
    def extract_food_regions(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Extract individual food regions from the image using segmentation masks
        
        Args:
            image: Original image
            detections: List of food detections with masks
            
        Returns:
            List of food regions with extracted image data
        """
        food_regions = []
        
        for detection in detections:
            mask = detection['mask']
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract region using bounding box
            region = image[y1:y2, x1:x2]
            region_mask = mask[y1:y2, x1:x2]
            
            # Apply mask to region (set background to black)
            masked_region = region.copy()
            masked_region[region_mask == 0] = 0
            
            food_region = {
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'region_image': masked_region,
                'mask': region_mask,
                'bbox': bbox,
                'area_pixels': detection['area_pixels']
            }
            
            food_regions.append(food_region)
        
        return food_regions
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize food detections on the image
        
        Args:
            image: Original image
            detections: List of food detections
            
        Returns:
            Image with visualized detections
        """
        vis_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            mask = detection['mask']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw mask overlay
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask == 1] = [0, 255, 255]  # Yellow overlay
            vis_image = cv2.addWeighted(vis_image, 0.8, colored_mask, 0.2, 0)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image

def test_food_detector():
    """Test function for the food detector"""
    detector = FoodDetector()
    
    # Create a dummy image for testing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect food items
    detections = detector.detect_food(test_image)
    
    print(f"Number of food items detected: {len(detections)}")
    for detection in detections:
        print(f"- {detection['class_name']}: {detection['confidence']:.2f}")

if __name__ == "__main__":
    test_food_detector()
