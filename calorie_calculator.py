"""
Calorie Calculation Module for Food Items
"""

import numpy as np
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalorieCalculator:
    """Calculate calories based on food type, volume, and nutritional density"""
    
    # Nutritional data for food items (calories per 100ml and density g/ml)
    # Based on typical South Asian/Bengali cuisine
    FOOD_NUTRITION_DATA = {
        'Beef curry': {
            'calories_per_100g': 250,
            'density_g_per_ml': 0.9,  # Curry with gravy
            'protein_per_100g': 20,
            'carbs_per_100g': 8,
            'fat_per_100g': 18
        },
        'Biriyani': {
            'calories_per_100g': 200,
            'density_g_per_ml': 0.7,  # Rice-based dish
            'protein_per_100g': 8,
            'carbs_per_100g': 35,
            'fat_per_100g': 6
        },
        'chicken curry': {
            'calories_per_100g': 220,
            'density_g_per_ml': 0.9,
            'protein_per_100g': 25,
            'carbs_per_100g': 6,
            'fat_per_100g': 12
        },
        'Egg': {
            'calories_per_100g': 155,
            'density_g_per_ml': 1.0,  # Whole egg
            'protein_per_100g': 13,
            'carbs_per_100g': 1,
            'fat_per_100g': 11
        },
        'egg curry': {
            'calories_per_100g': 180,
            'density_g_per_ml': 0.95,
            'protein_per_100g': 12,
            'carbs_per_100g': 5,
            'fat_per_100g': 14
        },
        'Eggplants': {
            'calories_per_100g': 80,
            'density_g_per_ml': 0.8,  # Cooked eggplant curry
            'protein_per_100g': 2,
            'carbs_per_100g': 12,
            'fat_per_100g': 4
        },
        'Fish': {
            'calories_per_100g': 200,
            'density_g_per_ml': 0.9,  # Fish curry
            'protein_per_100g': 22,
            'carbs_per_100g': 3,
            'fat_per_100g': 12
        },
        'Khichuri': {
            'calories_per_100g': 120,
            'density_g_per_ml': 0.8,  # Rice and lentil porridge
            'protein_per_100g': 4,
            'carbs_per_100g': 22,
            'fat_per_100g': 2
        },
        'Potato mash': {
            'calories_per_100g': 110,
            'density_g_per_ml': 0.9,  # Mashed potato
            'protein_per_100g': 2,
            'carbs_per_100g': 20,
            'fat_per_100g': 3
        },
        'Rice': {
            'calories_per_100g': 130,
            'density_g_per_ml': 0.75,  # Cooked rice
            'protein_per_100g': 3,
            'carbs_per_100g': 28,
            'fat_per_100g': 0.3
        }
    }
    
    def __init__(self):
        """Initialize the calorie calculator"""
        self.nutrition_data = self.FOOD_NUTRITION_DATA
    
    def calculate_calories(self, food_name: str, volume_ml: float) -> Dict[str, float]:
        """
        Calculate calories and nutritional information for a food item
        
        Args:
            food_name: Name of the food item
            volume_ml: Volume of the food item in milliliters
            
        Returns:
            Dictionary containing calorie and nutritional information
        """
        if food_name not in self.nutrition_data:
            logger.warning(f"Nutrition data not available for: {food_name}")
            return self._get_default_nutrition(volume_ml)
        
        nutrition = self.nutrition_data[food_name]
        
        # Calculate weight from volume and density
        weight_g = volume_ml * nutrition['density_g_per_ml']
        
        # Calculate nutritional values based on weight
        calories = (weight_g / 100) * nutrition['calories_per_100g']
        protein = (weight_g / 100) * nutrition['protein_per_100g']
        carbs = (weight_g / 100) * nutrition['carbs_per_100g']
        fat = (weight_g / 100) * nutrition['fat_per_100g']
        
        result = {
            'food_name': food_name,
            'volume_ml': volume_ml,
            'weight_g': weight_g,
            'calories': calories,
            'protein_g': protein,
            'carbohydrates_g': carbs,
            'fat_g': fat,
            'density_g_per_ml': nutrition['density_g_per_ml']
        }
        
        logger.info(f"Calculated nutrition for {food_name}: {calories:.1f} cal, {weight_g:.1f}g")
        return result
    
    def _get_default_nutrition(self, volume_ml: float) -> Dict[str, float]:
        """
        Get default nutritional values for unknown food items
        
        Args:
            volume_ml: Volume in milliliters
            
        Returns:
            Default nutritional information
        """
        # Use average values for unknown foods
        default_density = 0.8
        default_calories_per_100g = 150
        
        weight_g = volume_ml * default_density
        calories = (weight_g / 100) * default_calories_per_100g
        
        return {
            'food_name': 'Unknown',
            'volume_ml': volume_ml,
            'weight_g': weight_g,
            'calories': calories,
            'protein_g': weight_g * 0.1,  # 10% protein estimate
            'carbohydrates_g': weight_g * 0.2,  # 20% carbs estimate
            'fat_g': weight_g * 0.05,  # 5% fat estimate
            'density_g_per_ml': default_density
        }
    
    def calculate_total_nutrition(self, food_items: List[Dict]) -> Dict[str, float]:
        """
        Calculate total nutritional information for multiple food items
        
        Args:
            food_items: List of food items with volume information
            
        Returns:
            Total nutritional summary
        """
        total_nutrition = {
            'total_calories': 0.0,
            'total_protein_g': 0.0,
            'total_carbohydrates_g': 0.0,
            'total_fat_g': 0.0,
            'total_weight_g': 0.0,
            'total_volume_ml': 0.0,
            'food_count': 0
        }
        
        detailed_items = []
        
        for item in food_items:
            if 'class_name' in item and 'volume_ml' in item:
                nutrition = self.calculate_calories(item['class_name'], item['volume_ml'])
                
                # Add to totals
                total_nutrition['total_calories'] += nutrition['calories']
                total_nutrition['total_protein_g'] += nutrition['protein_g']
                total_nutrition['total_carbohydrates_g'] += nutrition['carbohydrates_g']
                total_nutrition['total_fat_g'] += nutrition['fat_g']
                total_nutrition['total_weight_g'] += nutrition['weight_g']
                total_nutrition['total_volume_ml'] += nutrition['volume_ml']
                total_nutrition['food_count'] += 1
                
                # Add detailed nutrition to item
                item_with_nutrition = item.copy()
                item_with_nutrition.update(nutrition)
                detailed_items.append(item_with_nutrition)
        
        # Calculate average density
        if total_nutrition['total_volume_ml'] > 0:
            avg_density = total_nutrition['total_weight_g'] / total_nutrition['total_volume_ml']
            total_nutrition['average_density_g_per_ml'] = avg_density
        
        result = {
            'summary': total_nutrition,
            'detailed_items': detailed_items
        }
        
        logger.info(f"Total nutrition calculated: {total_nutrition['total_calories']:.1f} calories from {total_nutrition['food_count']} items")
        return result
    
    def get_nutrition_info(self, food_name: str) -> Optional[Dict]:
        """
        Get nutritional information for a specific food item
        
        Args:
            food_name: Name of the food item
            
        Returns:
            Nutritional data or None if not found
        """
        return self.nutrition_data.get(food_name)
    
    def add_custom_food(self, food_name: str, calories_per_100g: float, 
                       density_g_per_ml: float, protein_per_100g: float = 0,
                       carbs_per_100g: float = 0, fat_per_100g: float = 0):
        """
        Add custom food nutritional data
        
        Args:
            food_name: Name of the food item
            calories_per_100g: Calories per 100 grams
            density_g_per_ml: Density in grams per milliliter
            protein_per_100g: Protein per 100 grams
            carbs_per_100g: Carbohydrates per 100 grams
            fat_per_100g: Fat per 100 grams
        """
        self.nutrition_data[food_name] = {
            'calories_per_100g': calories_per_100g,
            'density_g_per_ml': density_g_per_ml,
            'protein_per_100g': protein_per_100g,
            'carbs_per_100g': carbs_per_100g,
            'fat_per_100g': fat_per_100g
        }
        logger.info(f"Added custom nutrition data for: {food_name}")

def test_calorie_calculator():
    """Test function for the calorie calculator"""
    calculator = CalorieCalculator()
    
    # Test individual food calculation
    rice_nutrition = calculator.calculate_calories('Rice', 150)  # 150ml of rice
    print(f"Rice nutrition: {rice_nutrition}")
    
    # Test multiple foods
    food_items = [
        {'class_name': 'Rice', 'volume_ml': 150},
        {'class_name': 'chicken curry', 'volume_ml': 100},
        {'class_name': 'Egg', 'volume_ml': 50}
    ]
    
    total_nutrition = calculator.calculate_total_nutrition(food_items)
    print(f"Total nutrition: {total_nutrition['summary']}")

if __name__ == "__main__":
    test_calorie_calculator()
