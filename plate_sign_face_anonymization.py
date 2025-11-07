import os
import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Tuple


class LabelMePrivacyProcessor:
    def __init__(self, input_folder: str, output_folder: str = "last"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        
        # Create output folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Supported privacy types
        self.privacy_types = ['face', 'plate', 'sign']
    
    
    def apply_mosaic(self, image, x1, y1, x2, y2, mosaic_size=15):
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return image
        
        # Extract the region to be mosaicked
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return image
        
        # Calculate mosaic dimensions
        roi_h, roi_w = roi.shape[:2]
        mosaic_h = max(1, roi_h // mosaic_size)
        mosaic_w = max(1, roi_w // mosaic_size)
        
        # Shrink then enlarge to create mosaic effect
        small = cv2.resize(roi, (mosaic_w, mosaic_h), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        
        # Replace the mosaic region back to original image
        image[y1:y2, x1:x2] = mosaic
        
        return image
    
    def apply_blur(self, image, x1, y1, x2, y2, blur_strength=51):
        # Ensure blur strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return image
        
        # Extract the region to be blurred
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return image
        
        # Apply Gaussian blur
        blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
        
        # Replace the blurred region back to original image
        image[y1:y2, x1:x2] = blurred_roi
        
        return image
    
    def apply_black_box(self, image, x1, y1, x2, y2):
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return image
        
        # Fill region with black color
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return image
    
    def parse_labelme_json(self, json_path: Path) -> List[Dict]:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            annotations = []
            
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower()
                shape_type = shape.get('shape_type', '')
                points = shape.get('points', [])
                
                # Only process bounding box type annotations
                if shape_type == 'rectangle' and label in self.privacy_types:
                    # Extract bounding box coordinates
                    if len(points) >= 2:
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        
                        x1 = min(x_coords)
                        y1 = min(y_coords)
                        x2 = max(x_coords)
                        y2 = max(y_coords)
                        
                        annotations.append({
                            'label': label,
                            'bbox': (x1, y1, x2, y2)
                        })
            
            return annotations
        
        except Exception as e:
            print(f"Failed to parse JSON file {json_path}: {str(e)}")
            return []
    
    def process_single_image(self, image_path: Path, json_path: Path, 
                           privacy_method: str = 'mosaic', 
                           mosaic_size: int = 15, 
                           blur_strength: int = 51) -> Dict:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Unable to read image: {image_path}")
            return {'success': False, 'error': 'Unable to read image'}
        
        # Parse annotation file
        annotations = self.parse_labelme_json(json_path)
        
        if not annotations:
            print(f"{image_path.name}: No valid annotations found")
            return {'success': False, 'error': 'No valid annotations'}
        
        # Statistics information
        stats = {'face': 0, 'plate': 0, 'sign': 0}
        
        # Process each annotated region
        for ann in annotations:
            label = ann['label']
            x1, y1, x2, y2 = ann['bbox']
            
            # Apply privacy protection
            if privacy_method == 'mosaic':
                image = self.apply_mosaic(image, x1, y1, x2, y2, mosaic_size)
            elif privacy_method == 'blur':
                image = self.apply_blur(image, x1, y1, x2, y2, blur_strength)
            elif privacy_method == 'black_box':
                image = self.apply_black_box(image, x1, y1, x2, y2)
            
            stats[label] += 1
        
        # Save processed image
        output_path = self.output_folder / image_path.name
        success = cv2.imwrite(str(output_path), image)
        
        if not success:
            print(f"Failed to save image: {output_path}")
            return {'success': False, 'error': 'Failed to save'}
        
        return {
            'success': True,
            'stats': stats,
            'total': sum(stats.values())
        }
    
    def copy_original_image(self, image_path: Path) -> bool:
        try:
            output_path = self.output_folder / image_path.name
            shutil.copy2(str(image_path), str(output_path))
            return True
        except Exception as e:
            print(f"Failed to copy image {image_path.name}: {str(e)}")
            return False
    
    def process_folder(self, privacy_method: str = 'mosaic', 
                      mosaic_size: int = 15, 
                      blur_strength: int = 51):
        image_files = list(self.input_folder.glob("*.jpg")) + \
                     list(self.input_folder.glob("*.JPG"))
        
        if not image_files:
            print("❌ No JPG image files found")
            return
        
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Found {len(image_files)} image files")
        print(f"Privacy protection method: {privacy_method}")
        print("-" * 60)
        
        total_stats = {'face': 0, 'plate': 0, 'sign': 0}
        processed_count = 0
        copied_count = 0
        failed_count = 0
        
        for image_path in image_files:
            json_path = image_path.with_suffix('.json')
            
            if not json_path.exists():
                print(f"{image_path.name}: No JSON annotation file found, copying original image")
                if self.copy_original_image(image_path):
                    copied_count += 1
                    print(f" Copied original image")
                else:
                    failed_count += 1
                continue
            
            print(f"Processing: {image_path.name}")
            
            result = self.process_single_image(
                image_path, json_path, 
                privacy_method, mosaic_size, blur_strength
            )
            
            if result['success']:
                stats = result['stats']
                for key in total_stats:
                    total_stats[key] += stats[key]
                
                print(f"   ✅ Done - Face: {stats['face']}, Plate: {stats['plate']}, Sign: {stats['sign']}")
                processed_count += 1
            else:
                failed_count += 1
        
        print("-" * 60)
        print(f" Processing completed:")
        print(f"    Processed: {processed_count} files")
        print(f"    Copied: {copied_count} files")
        print(f"    Failed: {failed_count} files")
        print(f"    Total privacy protections:")
        print(f"      - Face: {total_stats['face']}")
        print(f"      - Plate: {total_stats['plate']}")
        print(f"      - Sign: {total_stats['sign']}")
        print(f"      - Total: {sum(total_stats.values())}")

def main():
    INPUT_FOLDER = r"F:\train_an\ed1"  # Input folder path
    OUTPUT_FOLDER = r"F:\train_an\last"  # Output folder path
    PRIVACY_METHOD = "mosaic"  # Options: 'mosaic', 'blur', 'black_box'
    MOSAIC_SIZE = 15  # Mosaic block size
    BLUR_STRENGTH = 51  # Blur strength (must be odd number)

    processor = LabelMePrivacyProcessor(INPUT_FOLDER, OUTPUT_FOLDER)
    
    processor.process_folder(
        privacy_method=PRIVACY_METHOD,
        mosaic_size=MOSAIC_SIZE,
        blur_strength=BLUR_STRENGTH
    )
    
if __name__ == "__main__":
    main()