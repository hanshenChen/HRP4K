import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
from tqdm import tqdm

class LicensePlatePrivacyMasker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        print(f"Model loaded: {model_path}")
    
    def apply_mosaic(self, image, x1, y1, x2, y2, mosaic_size=15):
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return image
        
        # Extract the region to apply mosaic
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return image
        
        # Calculate mosaic dimensions
        roi_h, roi_w = roi.shape[:2]
        mosaic_h = max(1, roi_h // mosaic_size)
        mosaic_w = max(1, roi_w // mosaic_size)
        
        # Shrink and then expand to create mosaic effect
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
        
        # Extract the region to blur
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
        
        # Fill the region with black
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return image
    
    def process_single_image(self, image_path, output_path=None, privacy_method='mosaic', 
                           confidence_threshold=0.25, mosaic_size=15, blur_strength=51):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Unable to read image: {image_path}")
            return 0
        
        # Perform prediction
        results = self.model.predict(source=image_path, conf=confidence_threshold, verbose=False)
        
        license_plate_count = 0
        
        # Process detection results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
                confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    if conf >= confidence_threshold:
                        x1, y1, x2, y2 = box
                        license_plate_count += 1
                        
                        # Apply privacy protection based on selected method
                        if privacy_method == 'mosaic':
                            image = self.apply_mosaic(image, x1, y1, x2, y2, mosaic_size)
                        elif privacy_method == 'blur':
                            image = self.apply_blur(image, x1, y1, x2, y2, blur_strength)
                        elif privacy_method == 'black_box':
                            image = self.apply_black_box(image, x1, y1, x2, y2)
                        
                        print(f"Object {i+1}: Coordinates({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), Confidence: {conf:.3f}")
        
        # Save processed image
        if output_path is None:
            output_path = image_path
        
        success = cv2.imwrite(output_path, image)
        if not success:
            print(f"Failed to save image: {output_path}")
            return 0
        
        return license_plate_count
    
    def process_folder(self, input_folder, output_folder=None, privacy_method='mosaic', 
                      confidence_threshold=0.25, mosaic_size=15, blur_strength=51):
        input_path = Path(input_folder)
        
        if not input_path.exists():
            print(f"Input folder does not exist: {input_folder}")
            return
        
        jpg_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.JPG")) + \
                   list(input_path.glob("*.jpeg")) + list(input_path.glob("*.JPEG"))
        
        if not jpg_files:
            print(f"No JPG image files found in folder: {input_folder}")
            return
        
        print(f"Found {len(jpg_files)} JPG image files")
        
        if output_folder is not None:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Output folder: {output_folder}")
        else:
            print("Will overwrite original image files")
        
        total_license_plates = 0
        processed_images = 0
        
        for jpg_file in tqdm(jpg_files, desc="Processing images", unit="images"):
            try:
                if output_folder is not None:
                    output_file_path = output_path / jpg_file.name
                else:
                    output_file_path = None
                
                license_plate_count = self.process_single_image(
                    str(jpg_file), 
                    str(output_file_path) if output_file_path else None,
                    privacy_method,
                    confidence_threshold,
                    mosaic_size,
                    blur_strength
                )
                
                total_license_plates += license_plate_count
                processed_images += 1
                
                if license_plate_count > 0:
                    print(f"✓ {jpg_file.name}: Detected {license_plate_count} object(s)")
                
            except Exception as e:
                print(f"✗ Error processing image {jpg_file.name}: {str(e)}")
        
        print(f"\nProcessing completed!")
        print(f"Total images processed: {processed_images}")
        print(f"Total objects detected: {total_license_plates}")
        print(f"Average objects per image: {total_license_plates/processed_images:.2f}" if processed_images > 0 else "")

def main():
    parser = argparse.ArgumentParser(description='Privacy Masking Tool')
    parser.add_argument('--input_folder', '-i', required=True, help='Input image folder path')
    parser.add_argument('--output_folder', '-o', help='Output image folder path (optional, default overwrites original)')
    parser.add_argument('--model', '-m', default='yolov11l-license-plate.pt', 
                       help='YOLOv11 model file path (e.g., yolov11l-face.pt, license-plate-finetune-v1m.pt)')
    parser.add_argument('--method', '-method', choices=['mosaic', 'blur', 'black_box'], 
                       default='mosaic', help='Privacy protection method')
    parser.add_argument('--confidence', '-c', type=float, default=0.25, 
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--mosaic_size', '-ms', type=int, default=15, 
                       help='Mosaic block size')
    parser.add_argument('--blur_strength', '-bs', type=int, default=51, 
                       help='Blur strength (must be odd number)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        return
    
    masker = LicensePlatePrivacyMasker(args.model)
    
    masker.process_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        privacy_method=args.method,
        confidence_threshold=args.confidence,
        mosaic_size=args.mosaic_size,
        blur_strength=args.blur_strength
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("Privacy Masking Tool")
        print("=" * 50)
        
        print("\nAvailable models:")
        print("1. yolov11l-license-plate.pt (License Plate Detection)")
        print("2. yolov11l-face.pt (Face Detection)")
        print("3. Custom model")
        
        model_choice = input("\nSelect model (1/2/3) or enter custom path [default: 1]: ").strip()
        
        if model_choice == '2':
            model_path = 'yolov11l-face.pt'
        elif model_choice == '3':
            model_path = input("Enter custom model path: ").strip()
        else:
            model_path = 'yolov11l-license-plate.pt'
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            sys.exit(1)
        
        masker = LicensePlatePrivacyMasker(model_path)
        
        input_folder = r"D:\train_ed"
        output_folder = r"D:\train_last"

        method = 'mosaic'
        
        print(f"Using method: {method}")
        
        masker.process_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            privacy_method=method,
            confidence_threshold=0.25,
            mosaic_size=15,
            blur_strength=51
        )
    else:
        main()
