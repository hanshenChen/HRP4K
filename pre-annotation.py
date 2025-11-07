import argparse
import os
import cv2.dnn
import numpy as np
import json
import glob

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml("./road.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print(CLASSES)

score_thresh = 0.25
nms_thresh = 0.45
output_image_dir = "output_images"
output_label_dir = "output_labels"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

def is_only_manhole(preds):
    for pred in preds:
        if pred['class_name']!='Manhole': 
            return False
    return True

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def inference_single_frame2(model, frame):
    original_image = frame
    height, width, _ = original_image.shape

    length = max(height, width)
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    scale = length / 1280.0

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(1280, 1280), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()
    outputs = outputs[0]

    boxes = []
    scores = []
    class_ids = []

    for i in range(outputs.shape[0]):
        x, y, w, h = outputs[i, 0:4]
        objectness = outputs[i,4]
        classes_scores = outputs[i,5:]
        class_id = np.argmax(classes_scores)
        class_score = classes_scores[class_id]
        final_score = objectness * class_score

        if final_score >= score_thresh:
            x_min = (x - w/2) * scale
            y_min = (y - h/2) * scale
            w = w * scale
            h = h * scale

            x_min = int(x_min)
            y_min = int(y_min)
            x_max = x_min + int(w)
            y_max = y_min + int(h)

            x_min = max(0, min(x_min, width - 1))
            y_min = max(0, min(y_min, height - 1))
            x_max = max(0, min(x_max, width - 1))
            y_max = max(0, min(y_max, height - 1))

            boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
            scores.append(float(final_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_thresh, nms_thresh)

    detections = []
    if len(indices) > 0:
        for idx in indices:
            idx = int(idx)
            x_min = boxes[idx][0]
            y_min = boxes[idx][1]
            w = boxes[idx][2]
            h = boxes[idx][3]
            x_max = x_min + w
            y_max = y_min + h

            detections.append({
                "class_id": class_ids[idx],
                "class_name": CLASSES[class_ids[idx]],
                "confidence": scores[idx],
                "bbox": [x_min, y_min, x_max, y_max]
            })
        return detections
    else:
        print("No detections after NMS.")
        return None

def inference_single_frame(model, frame):
    original_image = frame
    height, width, _ = original_image.shape

    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    scale = length / 1280

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(1280, 1280), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= score_thresh:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(float(maxScore))
            class_ids.append(int(maxClassIndex))

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_thresh, nms_thresh)

    detections = []
    if len(indices) > 0:
        for idx in indices:
            idx = int(idx)
            box = boxes[idx]
            cls_id = class_ids[idx]
            score = scores[idx]

            x_min = round(box[0] * scale)
            y_min = round(box[1] * scale)
            w = round(box[2] * scale)
            h = round(box[3] * scale)
            x_max = x_min + w
            y_max = y_min + h

            x_min = max(0, min(x_min, width - 1))
            y_min = max(0, min(y_min, height - 1))
            x_max = max(0, min(x_max, width - 1))
            y_max = max(0, min(y_max, height - 1))

            detections.append({
                "class_id": cls_id,
                "class_name": CLASSES[cls_id],
                "confidence": score,
                "bbox": [x_min, y_min, x_max, y_max]
            })
        return detections
    else:
        print("No detections after NMS.")
        return None


def save_labelme_json(json_path, image_filename, detections, image_shape):
    """Save detection results in LabelMe JSON format."""
    shapes = []
    for det in detections:
        x_min, y_min, x_max, y_max = det["bbox"]
        shape = {
            "label": det["class_name"],
            "points": [[float(x_min), float(y_min)], [float(x_max), float(y_max)]],
            "group_id": None,
            "description":"",
            "shape_type": "rectangle",
            "flags": {}
        }
        shapes.append(shape)

    labelme_json = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": image_shape[0],
        "imageWidth": image_shape[1]
    }

    print(f"Saving JSON: {json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_json, f, ensure_ascii=False, indent=4)


def main(onnx_model, input_folder):
    model = cv2.dnn.readNetFromONNX(onnx_model)

    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"Error: No JPG images found in folder: {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process.")
    
    processed_count = 0
    detected_count = 0
    
    for image_path in image_files:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image: {image_path}")
            continue
        
        processed_count += 1
        print(f"Processing [{processed_count}/{len(image_files)}]: {os.path.basename(image_path)}")
        
        detections = inference_single_frame(model, frame)
        
        if detections is not None and len(detections) > 0:
            detected_count += 1
            print(f"  Found {len(detections)} detections")
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            output_image_name = base_name + ".jpg"
            output_image_path = os.path.join(output_image_dir, output_image_name)
            cv2.imwrite(output_image_path, frame)
            
            output_json_path = os.path.join(output_label_dir, base_name + ".json")
            save_labelme_json(output_json_path, output_image_name, detections, frame.shape)
        else:
            print(f"  No detections found")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total images processed: {processed_count}")
    print(f"Images with detections: {detected_count}")
    print(f"Output images saved to: {output_image_dir}")
    print(f"Output labels saved to: {output_label_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=r"D:\best.onnx", 
                        help="Path to your ONNX model.")
    parser.add_argument("--folder", default=r"D:\HRP4K\valid\Images", 
                        help="Path to input folder containing JPG images.")
    args = parser.parse_args()
    
    import onnxruntime
    session = onnxruntime.InferenceSession(args.model)
    outputs = session.get_outputs()

    print("ONNX Model Information:")
    for i, output in enumerate(outputs):
        print(f"Output {i}:")
        print(f"  Name: {output.name}")
        print(f"  Shape: {output.shape}")
        print(f"  Type: {output.type}")
    print()
    
    main(args.model, args.folder)