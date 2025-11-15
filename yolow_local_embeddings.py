import os
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

PT_OUTPUT_PATH = os.path.join(current_dir, "yolow_local_embeddings.pt")
MODEL_PATH = os.path.join(current_dir, "checkpoints/yolov8s-world.pt")
IMAGE_PATH = os.path.join(current_dir, "image")

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)

# Store data organized by filename
organized_data = {}

# --- Loop over images ---
for image_file in sorted(os.listdir(IMAGE_PATH)):
    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(IMAGE_PATH, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read {image_file}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        # Run detection to get object-level embeddings
        results = model.predict(image_rgb, device=device, verbose=False)
        
        # Initialize lists for this image
        image_embeddings = []
        image_bboxes = []
        image_confidences = []
        image_classes = []
        
        # Extract embeddings for each detected object
        for result in results:
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                
                for i in range(len(result.boxes)):
                    # Get bounding box
                    bbox = result.boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Crop the object region
                    cropped = image_rgb[y1:y2, x1:x2]
                    
                    # Get embedding for this crop
                    crop_embedding = model.embed([cropped], device=device)[0].cpu()
                    
                    # Append to lists for this image
                    image_embeddings.append(crop_embedding)
                    image_bboxes.append(result.boxes.xyxy[i].cpu())
                    image_confidences.append(result.boxes.conf[i].cpu().item())
                    image_classes.append(result.boxes.cls[i].cpu().item() if len(result.boxes.cls) > i else None)
                
                print(f"Processed {len(result.boxes)} objects from {image_file}")
            else:
                print(f"No objects detected in {image_file}")
        
        # Store data for this image if objects were detected
        if len(image_embeddings) > 0:
            organized_data[image_file] = {
                'embeddings': torch.stack(image_embeddings),  # [N_objects, embedding_dim]
                'bboxes': torch.stack(image_bboxes),          # [N_objects, 4]
                'confidences': image_confidences,              # List of floats
                'classes': image_classes,                      # List of class IDs
                'n_objects': len(image_embeddings)
            }
                
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        continue



try:
    if len(organized_data) > 0:
        torch.save(organized_data, PT_OUTPUT_PATH)
        
        total_objects = sum(data['n_objects'] for data in organized_data.values())
        print(f"\nSaved embeddings to {PT_OUTPUT_PATH}")
        print(f"Total images processed: {len(organized_data)}")
        print(f"Total objects detected: {total_objects}")
        
        # Print summary per image
        print("\nSummary per image:")
        for filename, data in organized_data.items():
            print(f"  {filename}: {data['n_objects']} objects, embedding shape: {data['embeddings'].shape}")
    else:
        print("No objects detected in any images!")
        
except Exception as e:
    print(f"Error saving embeddings: {e}")