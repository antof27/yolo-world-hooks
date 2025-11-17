import os
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
import os 
import sys
from tqdm import tqdm

# --- No changes needed up to here ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

PT_OUTPUT_PATH = os.path.join(current_dir, "yolow_local_embeddings_total.pt")
MODEL_PATH = "/storage/team/EgoTracksFull/v2/egotracks/yolo_files/checkpoints/yolov8l_new_taxonomy.pt"
IMAGE_PATH = "/storage/team/EgoTracksFull/v2/all_training_frames"

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
CFSCORE = 0.2

# One for detection (predict)
model_detect = YOLO(MODEL_PATH).to(device)
# One for embedding (embed)
model_embed = YOLO(MODEL_PATH).to(device)


organized_data = {}

for image_file in tqdm(sorted(os.listdir(IMAGE_PATH)), desc="Processing images"):
    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(IMAGE_PATH, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read {image_file}")
        continue
    
    # This print statement is fine, but I'm commenting it out to clean up the tqdm bar
    # print("Image_path", image_path) 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        # Run detection using the *detection* model
        results = model_detect.predict(image_rgb, device=device, conf=CFSCORE, verbose=False)
        
        all_crops = []
        all_bboxes = []
        all_confidences = []
        all_classes = []
        
        # Extract embeddings for each detected object
        for result in results:
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                
                for i in range(len(result.boxes)):
                    # Get bounding box
                    bbox = result.boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Crop the object region
                    cropped = image_rgb[y1:y2, x1:x2]
                    
                    # Add check for empty crops
                    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                        tqdm.write(f"Skipping zero-area box in {image_file}")
                        continue
                    
                    # Add data to lists to be batched
                    all_crops.append(cropped)
                    all_bboxes.append(result.boxes.xyxy[i].cpu())
                    all_confidences.append(result.boxes.conf[i].cpu().item())
                    all_classes.append(result.boxes.cls[i].cpu().item() if len(result.boxes.cls) > i else None)
                
            else:
                # This is normal, so just pass
                pass
        
        # Now, run embedding ONCE for all valid crops from this image
        if len(all_crops) > 0:
            
            # 1. model_embed.embed() returns a LIST of tensors (one per crop)
            list_of_embeddings = model_embed.embed(all_crops, device=device)
            
            # 2. Stack the list of tensors into a single tensor [N_objects, embedding_dim]
            # 3. Then move the final stacked tensor to the CPU
            batch_embeddings = torch.stack(list_of_embeddings).cpu()
            #print("batch_embeddings shape", batch_embeddings.shape)

            organized_data[image_file] = {
                'embeddings': batch_embeddings,              # [N_objects, embedding_dim]
                'bboxes': torch.stack(all_bboxes),          # [N_objects, 4]
                'score': all_confidences,                   # List of floats
                'class': all_classes,                       # List of class IDs
                'n_objects': len(all_crops)
            }
            # Use tqdm.write to avoid messing up the progress bar
            #tqdm.write(f"Processed and embedded {len(all_crops)} objects from {image_file}")
                
    except Exception as e:
        tqdm.write(f"Error processing {image_file}: {e}")
        continue


# --- No changes needed from here down ---
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
        print("\nNo objects detected in any images!")
        
except Exception as e:
    print(f"Error saving embeddings: {e}")