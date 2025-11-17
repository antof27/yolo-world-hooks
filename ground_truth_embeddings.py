import os
import cv2
import torch
import sys
import json
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


GT_JSON_PATH = "/storage/team/EgoTracksFull/v2/egotracks/train_dataset_versions/training_journal.json" 
GT_PT_OUTPUT_PATH = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/output_files/yolow_embeddings_GROUND_TRUTH.pt"
MODEL_PATH = "/storage/team/EgoTracksFull/v2/egotracks/yolo_files/checkpoints/yolov8l_new_taxonomy.pt"
IMAGE_PATH = "/storage/team/EgoTracksFull/v2/all_training_frames"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAVE_INTERVAL = 10000 

# --- Model Initialization ---
# We only need the embedding model for this task
try:
    model_embed = YOLO(MODEL_PATH).to(device)
    print(f"Embedding model loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


def load_existing_data(output_path):
    """Loads existing data or initializes a new structure."""
    if os.path.exists(output_path):
        try:
            data = torch.load(output_path, map_location='cpu')
            # Ensure all expected keys exist
            if not all(k in data for k in ["image_files", "embeddings", "bboxes", "scores", "classes", "n_objects"]):
                print(f"Warning: {output_path} is missing keys. Re-initializing.")
                return initialize_data_structure()
            
            print(f"Loaded existing data. {len(data['image_files'])} images processed.")
            return data
        except Exception as e:
            print(f"Error loading {output_path}: {e}. Re-initializing.")
            return initialize_data_structure()
    else:
        print("No existing file found. Initializing new data structure.")
        return initialize_data_structure()

def initialize_data_structure():
    """Returns a new, empty data dictionary."""
    return {
        "image_files": [],
        "embeddings": [],
        "bboxes": [],
        "scores": [],
        "classes": [],
        "n_objects": []
    }

def save_data(output_path, data):
    """Saves the data structure to the .pt file."""
    try:
        torch.save(data, output_path)
    except Exception as e:
        # Use tqdm.write instead of print to avoid messing up the progress bar
        tqdm.write(f"Error saving data to {output_path}: {e}")

# --- MAIN SCRIPT ---

# --- 1. Load and Process COCO JSON ---
print(f"Loading COCO JSON from {GT_JSON_PATH}...")
try:
    with open(GT_JSON_PATH, 'r') as f:
        coco_data = json.load(f)
    
    # Create a map of image_id -> file_name
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Create a map of image_id -> list of annotations
    gt_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        gt_annotations[ann['image_id']].append(ann)
        
    print(f"Loaded {len(image_id_to_filename)} images and {len(coco_data['annotations'])} annotations.")

except FileNotFoundError:
    print(f"Error: JSON file not found: {GT_JSON_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"Error reading JSON file: {e}")
    sys.exit(1)


# --- 2. Load existing processed data ---
all_data = load_existing_data(GT_PT_OUTPUT_PATH)
processed_files = set(all_data['image_files'])

# Get all image files to be processed from the JSON
# Sort by image_id for consistent order
images_to_process = sorted(image_id_to_filename.items(), key=lambda item: item[0])
total_images = len(images_to_process)
print(f"Found {total_images} total images in JSON file.")

# Temporary storage for new data
new_data_buffer = initialize_data_structure()
new_images_processed = 0
total_images_processed_in_run = 0

with torch.no_grad():  # Disable gradient calculations for inference
    # Wrap the enumerator in tqdm for the progress bar
    pbar = tqdm(images_to_process, desc="Processing GT images")
    for image_id, image_file in pbar:
        
        if image_file in processed_files:
            continue

        image_path = os.path.join(IMAGE_PATH, image_file)
        
        image = cv2.imread(image_path)
        if image is None:
            tqdm.write(f"Error: could not read {image_file} (ID: {image_id})")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]

        try:
            # --- 5. Get Ground Truth Annotations ---
            annotations = gt_annotations[image_id]
            
            all_crops, all_bboxes, all_scores, all_cls = [], [], [], []

            if len(annotations) > 0:
                for ann in annotations:
                    bbox_coco = ann['bbox'] # Format: [x, y, w, h]
                    
                    # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
                    x1 = bbox_coco[0]
                    y1 = bbox_coco[1]
                    x2 = x1 + bbox_coco[2]
                    y2 = y1 + bbox_coco[3]
                    
                    # Clip coordinates to be within image bounds
                    x1_c = max(0, int(x1))
                    y1_c = max(0, int(y1))
                    x2_c = min(img_w, int(x2))
                    y2_c = min(img_h, int(y2))

                    cropped = image_rgb[y1_c:y2_c, x1_c:x2_c]

                    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                        tqdm.write(f"Skipping zero-area GT box in {image_file} (Ann ID: {ann['id']})")
                        continue

                    all_crops.append(cropped)
                    # Store the *un-clipped* original box coordinates as a tensor
                    # to match the format of predicted boxes (which are also not clipped)
                    all_bboxes.append(torch.tensor([x1, y1, x2, y2]).cpu()) 
                    
                    # Use placeholder 1.0 for score 
                    all_scores.append(1.0) 
                    all_cls.append(ann['category_id'])


            if len(all_crops) == 0:
                # Still log this image as "processed" to avoid re-checking
                # but don't add any embedding data
                new_data_buffer["image_files"].append(image_file)
                new_data_buffer["embeddings"].append(torch.tensor([])) # Empty tensor
                new_data_buffer["bboxes"].append(torch.tensor([]))     # Empty tensor
                new_data_buffer["scores"].append([])
                new_data_buffer["classes"].append([])
                new_data_buffer["n_objects"].append(0)
                new_images_processed += 1
                total_images_processed_in_run += 1
                continue

            # --- 6. Get Embeddings ---
            emb_list = model_embed.embed(all_crops, device=device)
            batch_embeddings = torch.stack(emb_list).cpu() # Stack to [N, D]

            # --- 7. Add to buffer ---
            new_data_buffer["image_files"].append(image_file)
            new_data_buffer["embeddings"].append(batch_embeddings)
            new_data_buffer["bboxes"].append(torch.stack(all_bboxes).cpu()) # Stack to [N, 4]
            new_data_buffer["scores"].append(all_scores)
            new_data_buffer["classes"].append(all_cls)
            new_data_buffer["n_objects"].append(batch_embeddings.shape[0])

            new_images_processed += 1
            total_images_processed_in_run += 1

            # --- 8. Save incrementally ---
            if new_images_processed % SAVE_INTERVAL == 0:
                pbar.set_description(f"Processed {total_images_processed_in_run} new. Saving...")
                
                for key in all_data.keys():
                    all_data[key].extend(new_data_buffer[key])
                
                save_data(GT_PT_OUTPUT_PATH, all_data)
                
                new_data_buffer = initialize_data_structure()
                processed_files.update(all_data['image_files'])
                pbar.set_description("Processing GT images")


        except Exception as e:
            tqdm.write(f"Error processing {image_file}: {e}")
            continue

# --- FINAL SAVE ---
if len(new_data_buffer['image_files']) > 0:
    print(f"Saving remaining {len(new_data_buffer['image_files'])} images...")
    for key in all_data.keys():
        all_data[key].extend(new_data_buffer[key])
    
    save_data(GT_PT_OUTPUT_PATH, all_data)
    print("Final save complete.")


# --- FINAL SUMMARY ---
final = load_existing_data(GT_PT_OUTPUT_PATH) # Re-load to be sure
print("\n=== FINAL SUMMARY (Ground Truth) ===")
print(f"Total images: {len(final['image_files'])}")
print(f"Total objects: {sum(final['n_objects'])}")