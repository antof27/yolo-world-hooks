import os
import cv2
import torch
import sys
import json
import glob  # ### CHANGED ###
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- Parameters ---
GT_JSON_PATH = "/storage/team/EgoTracksFull/v2/egotracks/train_dataset_versions/training_journal.json" 
### CHANGED ###: We now have an output *directory*
OUTPUT_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/yolo_gt_embeddings"
MODEL_PATH = "/storage/team/EgoTracksFull/v2/egotracks/yolo_files/checkpoints/yolov8l_new_taxonomy.pt"
IMAGE_PATH = "/storage/team/EgoTracksFull/v2/all_training_frames"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# SAVE_INTERVAL is no longer needed

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True) # ### CHANGED ###

# --- Model Initialization ---
try:
    model_embed = YOLO(MODEL_PATH).to(device)
    print(f"Embedding model loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


# ### NEW FUNCTION ###
def get_clip_id(filename):
    """Extracts the clip ID from the filename."""
    try:
        return filename.split('_')[0]
    except Exception:
        return None # Handle malformed filenames

# ### NEW FUNCTION ###
def load_processed_clips(output_dir):
    """
    Scans the output directory for completed .pt files and returns
    a set of processed clip IDs.
    """
    print("Scanning for existing processed clip files...")
    processed_clips = set()
    # Find all .pt files
    clip_files = glob.glob(os.path.join(output_dir, "*.pt"))
    
    for f in clip_files:
        # The filename *is* the clip ID (minus extension)
        clip_id = os.path.basename(f).replace('.pt', '')
        processed_clips.add(clip_id)

    if processed_clips:
        print(f"Found {len(processed_clips)} already processed clips.")
    else:
        print("No processed clips found. Starting from scratch.")
    return processed_clips

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
        tqdm.write(f"Error saving data to {output_path}: {e}")

# --- MAIN SCRIPT ---

# --- 1. Load and Process COCO JSON ---
print(f"Loading COCO JSON from {GT_JSON_PATH}...")
try:
    with open(GT_JSON_PATH, 'r') as f:
        coco_data = json.load(f)
    
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
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


# --- 2. Load existing processed data & Group files ---

# ### CHANGED ###: Load the set of already completed clip IDs
processed_clips = load_processed_clips(OUTPUT_DIR)

# Get all image files to be processed from the JSON
images_in_json = sorted(image_id_to_filename.items(), key=lambda item: item[0])

# ### CHANGED ###: Group all image files by clip ID
print("Grouping images by clip ID...")
clips_to_process = defaultdict(list)
for image_id, image_file in tqdm(images_in_json, desc="Grouping files"):
    clip_id = get_clip_id(image_file)
    if clip_id is None:
        tqdm.write(f"Warning: Skipping malformed filename {image_file}")
        continue
    
    # Only add if the *entire clip* hasn't been processed
    if clip_id not in processed_clips:
        # We need both the ID and filename for processing
        clips_to_process[clip_id].append((image_id, image_file))

print(f"Found {len(clips_to_process)} new clips to process.")
print(f"({len(processed_clips)} clips are already complete and will be skipped).")


# --- 3. Main Processing Loop ---
with torch.no_grad():
    # ### CHANGED ###: The main loop now iterates over *clips*
    pbar = tqdm(clips_to_process.items(), desc="Processing clips")
    for clip_id, image_data_list in pbar:
        
        pbar.set_description(f"Processing clip {clip_id[:10]}... ({len(image_data_list)} images)")
        
        # Initialize a *new* buffer for *each* clip
        clip_data_buffer = initialize_data_structure()

        # ### CHANGED ###: Inner loop iterates over images *in this clip*
        # We sort by filename to ensure correct frame order
        for image_id, image_file in sorted(image_data_list, key=lambda x: x[1]):

            image_path = os.path.join(IMAGE_PATH, image_file)
            
            image = cv2.imread(image_path)
            if image is None:
                tqdm.write(f"Error: could not read {image_file} (ID: {image_id})")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image_rgb.shape[:2]

            try:
                annotations = gt_annotations[image_id]
                
                all_crops, all_bboxes, all_scores, all_cls = [], [], [], []

                if len(annotations) > 0:
                    for ann in annotations:
                        bbox_coco = ann['bbox'] # [x, y, w, h]
                        
                        x1 = bbox_coco[0]
                        y1 = bbox_coco[1]
                        x2 = x1 + bbox_coco[2]
                        y2 = y1 + bbox_coco[3]
                        
                        x1_c = max(0, int(x1))
                        y1_c = max(0, int(y1))
                        x2_c = min(img_w, int(x2))
                        y2_c = min(img_h, int(y2))

                        cropped = image_rgb[y1_c:y2_c, x1_c:x2_c]

                        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                            tqdm.write(f"Skipping zero-area GT box in {image_file} (Ann ID: {ann['id']})")
                            continue

                        all_crops.append(cropped)
                        # ### Quantizing to float16 ###
                        all_bboxes.append(torch.tensor([x1, y1, x2, y2]).cpu().to(torch.float16)) 
                        all_scores.append(1.0) # GT score is 1.0
                        all_cls.append(ann['category_id'])


                if len(all_crops) == 0:
                    clip_data_buffer["image_files"].append(image_file)
                    clip_data_buffer["embeddings"].append(torch.tensor([], dtype=torch.float16))
                    clip_data_buffer["bboxes"].append(torch.tensor([], dtype=torch.float16))
                    clip_data_buffer["scores"].append([])
                    clip_data_buffer["classes"].append([])
                    clip_data_buffer["n_objects"].append(0)
                    continue

                emb_list = model_embed.embed(all_crops, device=device)
                # ### Quantizing to float16 ###
                batch_embeddings = torch.stack(emb_list).cpu().to(torch.float16)

                clip_data_buffer["image_files"].append(image_file)
                clip_data_buffer["embeddings"].append(batch_embeddings)
                clip_data_buffer["bboxes"].append(torch.stack(all_bboxes).cpu()) # Already float16
                clip_data_buffer["scores"].append(all_scores)
                clip_data_buffer["classes"].append(all_cls)
                clip_data_buffer["n_objects"].append(batch_embeddings.shape[0])
                
                # ### DELETED ###: The incremental save block was here

            except Exception as e:
                tqdm.write(f"Error processing {image_file}: {e}")
                continue

        # --- 4. Save (End of clip) ---
        # ### CHANGED ###: Save the *entire clip's data* to one file
        if len(clip_data_buffer['image_files']) > 0:
            output_path = os.path.join(OUTPUT_DIR, f"{clip_id}.pt")
            tqdm.write(f"Saving clip {clip_id} ({len(clip_data_buffer['image_files'])} images)...")
            save_data(output_path, clip_data_buffer)
        else:
            tqdm.write(f"No valid data processed for clip {clip_id}. Skipping save.")


# --- FINAL SUMMARY ---
# ### CHANGED ###: We need to load all files to get the true summary
print("\n=== FINAL SUMMARY (Ground Truth) ===")
print("Calculating final totals from all clip files...")
all_clip_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.pt")))
total_images_final = 0
total_objects_final = 0

for f in tqdm(all_clip_files, desc="Summarizing"):
    try:
        data = torch.load(f, map_location='cpu')
        total_images_final += len(data['image_files'])
        total_objects_final += sum(data['n_objects'])
    except Exception as e:
        print(f"Error reading {f} for summary: {e}")

print(f"Total clips processed: {len(all_clip_files)}")
print(f"Total images processed (across all files): {total_images_final}")
print(f"Total objects extracted (across all files): {total_objects_final}")