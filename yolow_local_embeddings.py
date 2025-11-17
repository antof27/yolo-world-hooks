import os
import cv2
import torch
import sys
import glob
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict # ### CHANGED ###: Useful for grouping

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- Parameters ---
### CHANGED ###: We still have an output directory
OUTPUT_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/yolo_predicted_embeddings"
MODEL_PATH = "/storage/team/EgoTracksFull/v2/egotracks/yolo_files/checkpoints/yolov8l_new_taxonomy.pt"
IMAGE_PATH = "/storage/team/EgoTracksFull/v2/all_training_frames"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CFSCORE = 0.2
# SAVE_INTERVAL is no longer needed

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Initialization ---
try:
    model_detect = YOLO(MODEL_PATH).to(device)
    model_embed = YOLO(MODEL_PATH).to(device)
    print(f"Models loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


def get_clip_id(filename):
    """Extracts the clip ID from the filename."""
    try:
        return filename.split('_')[0]
    except Exception:
        return None # Handle malformed filenames

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

# ### CHANGED ###: Load the set of already completed clip IDs
processed_clips = load_processed_clips(OUTPUT_DIR)

# Get all image files and sort them
try:
    all_image_files = sorted([f for f in os.listdir(IMAGE_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_images = len(all_image_files)
    print(f"Found {total_images} total images.")
except FileNotFoundError:
    print(f"Error: Image path not found: {IMAGE_PATH}")
    sys.exit(1)

# ### CHANGED ###: Group all image files by clip ID
print("Grouping images by clip ID...")
clips_to_process = defaultdict(list)
for image_file in tqdm(all_image_files, desc="Grouping files"):
    clip_id = get_clip_id(image_file)
    if clip_id is None:
        tqdm.write(f"Warning: Skipping malformed filename {image_file}")
        continue
    
    # This is the key: only add if the *entire clip* hasn't been processed
    if clip_id not in processed_clips:
        clips_to_process[clip_id].append(image_file)

print(f"Found {len(clips_to_process)} new clips to process.")
print(f"({len(processed_clips)} clips are already complete and will be skipped).")


# --- MAIN SCRIPT ---
with torch.no_grad():
    # ### CHANGED ###: The main loop now iterates over *clips*
    pbar = tqdm(clips_to_process.items(), desc="Processing clips")
    for clip_id, image_files_in_clip in pbar:
        
        pbar.set_description(f"Processing clip {clip_id[:10]}... ({len(image_files_in_clip)} images)")
        
        # Initialize a *new* buffer for *each* clip
        clip_data_buffer = initialize_data_structure()

        # ### CHANGED ###: Inner loop iterates over images *in this clip*
        # We sort the images in the clip to ensure correct frame order
        for image_file in sorted(image_files_in_clip):
            
            # (We already know this file isn't "processed" because
            # we check by clip, but this demonstrates the inner loop)
            
            image_path = os.path.join(IMAGE_PATH, image_file)
            
            image = cv2.imread(image_path)
            if image is None:
                tqdm.write(f"Error: could not read {image_file}")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                results = model_detect.predict(image_rgb, device=device, conf=CFSCORE, verbose=False)

                all_crops, all_bboxes, all_conf, all_cls = [], [], [], []

                for result in results:
                    if hasattr(result, 'boxes') and len(result.boxes) > 0:
                        for i in range(len(result.boxes)):
                            bbox = result.boxes.xyxy[i].cpu().numpy()
                            x1, y1, x2, y2 = map(int, bbox)
                            cropped = image_rgb[y1:y2, x1:x2]

                            if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                                tqdm.write(f"Skipping zero-area box in {image_file}")
                                continue

                            all_crops.append(cropped)
                            # Using float16 for size reduction
                            all_bboxes.append(result.boxes.xyxy[i].cpu().to(torch.float16))
                            all_conf.append(result.boxes.conf[i].cpu().item())
                            
                            cls_val = result.boxes.cls[i].cpu().item() if len(result.boxes.cls) > i else None
                            all_cls.append(cls_val)

                if len(all_crops) == 0:
                    clip_data_buffer["image_files"].append(image_file)
                    clip_data_buffer["embeddings"].append(torch.tensor([], dtype=torch.float16))
                    clip_data_buffer["bboxes"].append(torch.tensor([], dtype=torch.float16))
                    clip_data_buffer["scores"].append([])
                    clip_data_buffer["classes"].append([])
                    clip_data_buffer["n_objects"].append(0)
                    continue

                emb_list = model_embed.embed(all_crops, device=device)
                # Using float16 for size reduction
                batch_embeddings = torch.stack(emb_list).cpu().to(torch.float16)

                clip_data_buffer["image_files"].append(image_file)
                clip_data_buffer["embeddings"].append(batch_embeddings)
                clip_data_buffer["bboxes"].append(torch.stack(all_bboxes).cpu()) # Already float16
                clip_data_buffer["scores"].append(all_conf)
                clip_data_buffer["classes"].append(all_cls)
                clip_data_buffer["n_objects"].append(batch_embeddings.shape[0])

            except Exception as e:
                tqdm.write(f"Error processing {image_file}: {e}")
                continue

        # --- 6. Save (End of clip) ---
        # ### CHANGED ###: Save the *entire clip's data* to one file
        if len(clip_data_buffer['image_files']) > 0:
            output_path = os.path.join(OUTPUT_DIR, f"{clip_id}.pt")
            tqdm.write(f"Saving clip {clip_id} ({len(clip_data_buffer['image_files'])} images)...")
            save_data(output_path, clip_data_buffer)
        else:
            tqdm.write(f"No valid data processed for clip {clip_id}. Skipping save.")


# --- FINAL SUMMARY ---
print("\n=== FINAL SUMMARY ===")
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