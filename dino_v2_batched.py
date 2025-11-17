import torch
from transformers import AutoImageProcessor, AutoModel
import os
from PIL import Image
from tqdm import tqdm
import glob  # ### CHANGED ###
from collections import defaultdict # ### CHANGED ###

# --- Parameters ---
IMG_PATH = "/storage/team/EgoTracksFull/v2/all_training_frames"
### CHANGED ###: We now have an output *directory*
OUTPUT_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/dino_embeddings"

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

pretrained_model_name = "facebook/dinov2-base" 

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True) # ### CHANGED ###

# --- Helper Functions ---

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
    clip_files = glob.glob(os.path.join(output_dir, "*.pt"))
    
    for f in clip_files:
        clip_id = os.path.basename(f).replace('.pt', '')
        processed_clips.add(clip_id)

    if processed_clips:
        print(f"Found {len(processed_clips)} already processed clips.")
    else:
        print("No processed clips found. Starting from scratch.")
    return processed_clips

# --- Model and Processor Initialization ---
print(f"Loading model {pretrained_model_name}...")
try:
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name, use_fast=True)
    model = AutoModel.from_pretrained(pretrained_model_name).to(device)
    print(f"Model loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


# --- 1. Find and Group All Image Files ---
print("Scanning and grouping image files...")
valid_ext = ('.jpg', '.jpeg', '.png')
all_image_files = sorted([f for f in os.listdir(IMG_PATH) if f.lower().endswith(valid_ext)])

# ### CHANGED ###: Load processed clips and group files
processed_clips = load_processed_clips(OUTPUT_DIR)
clips_to_process = defaultdict(list)

for image_file in tqdm(all_image_files, desc="Grouping files"):
    clip_id = get_clip_id(image_file)
    if clip_id is None:
        tqdm.write(f"Warning: Skipping malformed filename {image_file}")
        continue
    
    if clip_id not in processed_clips:
        clips_to_process[clip_id].append(image_file)

print(f"Found {len(clips_to_process)} new clips to process.")
print(f"({len(processed_clips)} clips are already complete and will be skipped).")


# --- 2. Main Processing Loop (by Clip) ---

# ### CHANGED ###: Loop over clips, not individual files
pbar = tqdm(clips_to_process.items(), desc="Processing clips")
for clip_id, image_files_in_clip in pbar:
    
    pbar.set_description(f"Processing clip {clip_id[:10]}... ({len(image_files_in_clip)} images)")
    
    # This dictionary holds embeddings *only for this clip*
    clip_embeddings = {}

    # Inner loop for images *in this clip*
    for image_file in sorted(image_files_in_clip):
        image_path = os.path.join(IMG_PATH, image_file)

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.inference_mode():
                outputs = model(**inputs)

            # ### CHANGED ###: Quantize to float16 and save to clip dict
            pooled_output = outputs.pooler_output.squeeze(0).to(torch.float16).cpu()
            clip_embeddings[image_file] = pooled_output

        except Exception as e:
            tqdm.write(f"Error processing {image_file}: {e}")
            continue

    # --- 3. Save Clip Results ---
    if clip_embeddings:
        output_path = os.path.join(OUTPUT_DIR, f"{clip_id}.pt")
        torch.save(clip_embeddings, output_path)
        tqdm.write(f"Saved {len(clip_embeddings)} embeddings for clip {clip_id}")
    else:
        tqdm.write(f"No images processed for clip {clip_id}. Skipping save.")

print("\nProcessing complete.")

# --- 4. Final Summary ---
print("\n=== FINAL SUMMARY (DINOv2) ===")
print("Calculating final totals from all clip files...")
all_clip_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.pt")))
total_images_final = 0

for f in tqdm(all_clip_files, desc="Summarizing"):
    try:
        # We just need the *length* of the dictionary, which is fast to load
        data = torch.load(f, map_location='cpu')
        total_images_final += len(data)
    except Exception as e:
        print(f"Error reading {f} for summary: {e}")

print(f"Total clips processed: {len(all_clip_files)}")
print(f"Total images processed (across all files): {total_images_final}")