import os
import torch
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
# Directory containing your CURRENT DINOv2 .pt files
INPUT_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/dino"

# Directory where you want the NEW converted files to be saved
OUTPUT_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/dino_converted"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_clip_data(input_path, output_path):
    """
    Reads a DINOv2 .pt file (image_file -> embedding) and saves a .pt file
    organized into flattened lists of object_ids and embeddings.
    """
    try:
        data = torch.load(input_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    # Initialize new flattened dictionary
    new_data = {
        "object_ids": [],   # e.g., "filename_01" (since it's a whole-image embedding)
        "embeddings": [],   # Flattened list of tensors
    }

    # DINOv2 script saves image_file as the key, embedding as the value
    for image_file, embedding in data.items():
        # 1. Create the unique ID.
        # Since DINOv2 processes the whole image, we treat the whole image 
        # as a single "object" with the ID suffix "_01" to match the converter 
        # logic, but using the filename to identify the frame.
        
        # Split the filename to get the part before the extension (e.g., 'clipid_frameid.jpg' -> 'clipid_frameid')
        object_id_base = image_file.rsplit('.', 1)[0]
        # Append "_01" to denote the whole image/single object
        object_id = f"{object_id_base}"

        new_data["object_ids"].append(object_id)
        # The embedding is already a tensor [Dim], so we just append it
        new_data["embeddings"].append(embedding)

    # Save the new structure
    if len(new_data["object_ids"]) > 0:
        # Saving as a dictionary with lists of tensors/strings
        torch.save(new_data, output_path)
        print(f"Successfully converted and saved {len(new_data['object_ids'])} items to {output_path}")
    else:
        print(f"No embeddings found in {input_path}, skipping save.")

def main():
    # Find all existing .pt files from the DINOv2 output directory
    clip_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pt")))
    
    if not clip_files:
        print(f"No .pt files found in {INPUT_DIR}. Please check your INPUT_DIR.")
        return

    print(f"Found {len(clip_files)} files to convert.")

    # Process with progress bar
    for input_pt in tqdm(clip_files, desc="Converting DINOv2 files"):
        # Create corresponding output filename (same name, new directory)
        basename = os.path.basename(input_pt)
        output_pt = os.path.join(OUTPUT_DIR, basename)
        
        convert_clip_data(input_pt, output_pt)

    print("\nConversion complete!")
    print(f"New files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()