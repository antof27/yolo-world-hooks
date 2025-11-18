import os
import torch
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
# Directory containing your CURRENT .pt files (the ones created by your YOLOW script)
INPUT_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/predicted"

# Directory where you want the NEW converted files to be saved
OUTPUT_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/predicted_converted"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_clip_data(input_path, output_path):
    """
    Reads a .pt file organized by image and saves a .pt file organized by object.
    """
    try:
        # Load data on CPU to avoid GPU memory issues during simple conversion
        data = torch.load(input_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    # Initialize new flattened dictionary
    # We rename 'image_files' to 'object_ids' to reflect the new granularity
    new_data = {
        "object_ids": [],   # e.g., "filename.jpg_01"
        "embeddings": [],   # Flattened list of tensors
        "bboxes": [],       # Flattened list of tensors
        "scores": [],       # Flattened list of floats
        "classes": [],      # Flattened list of ints
    }

    total_images = len(data['image_files'])

    # Iterate over every image in the file
    for i in range(total_images):
        image_file = data['image_files'][i]
        n_objs = data['n_objects'][i]

        # Skip images with 0 objects
        if n_objs == 0:
            continue

        # Extract the lists/tensors for this specific image
        # Embeddings are typically a tensor of shape [N, Dim] inside a list
        img_embeddings = data['embeddings'][i] 
        img_bboxes = data['bboxes'][i]
        img_scores = data['scores'][i]
        img_classes = data['classes'][i]

        # Sanity check to ensure lengths match n_objects
        if not (len(img_embeddings) == n_objs and len(img_bboxes) == n_objs):
            print(f"Warning: Mismatch in data lengths for {image_file}. Skipping.")
            continue

        # Iterate over every object in the image
        for j in range(n_objs):
            # Create the unique ID: filename + _01, _02, etc.
            # removing the extension for the ID often looks cleaner, 
            # but based on your request, I append directly to the filename.
            # We use j+1 to get 1-based indexing (01, 02) instead of (00, 01)
            object_id = image_file.split('.')[0]
            object_id = f"{object_id}_{j+1:02d}"

            new_data["object_ids"].append(object_id)
            new_data["embeddings"].append(img_embeddings[j])
            new_data["bboxes"].append(img_bboxes[j])
            new_data["scores"].append(img_scores[j])
            new_data["classes"].append(img_classes[j])

    # If the file had valid objects, save the new structure
    if len(new_data["object_ids"]) > 0:
        # Optional: Stack embeddings/bboxes into a single large tensor 
        # if you prefer one massive tensor over a list of small tensors.
        # keeping them as lists is safer if you plan to concatenate later.
        # For now, we leave them as lists of tensors to match the 'columns' logic.
        
        torch.save(new_data, output_path)
    else:
        # If original file had data but no objects detected at all
        print(f"No objects found in {input_path}, skipping save.")

def main():
    # Find all existing .pt files
    clip_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pt")))
    
    if not clip_files:
        print(f"No .pt files found in {INPUT_DIR}")
        return

    print(f"Found {len(clip_files)} files to convert.")

    # Process with progress bar
    for input_pt in tqdm(clip_files, desc="Converting files"):
        # Create corresponding output filename
        basename = os.path.basename(input_pt)
        output_pt = os.path.join(OUTPUT_DIR, basename)
        
        convert_clip_data(input_pt, output_pt)

    print("\nConversion complete!")
    print(f"New files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()