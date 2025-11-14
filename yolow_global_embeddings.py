import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Changed to .pt extension
PT_OUTPUT_PATH = os.path.join(current_dir, "yolow_embeddings.pt")
MODEL_PATH = os.path.join(current_dir, "yolov8s-world.pt")
IMAGE_PATH = os.path.join(current_dir, "image")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)
feature_maps = {}


def hook_fn(module, input, output):
    # Save the output feature map
    if isinstance(output, tuple):
        feature_maps["feat"] = output[0].detach().cpu()
    else:
        feature_maps["feat"] = output.detach().cpu()


# Store embeddings as tensors and filenames separately
embeddings_list = []
filenames_list = []

# --- Loop over images ---
for image_file in os.listdir(IMAGE_PATH):
    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(IMAGE_PATH, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read {image_file}")
        continue

    # --- Convert BGR -> RGB  ---
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        # YOLO embed expects list of images
        embeddings = model.embed([image_rgb], device=device)
        # Keep as tensor (move to CPU for storage)
        global_embedding = embeddings[0].cpu()
        print("Shape:", global_embedding.shape)
        
        embeddings_list.append(global_embedding)
        filenames_list.append(image_file)
    except Exception as e:
        print(f"Error embedding {image_file}: {e}")
        continue


# --- Save as PyTorch file ---
try:
    # Stack embeddings into a single tensor
    embeddings_tensor = torch.stack(embeddings_list)
    
    # Save both embeddings and filenames
    torch.save({
        'embeddings': embeddings_tensor,
        'filenames': filenames_list
    }, PT_OUTPUT_PATH)
    
    print(f"Success! Global embeddings saved to {PT_OUTPUT_PATH}")
    print(f"Saved {len(filenames_list)} embeddings with shape {embeddings_tensor.shape}")
except Exception as e:
    print(f"Error saving .pt file: {e}")