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



JSON_OUTPUT_PATH = os.path.join(current_dir, "yolow_embeddings.json")
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



image_embeddings = []

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
        # Convert tensor to list
        global_embedding = embeddings[0].cpu().tolist()
        print("len", len(global_embedding))
    except Exception as e:
        print(f"Error embedding {image_file}: {e}")
        global_embedding = []

    # --- Store in JSON format ---
    image_embeddings.append({
        "image_file": image_file,
        "global_embedding": global_embedding
    })


try:
    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(image_embeddings, f, indent=4)
    print(f"Success! Global embeddings saved to {JSON_OUTPUT_PATH}")
except Exception as e:
    print(f"Error saving JSON: {e}")


