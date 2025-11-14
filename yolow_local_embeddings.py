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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)

embeddings_list = []
filename_list = []

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
        local_embedding = embeddings[0].cpu()
        print("len", len(local_embedding))

        embeddings_list.append(local_embedding)
        filename_list.append(image_file)
    except Exception as e:
        print(f"Error embedding {image_file}: {e}")
        continue



try:
    embeddings_tensor = torch.stack(embeddings_list)

    torch.save({
        "filenames": filename_list,
        "local_embeddings": embeddings_tensor
    }, PT_OUTPUT_PATH)

    print(f"Saved embeddings to {PT_OUTPUT_PATH}")
except Exception as e:
    print(f"Error saving embeddings: {e}")
