# pip install torchao
import torch
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification
import numpy as np
from transformers.image_utils import load_image
import os 
from PIL import Image


IMG_PATH = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/image"
OUTPUT_JSON = "dino_v2_output.json"
MODEL_NAME = "facebook/dinov2-vits8"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant-imagenet1k-1-layer')

model = AutoModelForImageClassification.from_pretrained(
    'facebook/dinov2-giant-imagenet1k-1-layer',
    dtype=torch.bfloat16,
    device_map="auto",
)

def get_global_embeddings(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    # pooler_output is a single vector representing the image, then we remove batch dimension and convert to list
    embeggings = outputs.pooler_output.cpu().squeeze().tolist()
    return embeggings



feature_list = []
for fname in sorted(os.listdir(IMG_PATH)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    path = os.path.join(IMG_PATH, fname)
    try:
        emb = get_global_embeddings(path)
        feature_list.append({
            "image_file": fname,
            "global_embedding": emb
        })
        print(f"Processed {fname}")
    except Exception as e:
        print(f"Error processing {fname}: {e}")

# Save JSON
import json
with open(OUTPUT_JSON, 'w') as f:
    json.dump(feature_list, f, indent=4)