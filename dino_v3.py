# pip install torchao
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import numpy as np
import os 
from PIL import Image


IMG_PATH = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/image"
OUTPUT_JSON = "/dino_v3_output.json"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

image = os.listdir(IMG_PATH)[0]

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)

# def get_global_embeddings(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt").to(model.device)
#     with torch.inference_mode():
#         outputs = model(**inputs)

#     # pooler_output is a single vector representing the image, then we remove batch dimension and convert to list
#     pooled_output = outputs.pooler_output
#     print("Pooled output shape:", pooled_output.shape)
#     return pooled_output



# feature_list = []
# for fname in sorted(os.listdir(IMG_PATH)):
#     if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
#         continue
#     path = os.path.join(IMG_PATH, fname)
#     try:
#         emb = get_global_embeddings(path)
#         feature_list.append({
#             "image_file": fname,
#             "global_embedding": emb
#         })
#         print(f"Processed {fname}")
#     except Exception as e:
#         print(f"Error processing {fname}: {e}")

# # Save JSON
# import json
# with open(OUTPUT_JSON, 'w') as f:
#     json.dump(feature_list, f, indent=4)