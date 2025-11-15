import torch
from transformers import AutoImageProcessor, AutoModel
import os 
from PIL import Image


IMG_PATH = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/image"
PT_OUTPUT_PATH = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/dino_v2_embeddings.pt"


device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

pretrained_model_name = "facebook/dinov2-base" 



processor = AutoImageProcessor.from_pretrained(pretrained_model_name, use_fast = True)
model = AutoModel.from_pretrained(pretrained_model_name).to(device)


dino_embeddings = {}

for image_file in sorted(os.listdir(IMG_PATH)):
    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(IMG_PATH, image_file)
    image = Image.open(image_path).convert("RGB")

    try: 
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
        
        pooled_output = outputs.pooler_output.cpu().squeeze(0)
        dino_embeddings[image_file] = pooled_output

    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        continue

if len(dino_embeddings) > 0:
    torch.save(dino_embeddings, PT_OUTPUT_PATH)
    print(f"\nSaved {len(dino_embeddings)} image embeddings to {PT_OUTPUT_PATH}")
else:
    print("No images processed!")