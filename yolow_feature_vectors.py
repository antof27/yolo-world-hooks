import os
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
import matplotlib.pyplot as plt


trained_weights_path = "/home/coloranto/Documents/PhD/yolow_logits/yolov8s-world.pt"
validation_images_path = "/home/coloranto/Documents/PhD/yolow_logits/image" # Use the sample folder from your script
json_output_path = 'yolowfinetuned-queries-predictions-with-fv.json'

feature_map_dir =  "./feature_maps"
os.makedirs(feature_map_dir, exist_ok=True)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO(trained_weights_path).to(device)


for i, module in enumerate(model.model.model):
    print(i, module)

# setup hook
target_layer = model.model.model[-1].cv4[0][2]
feature_maps = {}

def hook_fn(module, input, output):
    if isinstance(output, tuple):
        feature_maps["feat"] = output[0].detach().cpu()
    else:
        feature_maps["feat"] = output.detach().cpu()

handle = target_layer.register_forward_hook(hook_fn)




# Initialize the formatted predictions list
formatted_predictions = []


for image_file in os.listdir(validation_images_path):
    if not image_file.endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(validation_images_path, image_file)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        continue

    #launch inference
    try:
        inference_results = model.predict(image, conf=0.1, verbose=False)
    except Exception as e:
        print(f"Error during predict() for {image_file}: {e}")
        continue
        
    predictions = []
    
    # Check if any results were returned
    if inference_results and inference_results[0].boxes:
        
        boxes_xyxy = inference_results[0].boxes.xyxy
        scores = inference_results[0].boxes.conf.cpu().tolist()
        classes = inference_results[0].boxes.cls.cpu().tolist()

        if "feat" in feature_maps:
            feat = feature_maps["feat"][0]
            heatmap = feat.mean(dim=0).numpy()
            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
            overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
            cv2.imwrite(os.path.join(feature_map_dir, f"featuremap_{image_file}"), overlay)


        # --- 5. Crop Objects from Image ---
        # Create a list of cropped images (in BGR format)
        image_crops = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = map(int, box)
            # Crop the image (OpenCV format: y:y, x:x)
            crop = image[y1:y2, x1:x2]
            image_crops.append(crop)

        # --- 6. Get Embeddings (Logit Vectors) ---
        if image_crops:
            try:
                # model.embed() returns a list of feature vectors (tensors)
                embeddings = model.embed(image_crops, device=device)
                
                # Convert tensors to simple Python lists
                embeddings_list = [emb.cpu().tolist() for emb in embeddings]

                # --- 7. Combine All Data ---
                for box, score, cl, emb in zip(boxes_xyxy.cpu().tolist(), scores, classes, embeddings_list):
                    predictions.append({
                        "category_id": int(cl),
                        "bbox": [round(x, 2) for x in box],
                        "score": round(score, 4),
                        "logit_vector": emb  # Here is your vector
                    })
            except Exception as e:
                print(f"Error while embedding crops for {image_file}: {e}")
                # Fallback: add predictions without embeddings if embed fails
                for box, score, cl in zip(boxes_xyxy.cpu().tolist(), scores, classes):
                    predictions.append({
                        "category_id": int(cl),
                        "bbox": [round(x, 2) for x in box],
                        "score": round(score, 4),
                        "logit_vector": [] # Add empty list
                    })
        
    # Add the image file and its predictions to the list
    formatted_predictions.append({
        "image_file": image_file,
        "predictions": predictions
    })

# --- 8. Save the final JSON file ---
try:
    with open(json_output_path, 'w') as json_file:
        json.dump(formatted_predictions, json_file, indent=4)
    print(f"\n Success! Predictions (with embeddings) saved to {json_output_path}")
except Exception as e:
    print(f"\n Error saving JSON file: {e}")