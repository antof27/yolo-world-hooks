import os
import torch
import json
import glob
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURATION ---
PRED_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/predicted_converted"
GT_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/gt_converted"

# Outputs
OUTPUT_JSON = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/merged_dataset.json"
OUTPUT_PT = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/merged_dataset_tensors.pt"

IOU_THRESHOLD = 0.7

def calculate_iou(box1, box2):
    """Calculates IoU between two boxes [x1, y1, x2, y2]."""
    b1 = box1.tolist() if isinstance(box1, torch.Tensor) else box1
    b2 = box2.tolist() if isinstance(box2, torch.Tensor) else box2

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection
    
    if union == 0: return 0.0
    return intersection / union

def load_and_group_by_image(pt_path):
    """Loads a .pt file and groups objects by image filename."""
    try:
        data = torch.load(pt_path, map_location='cpu')
    except Exception as e:
        return {}

    grouped = defaultdict(list)
    num_objs = len(data['object_ids'])
    
    for i in range(num_objs):
        obj_id = data['object_ids'][i]
        # Infer parent image from ID if not present in keys
        parent_img = data['parent_image'][i] if 'parent_image' in data else obj_id.rsplit('_', 1)[0]

        obj_data = {
            'original_id': obj_id,
            'embedding': data['embeddings'][i],
            'bbox': data['bboxes'][i],
            'score': data['scores'][i] if 'scores' in data else 1.0,
        }
        grouped[parent_img].append(obj_data)
        
    return grouped

def main():
    # Buffers for the .pt file
    master_embeddings = []
    master_bboxes = []
    master_ids = []
    master_labels = []
    master_image_refs = []

    # Buffer for the .json file
    json_index = []
    
    # Get file lists
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.pt")))
    gt_files_map = {os.path.basename(f): f for f in glob.glob(os.path.join(GT_DIR, "*.pt"))}
    
    print(f"Found {len(pred_files)} prediction files.")

    for pred_file in tqdm(pred_files, desc="Merging Data"):
        filename = os.path.basename(pred_file)
        
        # Load Predictions
        preds_by_image = load_and_group_by_image(pred_file)
        
        # Load GTs (if exists)
        gts_by_image = {}
        if filename in gt_files_map:
            gts_by_image = load_and_group_by_image(gt_files_map[filename])

        # Union of all images in this clip/file
        all_images = set(preds_by_image.keys()) | set(gts_by_image.keys())
        
        for img_name in all_images:
            img_preds = preds_by_image.get(img_name, [])
            img_gts = gts_by_image.get(img_name, [])

            # 1. Find max prediction index (to append GTs correctly)
            max_pred_index = 0
            for p in img_preds:
                try:
                    idx = int(p['original_id'].rsplit('_', 1)[-1])
                    if idx > max_pred_index: max_pred_index = idx
                except ValueError: pass
            
            # 2. Filter Predictions (Keep if IoU < 0.7)
            for p in img_preds:
                is_covered = False
                for g in img_gts:
                    if calculate_iou(p['bbox'], g['bbox']) >= IOU_THRESHOLD:
                        is_covered = True
                        break
                
                if not is_covered:
                    # Keep Prediction -> Label 0
                    master_embeddings.append(p['embedding'])
                    master_bboxes.append(p['bbox'])
                    master_ids.append(p['original_id'])
                    master_labels.append(0)
                    master_image_refs.append(img_name)
                    
                    json_index.append({
                        "id": p['original_id'],
                        "label": 0
                    })

            # 3. Add Ground Truths (Always Keep) -> Label 1
            for i, g in enumerate(img_gts):
                # Progressive ID: max_pred + 1 + current_gt_index
                new_id = f"{img_name}_{max_pred_index + i + 1:02d}"
                
                master_embeddings.append(g['embedding'])
                master_bboxes.append(g['bbox'])
                master_ids.append(new_id)
                master_labels.append(1)
                master_image_refs.append(img_name)
                
                json_index.append({
                    "id": new_id,
                    "label": 1
                })

    # --- Save JSON ---
    print(f"\nSaving JSON index ({len(json_index)} items) to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_index, f, indent = 2) # Add indent=2 if you want it readable (larger size)

    # --- Save PT ---
    print(f"Stacking tensors for .pt file...")
    if len(master_embeddings) > 0:
        # Stack lists into tensors for efficiency
        final_pt_data = {
            "object_ids": master_ids,
            "embeddings": torch.stack(master_embeddings).cpu(),
            "bboxes": torch.stack(master_bboxes).cpu(),
            "labels": torch.tensor(master_labels, dtype=torch.long),
            "image_ids": master_image_refs
        }
        
        print(f"Saving .pt data to {OUTPUT_PT}...")
        torch.save(final_pt_data, OUTPUT_PT)
        print("Done.")
    else:
        print("No data found to save.")

if __name__ == "__main__":
    main()