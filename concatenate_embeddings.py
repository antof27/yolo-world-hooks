import torch 

yolow_emb = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/yolow_local_embeddings.pt")
dino_emb = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/dino_v2_embeddings.pt")


concatenated_embeddings = {}
for image_file in yolow_emb:
    
    yolow_data = yolow_emb[image_file]
    yolow_embeddings = yolow_data['embeddings']
    print("yolow_embeddings", yolow_embeddings.shape)
    n_objects = yolow_data['n_objects']
    
    if image_file in dino_emb:
        dino_embeddings = dino_emb[image_file]
        expand_embeddings = dino_embeddings.unsqueeze(0).expand(n_objects, -1)

        concatenated_vector = torch.cat((yolow_embeddings, expand_embeddings), dim=1)
        print(f"  Concatenated shape: {concatenated_vector.shape}")
        
        # Store the result with all metadata
        concatenated_embeddings[image_file] = {
            'combined_embedding': concatenated_vector,
            'yolow_embedding': yolow_embeddings,
            'dino_embedding': dino_embeddings,
            'bboxes': yolow_data['bboxes'],
            'confidences': yolow_data['confidences'],
            'classes': yolow_data['classes'],
            'n_objects': n_objects
        }
    else:
        print(f"  Warning: {image_file} not found in DINOv2 embeddings!")

# Save the concatenated embeddings
output_path = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/combined_embeddings.pt"
torch.save(concatenated_embeddings, output_path)
print(f"\nSaved combined embeddings to: {output_path}")
print(f"Total images processed: {len(concatenated_embeddings)}")