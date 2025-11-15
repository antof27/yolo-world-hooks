import torch 

data = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/yolow_local_embeddings.pt")

print(f"Type: {type(data)}")
print(f"Keys: {data.keys() if isinstance(data, dict) else 'Not a dictionary'}")

for key in list(data.keys())[:5]:  # Print first 5 entries
    print(f"{key}: {data[key]}")


