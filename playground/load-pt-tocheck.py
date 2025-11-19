import torch

data1 = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/predicted_converted/0a6ba0bd-d880-4a50-92e0-b1b3df278547.pt")
data2 = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/predicted_converted/0a7cef36-8d0d-4d5b-a2fa-020619292021.pt")

data3 = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/gt_converted/0a6ba0bd-d880-4a50-92e0-b1b3df278547.pt")

dino = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/dino_converted/0a6ba0bd-d880-4a50-92e0-b1b3df278547.pt")

print(data1.keys())

len1 = len(data1["object_ids"])
len2 = len(data2["object_ids"])

total_length = len1 + len2

print(f"Length of first data: {len1}")
print(f"Length of second data: {len2}")
print(f"Sum of lengths: {total_length}")

print("Len of embeddings in first data:", len(data1["embeddings"]))

# print("n_objects, first 10:", data1["n_objects"][:10])

# #sum all the n_objects
# total_objects = sum(data1["n_objects"]) + sum(data2["n_objects"])
# print(f"Total number of objects in both datasets: {total_objects}")
print(len(data1["object_ids"]))
print(len(data1["embeddings"]))
print(len(data1["bboxes"]))


# print the first 5 values of all the keys in data1 except embeddings
# for key in data1.keys():
#     if key != "embeddings":
#         print(f"{key}: {data1[key][:10]}")

# print the first 5 for data3
for key in data3.keys():
    if key == "classes":
        print(f"{key}: {data3[key][:100]}")


print("DINO embeddings", dino.keys())
print(dino["object_ids"][:10])