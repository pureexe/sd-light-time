import json 

with open("filenames.json", "r") as f:
    dataset_listing = json.load(f)

dataset_listing = set(dataset_listing)
print(f"Number of files in dataset listing: {len(dataset_listing)}")