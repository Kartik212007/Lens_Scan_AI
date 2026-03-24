import os
import pickle

# The actual nested path discovered from directory listing
train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Plant_Disease_Dataset", "Plant_Disease_Dataset", "train")

if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"Train directory not found: {train_dir}")

# Use sorted order to be consistent with ImageDataGenerator (alphabetical)
# Filter out .DS_Store and any other hidden files
classes = sorted([d for d in os.listdir(train_dir) if not d.startswith('.')])
class_indices = {cls: idx for idx, cls in enumerate(classes)}

print(f"Found {len(classes)} classes:")
for k, v in class_indices.items():
    print(f"  {v}: {k}")

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_labels.pkl")
with open(output_path, "wb") as f:
    pickle.dump(class_indices, f)

print(f"\n[OK] class_labels.pkl saved to: {output_path}")
