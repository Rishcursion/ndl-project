import os
import shutil
import random
from collections import defaultdict

# Define dataset paths
SOURCE_DIR = "../dataset"  # Modify this to match your dataset folder
DEST_DIR = "../dataset/balanced_dataset"  # Where the structured dataset will be saved

# Define split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Number of unseen classes in the test set
NUM_TEST_CLASSES = 2  

# Create target directories
for subset in ["train", "val", "test"]:
    os.makedirs(os.path.join(DEST_DIR, subset), exist_ok=True)

# Read class distribution
class_to_images = defaultdict(list)
for class_name in sorted(os.listdir(SOURCE_DIR)):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if os.path.isdir(class_path):
        images = sorted(os.listdir(class_path))
        class_to_images[class_name] = [os.path.join(class_path, img) for img in images]

# Select exactly 2 unseen classes for the test set
all_classes = list(class_to_images.keys())
random.shuffle(all_classes)
test_classes = set(all_classes[:NUM_TEST_CLASSES])  # Select 2 random unseen classes

# Create train/val/test splits
for class_name, images in class_to_images.items():
    random.shuffle(images)  # Shuffle to avoid biases

    num_images = len(images)
    if num_images < 3:  # Avoid classes with too few samples
        print(f"Skipping class {class_name} (too few images: {num_images})")
        continue

    if class_name in test_classes:
        # Assign all images of these 2 classes to the test set
        subset = "test"
        for img_path in images:
            dest_folder = os.path.join(DEST_DIR, subset, class_name)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(img_path, os.path.join(dest_folder, os.path.basename(img_path)))
    else:
        # Split into train, val, and test
        train_cutoff = int(num_images * TRAIN_RATIO)
        val_cutoff = train_cutoff + int(num_images * VAL_RATIO)

        for i, img_path in enumerate(images):
            if i < train_cutoff:
                subset = "train"
            elif i < val_cutoff:
                subset = "val"
            else:
                subset = "test"

            # Move the image
            dest_folder = os.path.join(DEST_DIR, subset, class_name)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(img_path, os.path.join(dest_folder, os.path.basename(img_path)))

# Summary
print("✅ Dataset restructuring complete!")
print(f"Train classes: {len(set(class_to_images.keys()) - test_classes)}")
print(f"Test classes (unseen): {test_classes}")
