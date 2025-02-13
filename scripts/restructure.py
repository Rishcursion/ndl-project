import os
import shutil
import random

# Define dataset directories
DATASET_DIR = "dataset"  # Modify based on dataset location
OUTPUT_DIR = "output"  # Output directory

# Define splits
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10  # Test will contain seen + unseen emotions

# Create output directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Get all emotion categories
emotions = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

# Select emotions that will be *completely unseen* during training
UNSEEN_COUNT = max(1, len(emotions) // 4)  # 25% of classes as unseen
random.shuffle(emotions)

unseen_emotions = emotions[:UNSEEN_COUNT]  # These will only be in test
seen_emotions = emotions[UNSEEN_COUNT:]  # These will be split across train/val/test

print(f"Unseen emotions (test only): {unseen_emotions}")

# Move unseen emotions directly to test set
for emotion in unseen_emotions:
    src_dir = os.path.join(DATASET_DIR, emotion)
    dest_dir = os.path.join(OUTPUT_DIR, "test", emotion)
    shutil.move(src_dir, dest_dir)  # Move entire folder

# Process seen emotions and split their images
for emotion in seen_emotions:
    src_dir = os.path.join(DATASET_DIR, emotion)
    images = sorted(os.listdir(src_dir))
    random.shuffle(images)

    # Compute split sizes
    total_images = len(images)
    train_count = int(total_images * TRAIN_SPLIT)
    val_count = int(total_images * VAL_SPLIT)
    test_count = total_images - train_count - val_count

    # Assign images to splits
    split_mapping = {
        "train": images[:train_count],
        "val": images[train_count:train_count + val_count],
        "test": images[train_count + val_count:]
    }

    # Move images into respective folders
    for split, split_images in split_mapping.items():
        dest_dir = os.path.join(OUTPUT_DIR, split, emotion)
        os.makedirs(dest_dir, exist_ok=True)
        for img in split_images:
            shutil.move(os.path.join(src_dir, img), os.path.join(dest_dir, img))

    # Remove empty source directory
    os.rmdir(src_dir)

print("Dataset restructuring completed!")
