import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class EmotionData(Dataset):
    def __init__(
        self,
        root_dir: str = "",
        num_classes_per_episode: int = 5,
        num_support: int = 5,
        num_query: int = 5,
        transform=None,
    ):
        """
        Args:
            root_dir: Path to dataset (e.g., "train", "test", "val")
            num_classes_per_episode: Number of classes sampled per episode.
            num_support: Number of support images per class.
            num_query: Number of query images per class.
            transform: Image transformations.
        """
        self.root_dir = os.path.join(os.getcwd(), "dataset", root_dir)
        self.transform = transform
        self.num_classes_per_episode = num_classes_per_episode
        self.num_support = num_support
        self.num_query = num_query
        # Load class names and image paths
        self.class_to_images = defaultdict(list)
        self.class_to_labels = defaultdict(int)
        label = 0
        for class_name in sorted(os.listdir(self.root_dir)):  # Sort for consistency
            self.class_to_labels = {
                class_name: idx
                for idx, class_name in enumerate(sorted(os.listdir(self.root_dir)))
            }
            label += 1
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_images[class_name] = sorted(
                    [os.path.join(class_path, img) for img in os.listdir(class_path)]
                )

    def __len__(self):
        return 1000  # Arbitrary large number since episodes are sampled dynamically

    def __getitem__(self, _):
        sampled_classes = random.sample(
            list(self.class_to_images.keys()), self.num_classes_per_episode
        )

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        local_transform = ToTensor()
        for label_idx, class_name in enumerate(sampled_classes):
            available_images = self.class_to_images[class_name]

            # Ensure there are enough images
            if len(available_images) < (self.num_support + self.num_query):
                print(
                    f"Warning: Not enough images in class {class_name}. Using all available images."
                )
                images = available_images  # Use all images if too few exist
            else:
                images = random.sample(
                    available_images, self.num_support + self.num_query
                )

            support, query = images[: self.num_support], images[self.num_support :]

            for img_path in support:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                support_images.append(img)
                support_labels.append(label_idx)

            for img_path in query:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                query_images.append(img)
                query_labels.append(label_idx)

        return (
            torch.stack(support_images),
            torch.tensor(support_labels),
            torch.stack(query_images),
            torch.tensor(query_labels),
        )
