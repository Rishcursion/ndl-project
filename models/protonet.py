import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_extractor import FeatureExtractor
from torch import Tensor

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class ProtoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(model_name="resnet34")
        self.feature_extractor.to(device)

    def forward(
        self, support_images: Tensor, support_labels: Tensor, query_images: Tensor
    ):
        support_labels = support_labels.squeeze(0)
        support_images = support_images.view(-1, *support_images.shape[2:])  # (batch_size * num_samples, C, H, W)
        query_images = query_images.view(-1, *query_images.shape[2:])  # (batch_size * num_samples, C, H, W)
        # Extract Embeddings Of Support And Query Images
        support_embeddings = self.feature_extractor(support_images)
        query_embeddings = self.feature_extractor(query_images)

        # Prototypes Of Each Class Is Obtained By Getting The Mean Vector Of The Embedded Points Belonging To The Class, Similar To Calculation Of Centroids In KNN Algorithm.
        unique_classes = torch.unique(support_labels)
        class_protos = torch.stack(
            [
                support_embeddings[support_labels == cls].mean(0)
                for cls in unique_classes
            ]
        )

        # Calculate Euclidean Distance Between Query Images And The Prototypes Of Each Class
        proto_dists = torch.cdist(query_embeddings, class_protos)

        # Get Class Probabilities For Each Query Image Using Distances
        log_p_y = F.log_softmax(-proto_dists, dim=1)
        # Propogate Class Probabilities Through The Network
        return log_p_y
