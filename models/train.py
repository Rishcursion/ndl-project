import time
from collections import defaultdict
from typing import Literal

import torch
import torch.nn.functional as F
import torch.optim as optim
from scripts.emotionData import EmotionData
from torch.utils.data import DataLoader

from .protonet import ProtoNet

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initializing The Prototypical Network
proto_net = ProtoNet()


def initialize_data(subset: Literal["val", "train"], batch_size: int = 1):
    data = EmotionData(
        root_dir=subset, transform=proto_net.feature_extractor.transforms
    )
    return DataLoader(data, shuffle=True, batch_size=batch_size)


def train_model(
    train_data: DataLoader,
    val_data: DataLoader,
    optimizer=optim.Adam(proto_net.parameters(), lr=1e-3),
    num_iters: int = 10,
    verbose: bool = True,
):
    train_metrics = defaultdict(dict)
    val_metrics = defaultdict(dict)
    print("=" * 40)
    print("Starting Prototypical Network Learning.")
    print("=" * 40)
    start_time = time.time()
    for epoch in range(num_iters):
        epoch_time = time.time()
        if verbose:
            print(
                f"Starting Epoch {epoch+1}\nProgress: {(epoch+1)*100/num_iters:.2f}%\n"
            )
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        print("Training Phase...\n")
        for support_images, support_labels, query_images, query_labels in train_data:
            support_images, support_labels, query_images, query_labels = (
                support_images.to(device),
                support_labels.to(device),
                query_images.to(device),
                query_labels.to(device),
            )
            query_labels = query_labels.squeeze(0)
            log_p_y = proto_net(support_images, support_labels, query_images)
            query_labels = torch.clamp(query_labels, max=log_p_y.shape[1] - 1)
            loss = F.nll_loss(log_p_y, query_labels)
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Model Performance Tracking
            epoch_loss += loss.item()
            predictions = torch.argmax(log_p_y, dim=1)
            train_correct += predictions == query_labels
            train_total += query_labels.size(0)
            train_time = round(time.time() - epoch_time, 2)
            train_acc = train_correct * 100 / train_total
            train_metrics[epoch] = {"Accuracy": train_acc, "Loss": epoch_loss}
        if verbose:
            print(
                f"""
            Finished Epoch Training 
            Loss: {epoch_loss/len(train_data)}
            Accuracy: {train_acc.mean().item():.2f} %
            Time Taken: {train_time}
            """
            )

        print("Validation Phase...\n")
        proto_net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_acc = 0
        with torch.no_grad():
            for support_images, support_labels, query_images, query_labels in val_data:
                support_images, support_labels, query_images, query_labels = (
                    support_images.to(device),
                    support_labels.to(device),
                    query_images.to(device),
                    query_labels.to(device),
                )
                query_labels = query_labels.squeeze(0)
                log_p_y = proto_net(support_images, support_labels, query_images)
                query_labels = torch.clamp(query_labels, max=log_p_y.shape[1] - 1)
                loss = F.nll_loss(log_p_y, query_labels)
                val_loss += loss.item()
                predictions = torch.argmax(log_p_y, dim=1)
                val_correct += (predictions == query_labels).sum().item()
                val_total += len(query_labels)
                val_acc = val_correct * 100 / val_total
                val_metrics[epoch] = {"Accuracy": val_acc, "Loss": val_loss}
        if verbose:
            print(
                f"""
            Finished Epoch Validation
            Loss: {epoch_loss/len(val_data)}
            Accuracy: {val_acc:.2f} %
            Time Taken: {round(train_time - time.time(), 2)} 
            """
            )
    print(f"Model Training Finished In {round(time.time()-start_time,2)}")
    return train_metrics, val_metrics


if __name__ == "__main__":
    train_data = initialize_data("train")
    val_data = initialize_data("val")
    train_metrics, val_metrics = train_model(train_data, val_data)
    print("Train Metrics", train_metrics)
    print("Validation Metrics", val_metrics)
