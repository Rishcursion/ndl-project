import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, ResNet50_Weights, resnet34, resnet50

# Get device Capabilities
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "resnet34") -> None:
        super().__init__()
        if model_name == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            for layer in self.model.parameters():
                layer.requires_grad = False
            self.transforms = ResNet50_Weights.DEFAULT.transforms()
            del self.model.fc
        else:
            self.model = resnet34(ResNet34_Weights.DEFAULT)
            for layer in self.model.parameters():
                layer.requires_grad = False
            self.transforms = ResNet34_Weights.DEFAULT.transforms()
        # Utilizing CUDA if available on device else utilizes CPU
        setattr(self.model,"fc", nn.Identity())
        self.model.to(device=device)

    def forward(self, x):
        x = self.model(x)
        return x
