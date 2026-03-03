import torch
import torch.nn as nn

class RBFHiddenLayer(nn.Module):
    def __init__(self, n_hidden: int, gamma: float = 1.0, in_features: int = 1):
        super(RBFHiddenLayer, self).__init__()
        self.gamma = gamma
        self.in_features = in_features

        self.centers = nn.Parameter(torch.randn(n_hidden, in_features))
        self.centers.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(x, self.centers)
        return torch.exp(-self.gamma * distances.pow(2) / self.in_features)
