import typing
import torch
import torch.nn.functional as F
import numpy as np

def torch_activation(x: torch.Tensor, act_name: str) -> torch.Tensor:
    """Helper to apply the specified activation function to a tensor."""
    if act_name == "ReLU":
        return F.relu(x)
    if act_name == "LeakyReLU":
        return F.leaky_relu(x)
    if act_name == "ELU":
        return F.elu(x)
    if act_name == "GELU":
        return F.gelu(x)
    return x

def ensure_tensor(X: typing.Union[np.ndarray, torch.Tensor], device: typing.Optional[torch.device] = None) -> torch.Tensor:
    """Ensure that the input is a float64 PyTorch tensor on the correct device."""
    if device is None:
        device = torch.device("cpu")
        
    if isinstance(X, np.ndarray):
        return torch.from_numpy(X).to(dtype=torch.float64, device=device)
    if isinstance(X, torch.Tensor):
        return X.to(dtype=torch.float64, device=device)
        
    raise ValueError("Unsupported input type. Must be numpy array or torch tensor.")
