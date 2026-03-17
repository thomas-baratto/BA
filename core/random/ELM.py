"""Extreme Learning Machine implementation for tabular regression."""

import typing
import torch
import numpy as np
from .utils import torch_activation, ensure_tensor
from .RBF import RBFHiddenLayer

class ELM:
    """Extreme Learning Machine, with adjustable hidden layer size and amount."""

    def __init__(self,
                 n_hidden: int = 1024,
                 activation: str = "ReLU",
                 alpha: float = 1e-3,
                 include_bias: bool = True,
                 gamma: float = 1.0,
                 device: typing.Optional[torch.device] = None,
                 random_state: typing.Optional[int] = None):
        self.n_hidden = int(n_hidden)
        self.activation = activation
        self.alpha = float(alpha)
        self.include_bias = bool(include_bias)
        self.gamma = float(gamma)
        self.device = device if device is not None else torch.device("cpu")
        self.random_state = random_state
        self.rbf_layer: typing.Optional[RBFHiddenLayer] = None

        # weights (torch tensors)
        self.W_hidden: typing.Optional[torch.Tensor] = None  # shape (n_features, n_hidden)
        self.b_hidden: typing.Optional[torch.Tensor] = None  # shape (n_hidden,)
        self.W_out: typing.Optional[torch.Tensor] = None     # shape (n_hidden, n_outputs)

    def _init_weights(self, X_t: torch.Tensor):
        n_features = X_t.shape[1]
        n_samples = X_t.shape[0]
        gen = torch.Generator(device=self.device)
        if self.random_state is not None:
            gen.manual_seed(int(self.random_state))
        
        if self.activation.lower() == "rbf":
            self.rbf_layer = RBFHiddenLayer(n_hidden=self.n_hidden, gamma=self.gamma, in_features=n_features).to(self.device).to(torch.float64)
            # Sample centers directly from the training data distribution
            indices = torch.randint(0, n_samples, (self.n_hidden,), generator=gen, device=self.device)
            self.rbf_layer.centers.data = X_t[indices].clone()
        else:
            self.W_hidden = torch.randn(n_features, self.n_hidden, dtype=torch.float64, generator=gen, device=self.device)
            if self.include_bias:
                self.b_hidden = torch.randn(self.n_hidden, dtype=torch.float64, generator=gen, device=self.device)
            else:
                self.b_hidden = torch.zeros(self.n_hidden, dtype=torch.float64, device=self.device)



    def _compute_hidden(self, X: torch.Tensor) -> torch.Tensor:
        if self.activation.lower() == "rbf":
            return self.rbf_layer(X)
            
        # X: (N, D); W_hidden: (D, L) -> X @ W_hidden -> (N, L)
        H = X.matmul(self.W_hidden) + self.b_hidden
        H = torch_activation(H, self.activation)
        return H

    def fit(self, X: typing.Union[np.ndarray, torch.Tensor], y: typing.Union[np.ndarray, torch.Tensor]):
        """Fit the ELM for regression. X: (N, D), y: (N,) or (N, n_outputs)."""
        X_t = ensure_tensor(X, self.device)
        N, D = X_t.shape
        if self.W_hidden is None and self.rbf_layer is None:
            self._init_weights(X_t)

        # Prepare Y
        if isinstance(y, np.ndarray):
            Y_t = torch.from_numpy(y).to(dtype=torch.float64, device=self.device)
        else:
            Y_t = y.to(dtype=torch.float64, device=self.device)
        if Y_t.dim() == 1:
            Y_t = Y_t.unsqueeze(1)

        # Compute hidden layer output
        H = self._compute_hidden(X_t)  # (N, n_hidden)
        
        # Solve: W_out = (H^T H + alpha I)^(-1) H^T Y
        L = self.n_hidden
        A = H.T.matmul(H) + self.alpha * torch.eye(L, dtype=torch.float64, device=self.device)
        B = H.T.matmul(Y_t)
        self.W_out = torch.linalg.solve(A, B)  # (n_hidden, n_outputs)

        return self

    def predict(self, X: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict target values for regression."""
        X_t = ensure_tensor(X, self.device)
        H = self._compute_hidden(X_t)
        scores = H.matmul(self.W_out)
        
        scores_np = scores.cpu().numpy()
        if scores_np.shape[1] == 1:
            return scores_np.ravel()
        return scores_np
