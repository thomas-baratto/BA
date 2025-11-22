import typing
import torch
import torch.nn.functional as F
import numpy as np
from model import get_activation


def _torch_activation(x: torch.Tensor, act_name: str) -> torch.Tensor:
    # map activation names (same as model.get_activation keys)
    if act_name == "ReLU":
        return F.relu(x)
    if act_name == "LeakyReLU":
        return F.leaky_relu(x)
    if act_name == "ELU":
        return F.elu(x)
    if act_name == "GELU":
        return F.gelu(x)
    # fallback
    return x


class ExtremeLearningMachine:

    def __init__(self,
                 n_hidden: int = 1024,
                 activation: str = "ReLU",
                 alpha: float = 1e-3,
                 include_bias: bool = True,
                 device: typing.Optional[torch.device] = None,
                 random_state: typing.Optional[int] = None):
        self.n_hidden = int(n_hidden)
        self.activation = activation
        self.alpha = float(alpha)
        self.include_bias = bool(include_bias)
        self.device = device if device is not None else torch.device("cpu")
        self.random_state = random_state

        # weights (torch tensors)
        self.W_hidden: typing.Optional[torch.Tensor] = None  # shape (n_features, n_hidden)
        self.b_hidden: typing.Optional[torch.Tensor] = None  # shape (n_hidden,)
        self.W_out: typing.Optional[torch.Tensor] = None     # shape (n_hidden, n_outputs)

    def _init_weights(self, n_features: int):
        gen = torch.Generator(device=self.device)
        if self.random_state is not None:
            gen.manual_seed(int(self.random_state))
        # normal init
        self.W_hidden = torch.randn(n_features, self.n_hidden, dtype=torch.float64, generator=gen, device=self.device)
        if self.include_bias:
            self.b_hidden = torch.randn(self.n_hidden, dtype=torch.float64, generator=gen, device=self.device)
        else:
            self.b_hidden = torch.zeros(self.n_hidden, dtype=torch.float64, device=self.device)

    def _ensure_tensor(self, X: typing.Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X).to(dtype=torch.float64, device=self.device)
        if isinstance(X, torch.Tensor):
            return X.to(dtype=torch.float64, device=self.device)
        raise ValueError("Unsupported input type")

    def _compute_hidden(self, X: torch.Tensor) -> torch.Tensor:
        # X: (N, D); W_hidden: (D, L) -> X @ W_hidden -> (N, L)
        H = X.matmul(self.W_hidden) + self.b_hidden
        H = _torch_activation(H, self.activation)
        return H

    def fit(
        self,
        X: typing.Union[np.ndarray, torch.Tensor],
        y: typing.Union[np.ndarray, torch.Tensor],
        batch_size: typing.Optional[int] = None,
    ):
        """Fit the ELM for regression. X: (N, D), y: (N,) or (N, n_outputs).
        
        Args:
            X: Input features (numpy array or torch tensor)
            y: Target values (numpy array or torch tensor)
            batch_size: Unused placeholder to keep parity with legacy APIs
        """
        X_t = self._ensure_tensor(X)
        N, D = X_t.shape
        if self.W_hidden is None:
            self._init_weights(D)

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

    def decision_function(self, X: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        X_t = self._ensure_tensor(X)
        H = self._compute_hidden(X_t)
        scores = H.matmul(self.W_out)
        return scores.cpu().numpy()

    def predict(self, X: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict target values for regression.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values (1D array if single output, 2D otherwise)
        """
        scores = self.decision_function(X)
        if scores.shape[1] == 1:
            return scores.ravel()
        return scores

    def score(self, X: typing.Union[np.ndarray, torch.Tensor], y: typing.Union[np.ndarray, torch.Tensor]) -> float:
        """Compute R² score for regression.
        
        Args:
            X: Input features
            y: True target values
            
        Returns:
            R² score
        """
        from sklearn.metrics import r2_score
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return r2_score(y_true, y_pred)

    
