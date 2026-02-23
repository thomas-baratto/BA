import typing
import torch
import numpy as np
from .utils import torch_activation, ensure_tensor

class dRVFL:
    """Deep Random Vector Functional Link Network.
    
    This network stacks multiple random hidden layers. The output layer receives
    features not only from the final hidden layer, but concatenated from the
    original input and all hidden layers.
    """

    def __init__(self,
                 n_layers: int = 3,
                 n_hidden: int = 100,
                 activation: str = "ReLU",
                 alpha: float = 1e-3,
                 include_bias: bool = True,
                 device: typing.Optional[torch.device] = None,
                 random_state: typing.Optional[int] = None):
        self.n_layers = int(n_layers)
        self.n_hidden = int(n_hidden)
        self.activation = activation
        self.alpha = float(alpha)
        self.include_bias = bool(include_bias)
        self.device = device if device is not None else torch.device("cpu")
        self.random_state = random_state

        self.W_hidden: typing.List[torch.Tensor] = []
        self.b_hidden: typing.List[torch.Tensor] = []
        self.W_out: typing.Optional[torch.Tensor] = None

    def _init_weights(self, n_features: int):
        gen = torch.Generator(device=self.device)
        if self.random_state is not None:
            gen.manual_seed(int(self.random_state))
            
        self.W_hidden = []
        self.b_hidden = []
        
        input_dim = n_features
        for _ in range(self.n_layers):
            W = torch.randn(input_dim, self.n_hidden, dtype=torch.float64, generator=gen, device=self.device)
            if self.include_bias:
                b = torch.randn(self.n_hidden, dtype=torch.float64, generator=gen, device=self.device)
            else:
                b = torch.zeros(self.n_hidden, dtype=torch.float64, device=self.device)
                
            self.W_hidden.append(W)
            self.b_hidden.append(b)
            input_dim = self.n_hidden



    def _compute_features(self, X: torch.Tensor) -> torch.Tensor:
        """Propagate through hidden layers and return concatenated features."""
        H_list = [X]
        H_curr = X
        
        for W, b in zip(self.W_hidden, self.b_hidden):
            H_curr = H_curr.matmul(W) + b
            H_curr = torch_activation(H_curr, self.activation)
            H_list.append(H_curr)
            
        return torch.cat(H_list, dim=1)

    def fit(self, X: typing.Union[np.ndarray, torch.Tensor], y: typing.Union[np.ndarray, torch.Tensor]):
        """Fit the dRVFL for regression."""
        X_t = ensure_tensor(X, self.device)
        N, D = X_t.shape
        if not self.W_hidden:
            self._init_weights(D)

        # Prepare Y
        if isinstance(y, np.ndarray):
            Y_t = torch.from_numpy(y).to(dtype=torch.float64, device=self.device)
        else:
            Y_t = y.to(dtype=torch.float64, device=self.device)
        if Y_t.dim() == 1:
            Y_t = Y_t.unsqueeze(1)

        # Compute combined feature matrix
        H_final = self._compute_features(X_t)
        
        # Solve: W_out = (H_final^T H_final + alpha I)^(-1) H_final^T Y
        L = H_final.shape[1]
        A = H_final.T.matmul(H_final) + self.alpha * torch.eye(L, dtype=torch.float64, device=self.device)
        B = H_final.T.matmul(Y_t)
        self.W_out = torch.linalg.solve(A, B)

        return self

    def predict(self, X: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict target values for regression."""
        X_t = ensure_tensor(X, self.device)
        H_final = self._compute_features(X_t)
        scores = H_final.matmul(self.W_out)
        
        scores_np = scores.cpu().numpy()
        if scores_np.shape[1] == 1:
            return scores_np.ravel()
        return scores_np