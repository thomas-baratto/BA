import typing
import torch
import numpy as np
from .utils import torch_activation, ensure_tensor

class _ResidualBlock:
    """A single residual block for SResdRVFL.
    
    This block takes the input X and fits to the residual error (y - previous_predictions).
    It optionally uses a direct link (connecting X directly to the output weights along with 
    the final hidden layer).
    """
    def __init__(self,
                 n_layers: int = 1,
                 n_hidden: int = 100,
                 activation: str = "ReLU",
                 alpha: float = 1e-3,
                 include_bias: bool = True,
                 direct_link: bool = True,
                 device: typing.Optional[torch.device] = None,
                 random_state: typing.Optional[int] = None):
        self.n_layers = int(n_layers)
        self.n_hidden = int(n_hidden)
        self.activation = activation
        self.alpha = float(alpha)
        self.include_bias = bool(include_bias)
        self.direct_link = bool(direct_link)
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
        H_curr = X
        
        for W, b in zip(self.W_hidden, self.b_hidden):
            H_curr = H_curr.matmul(W) + b
            H_curr = torch_activation(H_curr, self.activation)
            
        # Asymmetric direct links: The input layer is only connected directly to the output 
        # alongside the final hidden layer activations (not densely to all hidden layers)
        if self.direct_link:
            return torch.cat([X, H_curr], dim=1)
        else:
            return H_curr

    def fit(self, X: torch.Tensor, y_res: torch.Tensor):
        N, D = X.shape
        if not self.W_hidden:
            self._init_weights(D)

        H_final = self._compute_features(X)
        
        L = H_final.shape[1]
        A = H_final.T.matmul(H_final) + self.alpha * torch.eye(L, dtype=torch.float64, device=self.device)
        B = H_final.T.matmul(y_res)
        self.W_out = torch.linalg.solve(A, B)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        H_final = self._compute_features(X)
        return H_final.matmul(self.W_out)


class SResdRVFL:
    """Stacked Residual Deep Random Vector Functional Link Network.
    
    This ensemble architecture trains successive blocks to predict the residual error
    (the difference between the target and the sum of predictions from previous blocks)
    rather than predicting the target directly. This mitigates stagnant fitting effects
    in deep randomized networks.
    """

    def __init__(self,
                 n_blocks: int = 5,
                 n_layers_per_block: int = 1,
                 n_hidden: int = 100,
                 activation: str = "ReLU",
                 alpha: float = 1e-3,
                 include_bias: bool = True,
                 direct_link: bool = True,
                 device: typing.Optional[torch.device] = None,
                 random_state: typing.Optional[int] = None):
        self.n_blocks = int(n_blocks)
        self.n_layers_per_block = int(n_layers_per_block)
        self.n_hidden = int(n_hidden)
        self.activation = activation
        self.alpha = float(alpha)
        self.include_bias = bool(include_bias)
        self.direct_link = bool(direct_link)
        self.device = device if device is not None else torch.device("cpu")
        self.random_state = random_state

        self.blocks: typing.List[_ResidualBlock] = []



    def fit(self, X: typing.Union[np.ndarray, torch.Tensor], y: typing.Union[np.ndarray, torch.Tensor]):
        """Fit the SResdRVFL network progressively learning residual errors."""
        X_t = ensure_tensor(X, self.device)
        
        if isinstance(y, np.ndarray):
            Y_t = torch.from_numpy(y).to(dtype=torch.float64, device=self.device)
        else:
            Y_t = y.to(dtype=torch.float64, device=self.device)
        if Y_t.dim() == 1:
            Y_t = Y_t.unsqueeze(1)
            
        self.blocks = []
        
        # The first target is the actual y
        current_residual = Y_t
        
        for i in range(self.n_blocks):
            seed = None
            if self.random_state is not None:
                seed = self.random_state + i
                
            block = _ResidualBlock(
                n_layers=self.n_layers_per_block,
                n_hidden=self.n_hidden,
                activation=self.activation,
                alpha=self.alpha,
                include_bias=self.include_bias,
                direct_link=self.direct_link,
                device=self.device,
                random_state=seed
            )
            
            # Fit the block to the current residual error
            block.fit(X_t, current_residual)
            
            # Predict the residual using the newly trained block
            block_pred = block.predict(X_t)
            
            # Update the residual: what is still left to learn?
            current_residual = current_residual - block_pred
            
            self.blocks.append(block)
            
        return self

    def predict(self, X: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict target values by summing the predictions from all residual blocks."""
        if not self.blocks:
            raise RuntimeError("The model hasn't been fitted. Call fit() first.")
            
        X_t = ensure_tensor(X, self.device)
        
        # Accumulate predictions from all blocks
        total_pred = torch.zeros(X_t.shape[0], self.blocks[0].W_out.shape[1], 
                                 dtype=torch.float64, device=self.device)
                                 
        for block in self.blocks:
            total_pred += block.predict(X_t)
            
        scores_np = total_pred.cpu().numpy()
        if scores_np.shape[1] == 1:
            return scores_np.ravel()
        return scores_np