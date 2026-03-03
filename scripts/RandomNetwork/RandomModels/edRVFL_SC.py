import typing
import torch
import numpy as np
from .utils import torch_activation, ensure_tensor
from .RBF import RBFHiddenLayer

class _dRVFL_SC:
    """Deep Random Vector Functional Link Network with Skip Connections.
    
    Supports "dense" skip connections (all previous layers connected to current)
    or "random" skip connections (randomly selected pathways).
    """
    def __init__(self,
                 n_layers: int = 3,
                 n_hidden: int = 100,
                 activation: str = "ReLU",
                 alpha: float = 1e-3,
                 include_bias: bool = True,
                 mode: str = "dense", # "dense" or "random"
                 rsc_prob: float = 0.5, # Probability for random skip connections
                 gamma: float = 1.0,
                 device: typing.Optional[torch.device] = None,
                 random_state: typing.Optional[int] = None):
        self.n_layers = int(n_layers)
        self.n_hidden = int(n_hidden)
        self.activation = activation
        self.alpha = float(alpha)
        self.include_bias = bool(include_bias)
        self.mode = mode.lower()
        self.rsc_prob = float(rsc_prob)
        self.gamma = float(gamma)
        self.device = device if device is not None else torch.device("cpu")
        self.random_state = random_state

        self.W_hidden: typing.List[torch.Tensor] = []
        self.b_hidden: typing.List[torch.Tensor] = []
        self.rbf_layers: typing.List[RBFHiddenLayer] = []
        self.layer_masks: typing.List[typing.List[bool]] = []
        self.W_out: typing.Optional[torch.Tensor] = None

    def _init_weights(self, X_t: torch.Tensor):
        n_features = X_t.shape[1]
        n_samples = X_t.shape[0]
        gen = torch.Generator(device=self.device)
        if self.random_state is not None:
            gen.manual_seed(int(self.random_state))
            
        self.W_hidden = []
        self.b_hidden = []
        self.rbf_layers = []
        self.layer_masks = []
        
        H_list = [X_t]
        
        for i in range(self.n_layers):
            if self.mode == "dense":
                mask = [True] * (i + 1)
            else: # "random"
                mask_tensor = torch.rand(i + 1, generator=gen, device=self.device) < self.rsc_prob
                mask = mask_tensor.cpu().tolist()
                if not any(mask):
                    mask[i] = True # connect to the immediately previous layer (or X if i=0)
            
            self.layer_masks.append(mask)
            
            gather_inputs = []
            for j, use_layer in enumerate(mask):
                if use_layer:
                    gather_inputs.append(H_list[j])
            
            H_in = torch.cat(gather_inputs, dim=1)
            input_dim = H_in.shape[1]
                
            if self.activation.lower() == "rbf":
                rbf_layer = RBFHiddenLayer(n_hidden=self.n_hidden, gamma=self.gamma, in_features=input_dim).to(self.device).to(torch.float64)
                # Sample centers directly from the combined input space
                indices = torch.randint(0, n_samples, (self.n_hidden,), generator=gen, device=self.device)
                rbf_layer.centers.data = H_in[indices].clone()
                self.rbf_layers.append(rbf_layer)
                H_curr = rbf_layer(H_in)
            else:
                W = torch.randn(input_dim, self.n_hidden, dtype=torch.float64, generator=gen, device=self.device)
                if self.include_bias:
                    b = torch.randn(self.n_hidden, dtype=torch.float64, generator=gen, device=self.device)
                else:
                    b = torch.zeros(self.n_hidden, dtype=torch.float64, device=self.device)
                    
                self.W_hidden.append(W)
                self.b_hidden.append(b)
                H_curr = H_in.matmul(W) + b
                H_curr = torch_activation(H_curr, self.activation)
                
            H_list.append(H_curr)



    def _compute_features(self, X: torch.Tensor) -> torch.Tensor:
        """Propagate through hidden layers and return concatenated features."""
        H_list = [X]
        
        for i in range(self.n_layers):
            gather_inputs = []
            for j, use_layer in enumerate(self.layer_masks[i]):
                if use_layer:
                    gather_inputs.append(H_list[j])
            
            H_in = torch.cat(gather_inputs, dim=1)
            
            if self.activation.lower() == "rbf":
                H_curr = self.rbf_layers[i](H_in)
            else:
                W = self.W_hidden[i]
                b = self.b_hidden[i]
                H_curr = H_in.matmul(W) + b
                H_curr = torch_activation(H_curr, self.activation)
                
            H_list.append(H_curr)
            
        return torch.cat(H_list, dim=1)

    def fit(self, X: typing.Union[np.ndarray, torch.Tensor], y: typing.Union[np.ndarray, torch.Tensor]):
        """Fit the dRVFL with SC for regression."""
        X_t = ensure_tensor(X, self.device)
        N, D = X_t.shape
        if not self.W_hidden and not self.rbf_layers:
            self._init_weights(X_t)

        if isinstance(y, np.ndarray):
            Y_t = torch.from_numpy(y).to(dtype=torch.float64, device=self.device)
        else:
            Y_t = y.to(dtype=torch.float64, device=self.device)
        if Y_t.dim() == 1:
            Y_t = Y_t.unsqueeze(1)

        H_final = self._compute_features(X_t)
        
        L = H_final.shape[1]
        A = H_final.T.matmul(H_final) + self.alpha * torch.eye(L, dtype=torch.float64, device=self.device)
        B = H_final.T.matmul(Y_t)
        self.W_out = torch.linalg.solve(A, B)

        return self

    def predict(self, X: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        X_t = ensure_tensor(X, self.device)
        H_final = self._compute_features(X_t)
        scores = H_final.matmul(self.W_out)
        
        scores_np = scores.cpu().numpy()
        if scores_np.shape[1] == 1:
            return scores_np.ravel()
        return scores_np


class edRVFL_SC:
    """Ensemble Deep Random Vector Functional Link Network with Skip Connections.
    
    Supports "dense" skip connections (edRVFL-SC) or 
    "random" skip connections (edRVFL-RSC) depending on the `mode` parameter.
    """

    def __init__(self,
                 n_ensemble: int = 10,
                 n_layers: int = 3,
                 n_hidden: int = 100,
                 activation: str = "ReLU",
                 alpha: float = 1e-3,
                 include_bias: bool = True,
                 mode: str = "dense", # "dense" or "random"
                 rsc_prob: float = 0.5, # Probability for random skip connections
                 gamma: float = 1.0,
                 device: typing.Optional[torch.device] = None,
                 random_state: typing.Optional[int] = None):
        self.n_ensemble = int(n_ensemble)
        self.n_layers = int(n_layers)
        self.n_hidden = int(n_hidden)
        self.activation = activation
        self.alpha = float(alpha)
        self.include_bias = bool(include_bias)
        self.mode = mode.lower()
        self.rsc_prob = float(rsc_prob)
        self.gamma = float(gamma)
        self.device = device if device is not None else torch.device("cpu")
        self.random_state = random_state

        self.models: typing.List[_dRVFL_SC] = []

    def fit(self, X: typing.Union[np.ndarray, torch.Tensor], y: typing.Union[np.ndarray, torch.Tensor]):
        """Fit the edRVFL_SC ensemble for regression."""
        self.models = []
        
        for i in range(self.n_ensemble):
            seed = None
            if self.random_state is not None:
                seed = self.random_state + i
                
            model = _dRVFL_SC(
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                activation=self.activation,
                alpha=self.alpha,
                include_bias=self.include_bias,
                mode=self.mode,
                rsc_prob=self.rsc_prob,
                gamma=self.gamma,
                device=self.device,
                random_state=seed
            )
            model.fit(X, y)
            self.models.append(model)
            
        return self

    def predict(self, X: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict target values using the ensemble (averaging)."""
        if not self.models:
            raise RuntimeError("The ensemble hasn't been fitted. Call fit() first.")
            
        preds = []
        for model in self.models:
            pred = model.predict(X)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            preds.append(pred)
            
        stacked_preds = np.stack(preds, axis=0)
        avg_preds = np.mean(stacked_preds, axis=0)
        
        if avg_preds.shape[1] == 1:
            return avg_preds.ravel()
        return avg_preds