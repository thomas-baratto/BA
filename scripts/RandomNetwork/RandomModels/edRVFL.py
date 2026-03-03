import typing
import torch
import numpy as np
from .dRVFL import dRVFL

class edRVFL:
    """Ensemble Deep Random Vector Functional Link Network.
    
    This network creates an ensemble of multiple deep RVFLs. Predictions are
    averaged across the ensemble.
    """

    def __init__(self,
                 n_ensemble: int = 10,
                 n_layers: int = 3,
                 n_hidden: int = 100,
                 activation: str = "ReLU",
                 alpha: float = 1e-3,
                 include_bias: bool = True,
                 gamma: float = 1.0,
                 device: typing.Optional[torch.device] = None,
                 random_state: typing.Optional[int] = None):
        self.n_ensemble = int(n_ensemble)
        self.n_layers = int(n_layers)
        self.n_hidden = int(n_hidden)
        self.activation = activation
        self.alpha = float(alpha)
        self.include_bias = bool(include_bias)
        self.gamma = float(gamma)
        self.device = device if device is not None else torch.device("cpu")
        self.random_state = random_state

        self.models: typing.List[dRVFL] = []

    def fit(self, X: typing.Union[np.ndarray, torch.Tensor], y: typing.Union[np.ndarray, torch.Tensor]):
        """Fit the edRVFL ensemble for regression."""
        self.models = []
        
        for i in range(self.n_ensemble):
            # Use a slightly different seed for each model in the ensemble to ensure diversity
            seed = None
            if self.random_state is not None:
                seed = self.random_state + i
                
            model = dRVFL(
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                activation=self.activation,
                alpha=self.alpha,
                include_bias=self.include_bias,
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
            
        # Collect predictions from all models
        preds = []
        for model in self.models:
            pred = model.predict(X)
            # Ensure shape is 2D for consistent stacking
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            preds.append(pred)
            
        # Stack shape: (n_ensemble, n_samples, n_outputs)
        stacked_preds = np.stack(preds, axis=0)
        
        # Average across the ensemble
        avg_preds = np.mean(stacked_preds, axis=0)
        
        if avg_preds.shape[1] == 1:
            return avg_preds.ravel()
        return avg_preds