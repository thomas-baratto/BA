"""Cross-validated ensemble skip-connection RVFL implementation."""

import typing
import torch
import numpy as np
from sklearn.model_selection import KFold

from .edRVFL_SC import edRVFL_SC

class esc_edRVFL:
    """Ensemble Skip Connection edRVFL (esc-edRVFL).
    
    This model utilizes several edRVFL-RSC models trained on different folds 
    of the training dataset and averages their outputs.
    """

    def __init__(self,
                 n_folds: int = 5,
                 n_ensemble: int = 10,
                 n_layers: int = 3,
                 n_hidden: int = 100,
                 activation: str = "ReLU",
                 alpha: float = 1e-3,
                 include_bias: bool = True,
                 rsc_prob: float = 0.5,
                 gamma: float = 1.0,
                 device: typing.Optional[torch.device] = None,
                 random_state: typing.Optional[int] = None):
        self.n_folds = int(n_folds)
        self.n_ensemble = int(n_ensemble)
        self.n_layers = int(n_layers)
        self.n_hidden = int(n_hidden)
        self.activation = activation
        self.alpha = float(alpha)
        self.include_bias = bool(include_bias)
        self.rsc_prob = float(rsc_prob)
        self.gamma = float(gamma)
        self.device = device if device is not None else torch.device("cpu")
        self.random_state = random_state

        self.models: typing.List[edRVFL_SC] = []

    def fit(self, X: typing.Union[np.ndarray, torch.Tensor], y: typing.Union[np.ndarray, torch.Tensor]):
        """Fit the esc-edRVFL ensemble for regression using K-Fold cross-validation."""
        self.models = []
        
        # Convert X to numpy for KFold splitting if it's a tensor
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X)
            
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = np.asarray(y)

        # Ensure we don't have more folds than samples
        actual_folds = min(self.n_folds, len(X_np))
        if actual_folds < 2:
            # Fallback: train n_folds models on the entire dataset
            kf = None
        else:
            kf = KFold(n_splits=actual_folds, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(X_np))

        for i in range(self.n_folds):
            seed = None
            if self.random_state is not None:
                seed = self.random_state + (i * 100) # Different seed per fold

            model = edRVFL_SC(
                n_ensemble=self.n_ensemble,
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                activation=self.activation,
                alpha=self.alpha,
                include_bias=self.include_bias,
                mode="random", # esc-edRVFL utilizes edRVFL-RSC
                rsc_prob=self.rsc_prob,
                gamma=self.gamma,
                device=self.device,
                random_state=seed
            )
            
            if kf is not None:
                # Use split data (cycle through splits if n_folds > actual_folds)
                train_idx, _ = splits[i % actual_folds]
                X_train = X_np[train_idx]
                y_train = y_np[train_idx]
            else:
                # Fallback
                X_train = X_np
                y_train = y_np
            
            model.fit(X_train, y_train)
            self.models.append(model)
            
        return self

    def predict(self, X: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict target values using the ensemble of ensembles (averaging)."""
        if not self.models:
            raise RuntimeError("The ensemble hasn't been fitted. Call fit() first.")
            
        preds = []
        for model in self.models:
            pred = model.predict(X)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            preds.append(pred)
            
        stacked_preds = np.stack(preds, axis=0) # (n_folds, n_samples, n_outputs)
        avg_preds = np.mean(stacked_preds, axis=0)
        
        if avg_preds.shape[1] == 1:
            return avg_preds.ravel()
        return avg_preds