#!/usr/bin/env python3
"""
Multi-Layer ELM Implementation and Grid Search.
Supports multiple hidden layers with fixed random weights (Deep ELM / Stacked RVFL).
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import json
import time
import itertools
import logging
from datetime import datetime
from typing import List, Union, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import data loader from existing codebase
from core.data_loader import load_data

# Feature columns
FEATURE_COLUMN_NAMES = [
    "Flow_well", "Temp_diff", "kW_well", "Hydr_gradient", "Hydr_conductivity",
    "Aqu_thickness", "Long_dispersivity", "Trans_dispersivity", "Isotherm"
]

def _torch_activation(x: torch.Tensor, act_name: str) -> torch.Tensor:
    if act_name == "ReLU":
        return F.relu(x)
    if act_name == "LeakyReLU":
        return F.leaky_relu(x)
    if act_name == "ELU":
        return F.elu(x)
    if act_name == "GELU":
        return F.gelu(x)
    return x

class MultiLayerELM:
    """
    Extreme Learning Machine with multiple hidden layers.
    All hidden layers have fixed random weights.
    Only the output weights are learned.
    """
    def __init__(self,
                 hidden_layer_sizes: List[int],
                 activation: str = "ReLU",
                 alpha: float = 1e-3,
                 include_bias: bool = True,
                 device: Optional[torch.device] = None,
                 random_state: Optional[int] = None):
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = float(alpha)
        self.include_bias = bool(include_bias)
        self.device = device if device is not None else torch.device("cpu")
        self.random_state = random_state

        # List of weight tensors for each layer
        self.W_layers = [] # List[torch.Tensor]
        self.b_layers = [] # List[torch.Tensor]
        
        # Output weights
        self.W_out: Optional[torch.Tensor] = None

    def _init_weights(self, n_features: int):
        gen = torch.Generator(device=self.device)
        if self.random_state is not None:
            gen.manual_seed(int(self.random_state))
            
        self.W_layers = []
        self.b_layers = []
        
        input_dim = n_features
        
        for hidden_size in self.hidden_layer_sizes:
            # Initialize weights for this layer: input_dim -> hidden_size
            W = torch.randn(input_dim, hidden_size, dtype=torch.float64, generator=gen, device=self.device)
            self.W_layers.append(W)
            
            if self.include_bias:
                b = torch.randn(hidden_size, dtype=torch.float64, generator=gen, device=self.device)
            else:
                b = torch.zeros(hidden_size, dtype=torch.float64, device=self.device)
            self.b_layers.append(b)
            
            # Next layer input is this layer's output
            input_dim = hidden_size

    def _ensure_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X).to(dtype=torch.float64, device=self.device)
        if isinstance(X, torch.Tensor):
            return X.to(dtype=torch.float64, device=self.device)
        raise ValueError("Unsupported input type")

    def _compute_hidden(self, X: torch.Tensor) -> torch.Tensor:
        H = X
        # Propagate through all hidden layers
        for W, b in zip(self.W_layers, self.b_layers):
            H = H.matmul(W) + b
            H = _torch_activation(H, self.activation)
        return H

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        X_t = self._ensure_tensor(X)
        N, D = X_t.shape
        
        if not self.W_layers:
            self._init_weights(D)

        # Prepare Y
        if isinstance(y, np.ndarray):
            Y_t = torch.from_numpy(y).to(dtype=torch.float64, device=self.device)
        else:
            Y_t = y.to(dtype=torch.float64, device=self.device)
        if Y_t.dim() == 1:
            Y_t = Y_t.unsqueeze(1)

        # Compute output of the last hidden layer
        H = self._compute_hidden(X_t)
        
        # H is (N, last_hidden_size)
        L = self.hidden_layer_sizes[-1]
        
        # Ridge Regression: W_out = (H^T H + alpha I)^(-1) H^T Y
        A = H.T.matmul(H) + self.alpha * torch.eye(L, dtype=torch.float64, device=self.device)
        B = H.T.matmul(Y_t)
        self.W_out = torch.linalg.solve(A, B)

        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        X_t = self._ensure_tensor(X)
        H = self._compute_hidden(X_t)
        scores = H.matmul(self.W_out)
        
        scores_np = scores.cpu().numpy()
        if scores_np.shape[1] == 1:
            return scores_np.ravel()
        return scores_np


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Layer ELM Grid Search")
    parser.add_argument('--target', type=str, default='all', 
                        choices=['Area', 'Iso_distance', 'Iso_width', 'all'])
    parser.add_argument('--feature-scaler', type=str, default='standard')
    parser.add_argument('--label-scaler', type=str, default='standard')
    parser.add_argument('--csv-file', type=str, default='./data/Clean_Results_Isotherm.csv')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--random-state', type=int, default=42, help="Seed for reproducibility")
    parser.add_argument('--no-cuda', action='store_true')
    return parser.parse_args()

def train_and_evaluate(hidden_layers, activation, alpha, X_train, X_val, y_train, y_val, device, random_state):
    elm = MultiLayerELM(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        alpha=alpha,
        include_bias=True,
        device=device,
        random_state=random_state
    )
    
    start_time = time.time()
    elm.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_val_pred = elm.predict(X_val)
    if y_val_pred.ndim == 1:
        y_val_pred = y_val_pred.reshape(-1, 1)
        
    # Check shape of y_val
    if isinstance(y_val, np.ndarray) and y_val.ndim == 1:
        y_val_r = y_val.reshape(-1, 1)
    else:
        y_val_r = y_val

    val_rmse = np.sqrt(mean_squared_error(y_val_r, y_val_pred))
    val_mae = mean_absolute_error(y_val_r, y_val_pred)
    val_r2 = r2_score(y_val_r, y_val_pred)
    
    return {
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'train_time': train_time
    }

def main():
    args = parse_args()
    
    if args.target == 'all':
        label_names = ['Area', 'Iso_distance', 'Iso_width']
    else:
        label_names = [args.target]

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label_str = "_".join(label_names)
        output_dir = f"runs/mlem_grid_{timestamp}_{label_str}"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "run.log")),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Output directory: {output_dir}")
    
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load Data
    X_train_full, X_test, X_scaler, y_train_full, y_test, y_scaler = load_data(
        csv_file=args.csv_file,
        feature_cols=FEATURE_COLUMN_NAMES,
        label_cols=label_names,
        plots=False,
        rf=output_dir,
        feature_scaler_type=args.feature_scaler,
        label_scaler_type=args.label_scaler
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=args.random_state
    )
    
    # Define Grid
    # We define a few multi-layer architectures to test
    hidden_configs = [
        [1000, 500],
        [2000, 1000],
        [1000, 1000, 500],
        [3000, 1000, 500],
        [5000, 2000, 1000]
    ]
    
    activations = ['ReLU', 'GELU']
    alphas = [1e-3, 1e-1]
    
    param_combinations = list(itertools.product(hidden_configs, activations, alphas))
    logging.info(f"Total config combinations: {len(param_combinations)}")
    
    results = []
    best_val_r2 = -float('inf')
    best_params = None
    
    for idx, (layers, act, alpha) in enumerate(param_combinations, 1):
        logging.info(f"[{idx}/{len(param_combinations)}] Layers={layers}, Act={act}, Alpha={alpha}")
        
        metrics = train_and_evaluate(
            layers, act, alpha,
            X_train, X_val, y_train, y_val,
            device, args.random_state
        )
        
        res = {
            'layers': str(layers),
            'activation': act,
            'alpha': alpha,
            **metrics
        }
        results.append(res)
        
        logging.info(f"  -> R2: {metrics['val_r2']:.4f}, RMSE: {metrics['val_rmse']:.4f}")
        
        if metrics['val_r2'] > best_val_r2:
            best_val_r2 = metrics['val_r2']
            best_params = res.copy()
            logging.info("  *** NEW BEST ***")

    # Evaluate Best on Test
    logging.info("\nFinal training with best params on full train set...")
    best_layers = eval(best_params['layers'])
    
    final_model = MultiLayerELM(
        hidden_layer_sizes=best_layers,
        activation=best_params['activation'],
        alpha=best_params['alpha'],
        device=device,
        random_state=args.random_state
    )
    
    final_model.fit(X_train_full, y_train_full)
    y_test_pred = final_model.predict(X_test)
    
    test_r2 = r2_score(y_test, y_test_pred)
    logging.info(f"Final Test R2: {test_r2:.6f}")
    
    results_file = os.path.join(output_dir, 'mlem_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
