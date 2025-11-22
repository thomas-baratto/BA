from typing import Any, Dict, Optional

import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error
# The imports below are assumed to be correct based on the original file
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

from data_loader import CSVDataset
from model import NeuralNetwork
from optuna_config import MAX_EPOCHS, PATIENCE
from trainer import evaluate, train_epoch

# DEVICE is set once and will reflect the GPU pinned by the Slurm script
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_objective(
    data: Dict[str, Any],
    base_log_dir: str,
    objective_config: Optional[Dict[str, Any]] = None
):
    # base_log_dir is ignored for objective, as logging per trial is too much overhead
    _ = base_log_dir 

    obj_cfg = {
        "batch_size": 64,
        "nr_hidden_layers": 3,
        "activation_name": "GELU",
        "loss_name": "SmoothL1",
        "batch_size_choices": [32, 64, 128, 256],
        "nr_hidden_layers_range": (2, 5),
        "activation_choices": ['ReLU', 'GELU', 'SiLU', 'LeakyReLU', 'Tanh'],
        "loss_choices": ['L1', 'SmoothL1', 'MSE'],
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "tune_core_hparams": True,
    }
    if objective_config:
        obj_cfg.update({k: v for k, v in objective_config.items() if v is not None})
        
    # Prepare data arrays for scaling and training
    X_train_raw = np.array(data["X_train"])
    X_val_raw = np.array(data["X_val"])
    X_test_raw = np.array(data["X_test"])
    y_train_raw = np.array(data["y_train"])
    y_val_raw = np.array(data["y_val"])
    y_test_raw = np.array(data["y_test"])

    # Determine safe upper bound for quantile n_quantiles
    n_samples_X = X_train_raw.shape[0]
    n_samples_y = y_train_raw.shape[0]

    def objective(trial: optuna.Trial) -> float:
        # --- 1. Suggest Core Model and Optimizer Hyperparameters ---
        
        tune_core = obj_cfg.get("tune_core_hparams", True)

        batch_choices = obj_cfg.get("batch_size_choices") or [obj_cfg.get("batch_size", 64)]
        batch_choices = sorted({int(choice) for choice in batch_choices if isinstance(choice, (int, float)) and choice > 0})
        if not batch_choices:
            batch_choices = [obj_cfg.get("batch_size", 64)]

        hidden_range = obj_cfg.get("nr_hidden_layers_range") or (obj_cfg["nr_hidden_layers"], obj_cfg["nr_hidden_layers"])
        hidden_min = max(1, int(hidden_range[0]))
        hidden_max = max(hidden_min, int(hidden_range[1]))

        activation_choices = obj_cfg.get("activation_choices") or ['ReLU', 'GELU', 'SiLU', 'LeakyReLU', 'Tanh']
        loss_choices = obj_cfg.get("loss_choices") or ['L1', 'SmoothL1', 'MSE']

        if tune_core:
            batch_size = trial.suggest_categorical("batch_size", tuple(batch_choices))
            nr_hidden_layers = trial.suggest_int("nr_hidden_layers", hidden_min, hidden_max)
            activation_name = trial.suggest_categorical("activation_name", tuple(activation_choices))
            loss_name = trial.suggest_categorical("loss_criterion", tuple(loss_choices))
        else:
            batch_size = obj_cfg["batch_size"]
            nr_hidden_layers = obj_cfg["nr_hidden_layers"]
            activation_name = obj_cfg["activation_name"]
            loss_name = obj_cfg["loss_name"]
            
        max_epochs = obj_cfg["max_epochs"]
        patience_limit = obj_cfg["patience"]

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        nr_neurons = trial.suggest_int("nr_neurons", 16, 256, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        
        # Force ReduceLROnPlateau scheduler and tune its internal parameters
        rlr_factor = trial.suggest_float("rlr_factor", 0.1, 0.9)
        rlr_patience = trial.suggest_int("rlr_patience", 1, 10)
        rlr_min_lr = trial.suggest_float("rlr_min_lr", 1e-6, 1e-4, log=True)
        
        # --- 2. Suggest Scaler Choices and their Parameters (As per original) ---
        
        feature_scaler = trial.suggest_categorical(
            "feature_scaler", ('minmax', 'standard', 'robust', 'quantile', 'none')
        )
        label_scaler = trial.suggest_categorical(
            "label_scaler", ('minmax', 'standard', 'robust', 'quantile', 'none')
        )

        feature_min, feature_max = 0.0, 1.0
        label_min, label_max = 0.0, 1.0
        
        # MinMax: pick signed or unsigned range
        if feature_scaler == 'minmax':
            feature_signed = trial.suggest_categorical("minmax_feature_signed", (False, True))
            feature_min, feature_max = (-1.0, 1.0) if feature_signed else (0.0, 1.0)

        if label_scaler == 'minmax':
            label_signed = trial.suggest_categorical("minmax_label_signed", (False, True))
            label_min, label_max = (-1.0, 1.0) if label_signed else (0.0, 1.0)

        # StandardScaler flags
        std_with_mean = True
        std_with_std = True
        if feature_scaler == 'standard' or label_scaler == 'standard':
            # Only suggest once, apply to both feature/label if standard is chosen for either
            std_with_mean = trial.suggest_categorical("std_with_mean", (True, False))
            std_with_std = trial.suggest_categorical("std_with_std", (True, False))

        # Robust scaler quantile range
        robust_q_low = 25.0
        robust_q_high = 75.0
        if feature_scaler == 'robust' or label_scaler == 'robust':
            # Only suggest once, apply to both feature/label if robust is chosen for either
            robust_q_low = float(trial.suggest_int("robust_q_low", 1, 40))
            robust_q_high = float(trial.suggest_int("robust_q_high", 60, 99))

        # Quantile transformer params
        quantile_n_q_X, quantile_n_q_y = None, None
        quantile_output = 'normal'
        
        if feature_scaler == 'quantile' or label_scaler == 'quantile':
            low = 2
            high_X = min(1000, n_samples_X)
            high_y = min(1000, n_samples_y)
            
            quantile_n_q_X = trial.suggest_int("quantile_n_q_X", min(50, high_X), high_X)
            quantile_n_q_y = trial.suggest_int("quantile_n_q_y", min(50, high_y), high_y)
            quantile_output = trial.suggest_categorical("quantile_output", ('normal', 'uniform'))
        
        # --- 3. Instantiate Scalers and Apply Transformation ---
        
        def make_scaler_with_params(kind: str, is_label: bool = False):
            k = (kind or '').lower()
            if k == 'minmax':
                if is_label:
                    return MinMaxScaler(feature_range=(label_min, label_max))
                return MinMaxScaler(feature_range=(feature_min, feature_max))
            if k == 'standard':
                return StandardScaler(with_mean=std_with_mean, with_std=std_with_std)
            if k == 'robust':
                return RobustScaler(quantile_range=(robust_q_low, robust_q_high))
            if k == 'quantile':
                n_q = quantile_n_q_y if is_label else quantile_n_q_X
                return QuantileTransformer(n_quantiles=n_q, output_distribution=quantile_output, subsample=n_q)
            if k == 'none':
                return None
            return MinMaxScaler(feature_range=(0.0, 1.0)) # Default fallback

        X_scaler = make_scaler_with_params(feature_scaler)
        y_scaler = make_scaler_with_params(label_scaler, is_label=True)

        # Create *copies* of raw data to be scaled per-trial
        X_train = X_train_raw.copy()
        X_val = X_val_raw.copy()
        X_test = X_test_raw.copy()
        y_train = y_train_raw.copy()
        y_val = y_val_raw.copy()
        y_test = y_test_raw.copy()

        # Apply per-trial scaling (fit on training only to avoid leakage)
        if X_scaler is not None:
            X_train = X_scaler.fit_transform(X_train)
            X_val = X_scaler.transform(X_val)
            X_test = X_scaler.transform(X_test)
        if y_scaler is not None:
            y_train = y_scaler.fit_transform(y_train)
            y_val = y_scaler.transform(y_val)
            y_test = y_scaler.transform(y_test)
        
        # --- 4. Record User Attributes (as per original) ---
        
        trial.set_user_attr("batch_size", batch_size)
        trial.set_user_attr("nr_hidden_layers", nr_hidden_layers)
        trial.set_user_attr("activation_name", activation_name)
        trial.set_user_attr("loss_criterion", loss_name)
        trial.set_user_attr("weight_decay", float(weight_decay))
        trial.set_user_attr("dropout_rate", float(dropout_rate))
        trial.set_user_attr("feature_scaler", feature_scaler)
        trial.set_user_attr("label_scaler", label_scaler)
        
        if quantile_n_q_X is not None:
             trial.set_user_attr("quantile_n_q_X", int(quantile_n_q_X))
             trial.set_user_attr("quantile_n_q_y", int(quantile_n_q_y))
             trial.set_user_attr("quantile_output", str(quantile_output))
        
        trial.set_user_attr("rlr_factor", float(rlr_factor))
        trial.set_user_attr("rlr_patience", int(rlr_patience))
        trial.set_user_attr("rlr_min_lr", float(rlr_min_lr))
        
        # --- 5. Training and Evaluation ---

        train_dataset = CSVDataset(X_train, y_train)
        val_dataset = CSVDataset(X_val, y_val)
        
        # CRITICAL FIX: num_workers=0 to prevent contention with 56 tasks/1 core
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=True)

        model = NeuralNetwork(
            input_size=data["X_train"].shape[1],
            output_size=data["y_train"].shape[1],
            nr_hidden_layers=nr_hidden_layers,
            nr_neurons=nr_neurons,
            activation_name=activation_name,
            exp_layers=False,
            con_layers=False,
            dropout_rate=dropout_rate,
            use_batchnorm=trial.suggest_categorical("use_batchnorm", (True, False)),
        ).to(DEVICE)

        if loss_name == "L1":
            criterion = nn.L1Loss()
        elif loss_name == "SmoothL1":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Using ReduceLROnPlateau as suggested
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=rlr_factor, patience=rlr_patience, min_lr=rlr_min_lr, mode='min'
        )

        best_val_rmse = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

            val_loss, predictions, true_values = evaluate(model, val_loader, criterion, DEVICE)
            current_val_rmse = root_mean_squared_error(true_values, predictions)

            # ReduceLROnPlateau expects a metric; use validation RMSE
            scheduler.step(current_val_rmse)

            if current_val_rmse < best_val_rmse:
                best_val_rmse = current_val_rmse
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                break

            trial.report(current_val_rmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_rmse

    return objective