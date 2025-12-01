"""
Simple inference script for making predictions with a trained model.
"""
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
import json
import os

from core.model import NeuralNetwork
from core.data_loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_scalers(model_path: str, run_folder: str):
    """Load trained model and scalers from a run folder."""
    
    # Try to load model configuration from model_config.json
    config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
    
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        logging.info(f"Loaded model configuration from {config_path}")
    else:
        # Fall back to inferring from state dict
        logging.warning(f"Model config not found at {config_path}, inferring from state dict")
        model_state = torch.load(model_path, map_location='cpu')
        
        input_size = model_state['layers.0.weight'].shape[1]
        output_size = model_state[list(model_state.keys())[-1]].shape[0]
        
        logging.info(f"Inferred model architecture: input_size={input_size}, output_size={output_size}")
        
        # Use default values that should match trained model
        model_config = {
            'input_size': input_size,
            'output_size': output_size,
            'nr_hidden_layers': 3,
            'nr_neurons': 128,
            'activation_name': "GELU",
            'dropout_rate': 0.0,
            'feature_scaler_type': 'minmax',
            'label_scaler_type': 'minmax'
        }
    
    # Create model with loaded/inferred configuration
    model = NeuralNetwork(
        input_size=model_config['input_size'],
        output_size=model_config['output_size'],
        nr_hidden_layers=model_config['nr_hidden_layers'],
        nr_neurons=model_config['nr_neurons'],
        activation_name=model_config['activation_name'],
        dropout_rate=model_config['dropout_rate']
    )
    
    # Load model state
    model_state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_state)
    model.eval()
    
    logging.info(f"Model loaded successfully from {model_path}")
    logging.info(f"Model architecture: {model_config}")
    
    return model, model_config

def predict(model: torch.nn.Module, 
            X: np.ndarray,
            X_scaler,
            y_scaler) -> np.ndarray:
    """
    Make predictions with the trained model.
    
    Args:
        model: Trained neural network
        X: Input features (unscaled)
        X_scaler: Fitted scaler for features
        y_scaler: Fitted scaler for labels
        
    Returns:
        Predictions in original scale
    """
    # Apply same preprocessing as training
    X_log = np.log1p(X)
    X_scaled = X_scaler.transform(X_log)
    
    # Convert to tensor and predict
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        predictions_scaled = model(X_tensor).numpy()
    
    # Inverse transform
    predictions_log = y_scaler.inverse_transform(predictions_scaled)
    predictions = np.expm1(predictions_log)
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Make predictions with a trained model')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to model file (.pt)')
    parser.add_argument('--run-folder', type=str, required=True,
                       help='Path to run folder containing scalers')
    parser.add_argument('--csv', type=str, default='./data/Clean_Results_Isotherm.csv',
                       help='Path to CSV file with data')
    parser.add_argument('--features', type=str, nargs='+',
                       default=["Flow_well", "Temp_diff", "kW_well", "Hydr_gradient", 
                               "Hydr_conductivity", "Aqu_thickness", "Long_dispersivity", 
                               "Trans_dispersivity", "Isotherm"],
                       help='Feature column names')
    parser.add_argument('--labels', type=str, nargs='+',
                       default=["Area", "Iso_distance", "Iso_width"],
                       help='Label column names (for loading scalers)')
    
    args = parser.parse_args()
    
    # Load model and scaling config
    model, model_config = load_model_and_scalers(args.model, args.run_folder)
    
    # Load data and scalers
    logging.info("Loading data and fitting scalers...")
    X_train, X_test, X_scaler, y_train, y_test, y_scaler = load_data(
        csv_file=args.csv,
        feature_cols=args.features,
        label_cols=args.labels,
        plots=False,
        rf=args.run_folder,
        feature_scaler_type=model_config.get('feature_scaler_type','minmax'),
        label_scaler_type=model_config.get('label_scaler_type','minmax')
    )
    
    # Make predictions on test set
    logging.info("Making predictions on test set...")
    
    # Load original test data (before scaling)
    import pandas as pd
    df = pd.read_csv(args.csv)
    X_original = df[args.features].values
    y_original = df[args.labels].values
    
    # Use the same test split
    from sklearn.model_selection import train_test_split
    _, X_test_original, _, y_test_original = train_test_split(
        X_original, y_original, test_size=0.3, random_state=42, shuffle=True
    )
    
    predictions = predict(model, X_test_original, X_scaler, y_scaler)
    
    # Display results
    logging.info("\nSample Predictions (first 10):")
    logging.info(f"{'True':<40} | {'Predicted':<40}")
    logging.info("-" * 85)
    
    for i in range(min(10, len(predictions))):
        true_str = ", ".join([f"{v:.2f}" for v in y_test_original[i]])
        pred_str = ", ".join([f"{v:.2f}" for v in predictions[i]])
        logging.info(f"{true_str:<40} | {pred_str:<40}")
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_test_original, predictions)
    rmse = root_mean_squared_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)
    
    logging.info(f"\nOverall Metrics:")
    logging.info(f"  MAE:  {mae:.4f}")
    logging.info(f"  RMSE: {rmse:.4f}")
    logging.info(f"  R²:   {r2:.4f}")

if __name__ == "__main__":
    main()
