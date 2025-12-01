import argparse
import logging
import os
import numpy as np

import optuna
from torch.utils.tensorboard import SummaryWriter
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Export Optuna Study to TensorBoard.")
    parser.add_argument(
        '--storage-path', 
        type=str,
        required=True,
        help='Path to the Journal Storage directory.'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        required=True,
        help='Name of the Optuna study.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for TensorBoard event files.'
    )
    return parser.parse_args()

def export_study_to_tensorboard(journal_dir: str, study_name: str, output_dir: str):
    
    journal_file_path = os.path.join(journal_dir, "journal.log")
    
    # 1. Load the study using the Journal Backend
    try:
        backend = JournalFileBackend(journal_file_path)
        optuna_storage = JournalStorage(backend)
        study = optuna.load_study(
            study_name=study_name,
            storage=optuna_storage
        )
    except Exception as e:
        logging.error(f"Failed to load Optuna study '{study_name}' from {journal_file_path}: {e}")
        return

    writer = SummaryWriter(log_dir=output_dir)
    
    logging.info(f"Loaded study with {len(study.trials)} trials. Exporting to {output_dir}")

    # 2. Log Trial-by-Trial History
    best_value = float('inf')
    
    for i, trial in enumerate(study.trials):
        # Only process completed or pruned trials that have values
        if trial.state.is_finished() and trial.value is not None:
            
            # Log primary metric
            writer.add_scalar('Optuna/Optimization_History', trial.value, i)
            
            # Log best value found so far
            if trial.value < best_value:
                best_value = trial.value
            writer.add_scalar('Optuna/Best_Value', best_value, i)

            # Log key user attributes (metadata)
            if 'final_test_rmse' in trial.user_attrs:
                writer.add_scalar(
                    'Metrics/Final_Test_RMSE', 
                    trial.user_attrs['final_test_rmse'], 
                    i
                )
            if 'trial_wall_time_sec' in trial.user_attrs:
                writer.add_scalar(
                    'Metrics/Trial_Wall_Time_sec', 
                    trial.user_attrs['trial_wall_time_sec'], 
                    i
                )
            
            # Log sampled hyperparameters individually
            for key, value in trial.params.items():
                try:
                    # Log as HParam/Value (assuming float/int types)
                    writer.add_scalar(f'Hyperparameters/{key}', value, i)
                except (TypeError, ValueError):
                    # Handle non-numeric types (e.g., categorical strings) as text
                    pass 
    
    # 3. Log Best Parameters and Final Metrics as Text
    best_trial = study.best_trial
    
    text_summary = f"""
    ## Optuna Study Summary: {study_name}
    * **Total Trials:** {len(study.trials)}
    * **Best Trial (#):** {best_trial.number}
    * **Best Validation Loss (Objective):** {best_trial.value:.6f}
    * **Final Test RMSE (User Metric):** {best_trial.user_attrs.get('final_test_rmse', 'N/A'):.6f}
    
    ### Best Hyperparameters:
    {best_trial.params}
    """
    writer.add_text('Summary/Best_Results', text_summary, 0)
    
    # 4. Log Parameter Importance (requires scikit-learn dependency)
    try:
        importance = optuna.importance.get_param_importances(study)
        
        # Convert dictionary to Markdown table
        importance_table = "| Parameter | Importance Score |\n|---|---|\n"
        for param, score in importance.items():
            importance_table += f"| {param} | {score:.4f} |\n"
            writer.add_scalar(f'Importance/{param}', score, 0)
            
        writer.add_text('Summary/Parameter_Importance', importance_table, 0)
    except Exception as e:
        logging.warning(f"Could not compute or log parameter importance (requires completed trials): {e}")

    writer.close()
    logging.info("TensorBoard export complete.")


if __name__ == "__main__":
    args = parse_args()
    export_study_to_tensorboard(args.storage_path, args.study_name, args.output_dir)