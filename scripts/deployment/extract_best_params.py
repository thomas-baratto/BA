import json
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os

from config.datasets import DATASET_CONFIGS

for name, cfg in DATASET_CONFIGS.items():
    journal_path = cfg["journal_path"]
    if not os.path.exists(journal_path):
        print(f"Skipping {name}: journal not found at {journal_path}")
        continue

    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.load_study(study_name=cfg["study_name"], storage=storage)

    data = {
        "best_params": study.best_trial.params,
        "trial_number": study.best_trial.number,
        "best_value": study.best_trial.value
    }
    out_file = cfg["best_params_file"]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {name} best params to {out_file}")
