The Isotherm study is study 830 and the Depression Cones study is study 832.

To run the final model training, use the following commands:

For Isotherm:
sbatch --export=STUDY_NAME=nn_study_isotherm_journal,JOURNAL_PATH=runs/global_run_830/optuna_journal_storage/journal.log scripts/slurm/run_final_train.sbatch

For Depression Cones:
sbatch --export=STUDY_NAME=depression_cones_mlp_journal_study,JOURNAL_PATH=runs/global_run_832/optuna_journal_storage/journal.log scripts/slurm/run_final_train.sbatch
