# copy this into terminal to submit all jobs
sbatch --export=TARGET_LABEL=Iso_distance run.sbatch
sbatch --export=TARGET_LABEL=Iso_width run.sbatch
sbatch --export=TARGET_LABEL=area run.sbatch
sbatch --export=TARGET_LABEL=all run.sbatch