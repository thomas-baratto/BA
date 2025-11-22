#!/bin/bash
#SBATCH --job-name=DB_TEST
#SBATCH --output=slurm_jobs/db_test_%j.out
#SBATCH --error=slurm_jobs/db_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-00:05:00
#SBATCH -w argon-gtx

# Script to test connection to the MySQL database via SSH tunnel.

set -euo pipefail

# --- Environment setup ---
module purge
module load cuda/12.4.1 || true
source /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/activate
VENV_PYTHON="${VENV_PYTHON:-$(which python 2>/dev/null || echo /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/python)}"
export VENV_PYTHON

echo "--- Running Database Access Test ---"
echo "Using python: $($VENV_PYTHON -V 2>&1)"

# --- Database Configuration ---
# Uses the 'mysql://' prefix, relying on pymysql to be installed.
DB_URL="mysql://root:strong_password@127.0.0.1:3307/optuna_study"
STUDY_NAME="test_connection_study"


# 🎯 CRITICAL FIX: Ensure pymysql is installed before running the connection code.
if ! $VENV_PYTHON -c "import pymysql" 2>/dev/null; then
    echo "Pymysql not found. Attempting to install it now..."
    $VENV_PYTHON -m pip install pymysql 
    
    # Check again if installation worked
    if ! $VENV_PYTHON -c "import pymysql" 2>/dev/null; then
        echo "[FAILURE] Failed to install pymysql. Please check environment dependencies." >&2
        exit 1
    fi
    echo "Pymysql installed successfully."
fi
# --- End dependency check ---


# --- Python Connection Test ---
$VENV_PYTHON - <<END_OF_PYTHON_SCRIPT
import optuna
import sys
import datetime
import warnings
import pymysql # <-- CRITICAL FIX: Import the module here

# Suppress PyMySQL warnings for a cleaner output
warnings.filterwarnings('ignore', category=pymysql.Warning)

# Bash variables expanded into Python strings
DB_URL = "$DB_URL"
STUDY_NAME = "$STUDY_NAME"

print(f"Attempting to connect to: {DB_URL}")
print(f"Using study name: {STUDY_NAME}")

try:
    # Attempt to load a study. This verifies database accessibility.
    optuna.create_study(
        study_name=STUDY_NAME, 
        storage=DB_URL, 
        direction="maximize",
        load_if_exists=True 
    )
    
    # If successful, try to store a value to confirm write access
    study = optuna.load_study(study_name=STUDY_NAME, storage=DB_URL)
    
    # Add a dummy trial to confirm write/read access
    # Use context manager for robust trial creation
    with study.context(optuna.trial.Trial(study, study._storage.create_new_trial_id(study.study_id))) as trial:
        trial.suggest_float("dummy_param", 0.0, 1.0)
        
    print(f"\n[SUCCESS] Optuna successfully connected to MySQL via SSH tunnel at {datetime.datetime.now()}")
    print("Database access is confirmed. You can submit your main optimization job.")
    sys.exit(0)
    
except Exception as e:
    print(f"\n[FAILURE] Connection Test Failed at {datetime.datetime.now()}")
    print("-" * 50)
    print("CRITICAL ERROR DETAILS:")
    print(e)
    print("-" * 50)
    print("Troubleshooting steps:")
    print("1. Verify the SSH tunnel is active on your MacBook.")
    print("2. Verify the MySQL Docker container is running (docker ps).")
    print("3. Check that the database 'optuna_study' exists.")
    print("4. Check your firewall/security settings if the connection timed out.")
    sys.exit(1)

END_OF_PYTHON_SCRIPT

echo "--- Database Access Test Complete ---"