#!/usr/bin/env bash
# One-shot cleanup script for BA artifact inconsistencies.
# Idempotent — safe to re-run; each step checks whether it's already applied.
#
# Usage:  cd /path/to/BA && bash scripts/cleanup_artifacts.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "=== BA Artifact Cleanup ==="
echo "Working directory: $(pwd)"

# ── 1. Promote timestamped cone dir if it exists ──
TS_CONE="artifacts/models/mlp/cone_MLP_20260330-171026_Cone"
if [[ -d "$TS_CONE" ]]; then
    echo "[1] Promoting timestamped cone -> canonical"
    rm -rf artifacts/models/mlp/cone_old
    mv artifacts/models/mlp/cone artifacts/models/mlp/cone_old
    mv "$TS_CONE" artifacts/models/mlp/cone
    rm -rf artifacts/models/mlp/cone_old
else
    echo "[1] Timestamped cone dir already promoted (skip)"
fi

# ── 2. Promote timestamped isotherm dir if it exists ──
TS_ISO="artifacts/models/mlp/isotherm_MLP_20260330-172252_Area_Iso_distance_Iso_width"
if [[ -d "$TS_ISO" ]]; then
    echo "[2] Promoting timestamped isotherm -> canonical"
    rm -rf artifacts/models/mlp/isotherm_old
    mv artifacts/models/mlp/isotherm artifacts/models/mlp/isotherm_old
    mv "$TS_ISO" artifacts/models/mlp/isotherm
    rm -rf artifacts/models/mlp/isotherm_old
else
    echo "[2] Timestamped isotherm dir already promoted (skip)"
fi

# ── 3. Regenerate results JSONs from stats/metrics_summary.json ──
echo "[3] Rebuilding results_MLP_*.json from stats"
.venv/bin/python -c "
import json, pathlib

for dataset in ('cone', 'isotherm'):
    base = pathlib.Path(f'artifacts/models/mlp/{dataset}')
    manifest = json.loads((base / 'artifact_manifest.json').read_text())
    stats = json.loads((base / 'stats/metrics_summary.json').read_text())
    cfg = json.loads((base / 'model_config.json').read_text())

    results = {
        'config': {
            'model': 'MLP',
            'dataset': dataset,
            'nr_hidden_layers': cfg['nr_hidden_layers'],
            'nr_neurons': cfg['nr_neurons'],
            'activation_name': cfg['activation_name'],
            'batch_size': manifest['training']['batch_size'],
            'loss_criterion': manifest['training']['loss_criterion'],
            'learning_rate': manifest['training']['learning_rate'],
            'dropout_rate': cfg['dropout_rate'],
            'weight_decay': manifest['training'].get('weight_decay', 0),
            'feature_scaler': cfg['feature_scaler_type'],
            'label_scaler': cfg['label_scaler_type'],
            'use_batchnorm': cfg['use_batchnorm'],
            'feature_scaler_type': cfg['feature_scaler_type'],
            'label_scaler_type': cfg['label_scaler_type'],
            'plots': True,
            'num_epochs': manifest['training']['epochs'],
            'patience': 250,
            'use_log': cfg.get('use_log', False),
        },
        'study_info': {
            'study_name': manifest.get('study_name', ''),
            'best_trial': manifest.get('trial_number', 0),
            'best_value': manifest.get('best_value', 0),
        },
        'metrics': {'test': {'aggregate': stats['test']['overall']}},
        'train_time_seconds': manifest['training']['train_time_seconds'],
        'device': 'cuda',
    }

    out = base / f'results_MLP_{dataset}.json'
    out.write_text(json.dumps(results, indent=2))

    text = out.read_text()
    assert 'Infinity' not in text, f'{out} still has Infinity!'
    assert 'NaN' not in text, f'{out} still has NaN!'
    print(f'  {out}: R2={stats[\"test\"][\"overall\"][\"r2\"]:.6f}  RMSE={stats[\"test\"][\"overall\"][\"rmse\"]:.4f}  OK')
"

# ── 4. Rebuild summary_table.csv ──
echo "[4] Rebuilding summary_table.csv"
.venv/bin/python -c "
import json, pathlib, csv, io

header = ['Model','Dataset','RMSE','MAE','MSE','MAPE','R2','nRMSE','KGE','Time(s)','Folder','N_Hidden','Layers','N_Ensemble','Blocks','Activation']
rows = []
for dataset in ('cone', 'isotherm'):
    base = pathlib.Path(f'artifacts/models/mlp/{dataset}')
    stats = json.loads((base / 'stats/metrics_summary.json').read_text())
    cfg = json.loads((base / 'model_config.json').read_text())
    manifest = json.loads((base / 'artifact_manifest.json').read_text())
    m = stats['test']['overall']
    rows.append([
        'MLP', dataset, m['rmse'], m['mae'], m['mse'], m['mape'],
        m['r2'], m['nrmse'], m['kge'],
        manifest['training']['train_time_seconds'],
        dataset, cfg['nr_neurons'], cfg['nr_hidden_layers'],
        '', '', cfg['activation_name']
    ])

out = pathlib.Path('artifacts/models/mlp/summary_table.csv')
buf = io.StringIO()
w = csv.writer(buf)
w.writerow(header)
w.writerows(rows)
out.write_text(buf.getvalue())
print(f'  Wrote {len(rows)} rows to {out}')
"

# ── 5. Fix config/best_params_cone.json use_log ──
echo "[5] Ensuring best_params_cone.json has use_log: true"
.venv/bin/python -c "
import json, pathlib
p = pathlib.Path('config/best_params_cone.json')
d = json.loads(p.read_text())
if d.get('use_log') != True:
    d['use_log'] = True
    p.write_text(json.dumps(d, indent=2))
    print('  Fixed use_log -> true')
else:
    print('  Already correct')
"

# ── 6. Remove stale backup config ──
if [[ -f config/best_params_cone_backup.json ]]; then
    echo "[6] Removing stale best_params_cone_backup.json"
    rm config/best_params_cone_backup.json
else
    echo "[6] Backup already removed (skip)"
fi

# ── 7. Fix datasets.py study name ──
echo "[7] Fixing isotherm study_name in datasets.py"
if grep -q 'nn_study_isotherm_arearoot' config/datasets.py; then
    sed -i 's/nn_study_isotherm_arearoot/nn_study_isotherm_journal/' config/datasets.py
    echo "  Fixed -> nn_study_isotherm_journal"
else
    echo "  Already correct (skip)"
fi

# ── 8. Clean stale packages dir ──
if [[ -d artifacts/packages/mlp ]] && find artifacts/packages/mlp -name "model_config.json" -exec grep -l '"nr_neurons": 221' {} + 2>/dev/null | grep -q .; then
    echo "[8] Removing stale packages (old trial 6375 models)"
    rm -rf artifacts/packages/mlp
else
    echo "[8] Packages already clean (skip)"
fi

# ── 9. Validation summary ──
echo ""
echo "=== Validation ==="
echo "Cone model_config:"
grep -o '"activation_name": "[^"]*"' artifacts/models/mlp/cone/model_config.json
grep -o '"nr_neurons": [0-9]*' artifacts/models/mlp/cone/model_config.json

echo "Isotherm model_config:"
grep -o '"activation_name": "[^"]*"' artifacts/models/mlp/isotherm/model_config.json
grep -o '"nr_neurons": [0-9]*' artifacts/models/mlp/isotherm/model_config.json

echo "Results sanity check (no Infinity/NaN):"
if grep -r "Infinity\|NaN" artifacts/models/mlp/*/results_MLP_*.json 2>/dev/null; then
    echo "  FAIL: corrupted values found!"
    exit 1
else
    echo "  PASS"
fi

echo "Config consistency:"
grep '"trial_number"' config/best_params_cone.json
grep '"use_log"' config/best_params_cone.json

echo ""
echo "=== Done ==="
