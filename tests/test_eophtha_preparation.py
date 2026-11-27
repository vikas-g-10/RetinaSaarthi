"""Verify local reconstruction using preserved private decision records."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'ml/explainability/e_ophtha_preparation'

def copy_input(tmp_path, relative):
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative, target)

def run(tmp_path, script):
    return subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)

def test_rebuilds_frozen_manifests_from_private_evidence(tmp_path):
    for name in ['outputs/explainability/e_ophtha_review/decisions_v2.csv',
                 'outputs/explainability/e_ophtha_inventory/image_inventory.csv',
                 'outputs/classification/splits/combined_train_aptos_idrid.csv',
                 'outputs/classification/splits/aptos_grouped_seed42.csv',
                 'outputs/classification/replacement_eophtha_v1/validation_decisions.csv']:
        copy_input(tmp_path, name)
    for script in [ROOT / 'ml/explainability/create_eophtha_cohort_v2.py', PREP / 'prepare_manifests.py', PREP / 'finalize_manifests.py']:
        result = run(tmp_path, script)
        assert result.returncode == 0, result.stderr
    for name in ['outputs/explainability/e_ophtha_review/cohort_v2.csv',
                 'outputs/classification/replacement_eophtha_v1/final_train.csv',
                 'outputs/classification/replacement_eophtha_v1/final_validation.csv',
                 'outputs/classification/replacement_eophtha_v1/validation_exclusions.csv']:
        assert (tmp_path / name).read_bytes() == (ROOT / name).read_bytes()

def test_refuses_modified_review_decisions(tmp_path):
    for name in ['outputs/explainability/e_ophtha_review/decisions_v2.csv',
                 'outputs/classification/replacement_eophtha_v1/validation_decisions.csv']:
        copy_input(tmp_path, name)
        with (tmp_path / name).open('ab') as f:
            f.write(b'\n')
    for script in [ROOT / 'ml/explainability/create_eophtha_cohort_v2.py', PREP / 'finalize_manifests.py']:
        result = run(tmp_path, script)
        assert result.returncode != 0
        assert 'decisions differ from the frozen record' in result.stderr
