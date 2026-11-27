"""Isolated replacement training; inputs and initialization must be verified."""
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sklearn
import PIL
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from ml.classification.dataset import get_transforms
from ml.classification.dataset_combined import CombinedFundusDataset, get_combined_class_weights
from ml.classification.train_combined import run_epoch, set_seed

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'outputs/classification/replacement_eophtha_v1'
TRAIN_SHA = '5081c3b6b3fc1803db0ba80f383e7d5666ca093d48c02603dccffca43068c22b'
VAL_SHA = '0178b5f4740eee80d2a3a4a3dbb5333f4ed27c28d2ead399b5e821a72f8f0291'
PROTOCOL_SHA = '7f2e776f7aa4d022e9d754dd1addad00b8bd5fd01412250efab044f59e0ccb86'
INITIAL_PREFIX = '7f5810bc'
OUTPUT_ROOT = ROOT / 'outputs/classification/replacement_runs'
SOURCE_DIRS = {'APTOS': ROOT / 'datasets/aptos/train_images',
               'IDRiD': ROOT / 'datasets/idrid/disease_grading/extracted/B. Disease Grading/1. Original Images/a. Training Set'}
CODE_FILES = ['ml/classification/train_replacement.py', 'ml/classification/train_combined.py',
              'ml/classification/dataset.py', 'ml/classification/dataset_combined.py',
              'ml/classification/model.py']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_file(path, expected):
    if not Path(path).is_file() or sha256(path) != expected:
        raise ValueError(f'Missing or changed input: {path}')


def validate_frames(train, val):
    required = {'source', 'id_code', 'diagnosis', 'image_path', 'duplicate_group'}
    for name, df in [('train', train), ('validation', val)]:
        if not required.issubset(df.columns) or df.empty or df[list(required)].isnull().any().any():
            raise ValueError(f'Invalid {name} manifest')
        if set(df['diagnosis']) != set(range(5)):
            raise ValueError(f'{name} must contain all five integer grades')
        if df['image_path'].duplicated().any() or df.duplicated(['source', 'id_code']).any():
            raise ValueError(f'Duplicate {name} image')
        for row in df.itertuples(index=False):
            if row.source not in SOURCE_DIRS or (name == 'validation' and row.source != 'APTOS'):
                raise ValueError(f'Forbidden source: {row.source}')
            path = (ROOT / row.image_path).resolve()
            if path.parent != SOURCE_DIRS[row.source].resolve() or not path.is_file():
                raise ValueError(f'Forbidden or missing image: {path}')
            if path.stem != row.id_code or not str(row.duplicate_group).startswith(row.source + ':'):
                raise ValueError(f'Invalid ID or group for {path}')
    if set(train['image_path']) & set(val['image_path']) or set(train['duplicate_group']) & set(val['duplicate_group']):
        raise ValueError('Train/validation overlap')


def load_inputs():
    train_path, val_path = INPUT / 'final_train.csv', INPUT / 'final_validation.csv'
    protocol = ROOT / 'reports/objective3_replacement_eophtha_protocol_v1.md'
    for path, expected in [(train_path, TRAIN_SHA), (val_path, VAL_SHA), (protocol, PROTOCOL_SHA)]:
        verify_file(path, expected)
    train, val = pd.read_csv(train_path), pd.read_csv(val_path)
    validate_frames(train, val)
    return train, val


def validate_weights(path, expected):
    if len(expected) != 64 or not expected.startswith(INITIAL_PREFIX) or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Expected full lowercase SHA-256 of official ImageNet weights, beginning 7f5810bc')
    verify_file(path, expected)


def build_initialized(path):
    model = efficientnet_b0(weights=None)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    return model


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--weights-sha256', required=True)
    parser.add_argument('--run-name', required=True, help='New simple directory name under replacement_runs')
    args = parser.parse_args()
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        parser.error('run-name must contain only letters, digits, underscores or hyphens')
    train, val = load_inputs()
    validate_weights(args.weights, args.weights_sha256)
    # Instantiate and strictly validate ImageNet state before creating output or training.
    set_seed(42)
    model = build_initialized(args.weights)
    output = OUTPUT_ROOT / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'code_snapshot'; snapshot.mkdir()
    code_hashes = {}
    for rel in CODE_FILES:
        source = ROOT / rel; target = snapshot / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target); code_hashes[rel] = sha256(target)
    for name in ('final_train.csv', 'final_validation.csv', 'validation_decisions.csv', 'validation_exclusions.csv', 'review_exclusions.csv'):
        shutil.copyfile(INPUT / name, output / name)
    shutil.copyfile(ROOT / 'reports/objective3_replacement_eophtha_protocol_v1.md', output / 'evaluation_protocol.md')
    (output / 'git_diff.patch').write_text(git('diff', '--binary', 'HEAD'))
    image_rows = []
    for partition, frame in [('train', train), ('validation', val)]:
        for row in frame.itertuples(index=False):
            image_rows.append(dict(partition=partition, image_path=row.image_path, diagnosis=int(row.diagnosis), sha256=sha256(ROOT / row.image_path)))
    with (output / 'image_hashes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0])); writer.writeheader(); writer.writerows(image_rows)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = dict(seed=42, batch_size=16, max_epochs=10, learning_rate=1e-4,
                  weight_decay=1e-4, patience=3, num_workers=0, image_size=224,
                  augmentation='horizontal flip p=0.5; rotation +/-10; jitter brightness/contrast 0.1 saturation 0.05',
                  normalization='ImageNet', loss='inverse-training-frequency weighted cross entropy',
                  selection='validation macro-F1 grades 0-4, first maximum', mixed_precision=device.type == 'cuda')
    metadata = dict(train_sha256=TRAIN_SHA, validation_sha256=VAL_SHA, protocol_sha256=PROTOCOL_SHA,
                    image_hash_list_sha256=sha256(output / 'image_hashes.csv'), initial_weights_sha256=args.weights_sha256,
                    initial_weights_source='torchvision EfficientNet_B0_Weights.IMAGENET1K_V1; supplied local file',
                    decision_sha256={name: sha256(output / name) for name in
                                     ('validation_decisions.csv', 'validation_exclusions.csv', 'review_exclusions.csv')},
                    code_sha256=code_hashes, git_head=git('rev-parse', 'HEAD').strip(),
                    git_branch=git('branch', '--show-current').strip(), git_status=git('status', '--short'),
                    git_diff_sha256=sha256(output / 'git_diff.patch'), command=sys.argv,
                    python=platform.python_version(), torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                    numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, pillow=PIL.__version__,
                    device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                    started_utc=datetime.now(timezone.utc).isoformat(), config=config,
                    train_grade_counts={str(k):int(v) for k,v in Counter(train['diagnosis']).items()},
                    validation_grade_counts={str(k):int(v) for k,v in Counter(val['diagnosis']).items()})
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    model.to(device)
    train_ds = CombinedFundusDataset(train, get_transforms(training=True))
    val_ds = CombinedFundusDataset(val, get_transforms())
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=device.type=='cuda')
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=device.type=='cuda')
    criterion = nn.CrossEntropyLoss(weight=get_combined_class_weights(train_ds).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    best = -1.; stale = 0
    with (output / 'training_history.csv').open('x', newline='', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_macro_f1', 'val_loss', 'val_macro_f1'])
        writer.writeheader()
        for epoch in range(1, 11):
            tl, tf = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler)
            vl, vf = run_epoch(model, val_loader, criterion, device)
            row = dict(epoch=epoch, train_loss=tl, train_macro_f1=tf, val_loss=vl, val_macro_f1=vf)
            writer.writerow(row); print(row, flush=True)
            if vf > best:
                best, stale = vf, 0
                checkpoint = output / 'best_model.pth'
                torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                                best_val_macro_f1=best, seed=42, provenance=metadata, config=config), checkpoint)
                (output / 'best_model.sha256').write_text(sha256(checkpoint) + '  best_model.pth\n')
            else:
                stale += 1
            if stale >= 3: break
    print('Completed:', output)


if __name__ == '__main__':
    main()
