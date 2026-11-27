"""Evaluate the frozen replacement checkpoint under its pre-training protocol."""
from pathlib import Path

import torch

from ml.classification.model import build_model
from ml.explainability import evaluate_eophtha as base
from ml.explainability.generate_gradcam import DEVICE, ROOT

RUN = ROOT / 'outputs/classification/replacement_runs/objective3_replacement_v2'
CHECKPOINT = RUN / 'best_model.pth'
CHECKPOINT_SHA = '91fcd3a6cdb4f238f63affe148a8c87e4ca0685b82e04bd4c4c718e39913ba71'
PROTOCOL = ROOT / 'reports/objective3_replacement_eophtha_protocol_v1.md'
PROTOCOL_SHA = '7f2e776f7aa4d022e9d754dd1addad00b8bd5fd01412250efab044f59e0ccb86'
OUT = ROOT / 'outputs/explainability/e_ophtha_replacement_v2'


def check_inputs():
    for path, expected in [(CHECKPOINT, CHECKPOINT_SHA), (PROTOCOL, PROTOCOL_SHA),
                           (base.REVIEW / 'cohort_v2.csv', base.COHORT_SHA)]:
        if not path.is_file() or base.digest(path) != expected:
            raise ValueError(f'Frozen input hash mismatch: {path}')


def load_replacement():
    """Load only the exact frozen replacement state, without pretrained downloads."""
    check_inputs()
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    if checkpoint['provenance']['protocol_sha256'] != PROTOCOL_SHA:
        raise ValueError('Checkpoint protocol provenance differs')
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(DEVICE).eval()
    print('Replacement checkpoint:', CHECKPOINT, 'epoch:', checkpoint['epoch'],
          'validation macro-F1:', checkpoint['best_val_macro_f1'], 'device:', DEVICE)
    return model


def main():
    model = load_replacement()
    OUT.mkdir(parents=True, exist_ok=False)
    base.evaluate(model, OUT, PROTOCOL_SHA, CHECKPOINT_SHA)


if __name__ == '__main__':
    main()
