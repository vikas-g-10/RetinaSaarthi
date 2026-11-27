import pytest
import torch
from ml.explainability import evaluate_eophtha_replacement as replacement
from ml.explainability import evaluate_eophtha as old


def test_replacement_loads_exact_checkpoint_and_separate_output():
    assert replacement.CHECKPOINT != old.CHECKPOINT_PATH
    assert replacement.OUT != old.OUT
    model = replacement.load_replacement()
    checkpoint = torch.load(replacement.CHECKPOINT, map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 4
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), checkpoint['model_state_dict'][key].cpu())


def test_rejects_changed_checkpoint_hash(monkeypatch):
    monkeypatch.setattr(replacement, 'CHECKPOINT_SHA', '0' * 64)
    with pytest.raises(ValueError, match='Frozen input hash mismatch'):
        replacement.load_replacement()


def test_main_uses_replacement_and_cannot_overwrite(monkeypatch, tmp_path):
    target = tmp_path / 'new_output'
    model = object()
    seen = []
    monkeypatch.setattr(replacement, 'OUT', target)
    monkeypatch.setattr(replacement, 'load_replacement', lambda: model)
    monkeypatch.setattr(replacement.base, 'evaluate', lambda *args: seen.append(args))
    replacement.main()
    assert seen == [(model, target, replacement.PROTOCOL_SHA, replacement.CHECKPOINT_SHA)]
    with pytest.raises(FileExistsError):
        replacement.main()
