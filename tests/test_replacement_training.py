from pathlib import Path

import pandas as pd
import pytest

from ml.classification import train_replacement as replacement


def test_frozen_manifests_and_separation():
    train, val = replacement.load_inputs()
    assert len(train) == 2902 and len(val) == 536
    assert set(train.duplicate_group).isdisjoint(val.duplicate_group)
    assert set(train.diagnosis) == set(val.diagnosis) == set(range(5))


def test_manifest_rejects_foreign_source_and_group_overlap():
    train, val = replacement.load_inputs()
    bad = val.copy()
    bad.loc[0, 'image_path'] = 'datasets/e_ophtha/ex/e_optha_EX/EX/image.jpg'
    with pytest.raises(ValueError, match='Forbidden or missing'):
        replacement.validate_frames(train, bad)
    bad = val.copy()
    bad.loc[0, 'duplicate_group'] = train.loc[0, 'duplicate_group']
    with pytest.raises(ValueError, match='overlap'):
        replacement.validate_frames(train, bad)


def test_hash_and_initialization_fail_closed(tmp_path):
    path = tmp_path / 'weights.pth'
    path.write_bytes(b'not an ImageNet state dictionary')
    with pytest.raises(ValueError, match='Expected full'):
        replacement.validate_weights(path, 'a' * 64)
    with pytest.raises(ValueError, match='Missing or changed'):
        replacement.validate_weights(path, '7f5810bc' + '0' * 56)
    with pytest.raises(ValueError, match='Missing or changed'):
        replacement.verify_file(path, '0' * 64)
