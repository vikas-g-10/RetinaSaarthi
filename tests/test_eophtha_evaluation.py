import numpy as np
import pytest
from ml.explainability.evaluate_eophtha import mask_for_rule, measure, summarize


def test_mask_value_rules_and_empty_sensitivity():
    mask = np.array([[0, 200, 239, 247, 255]], dtype=np.uint8)
    assert mask_for_rule(mask, 'nonzero').sum() == 4
    assert mask_for_rule(mask, '255_only').sum() == 1
    assert mask_for_rule(mask, 'exclude_239_247').sum() == 2
    result = measure(np.zeros((1, 1)), np.zeros((1, 1), dtype=bool), 0.5)
    assert 'iou' not in result
    with pytest.raises(ValueError):
        mask_for_rule(mask, 'unknown')


def test_threshold_empty_map_and_negative():
    mask = np.array([[True, True, False, False]])
    result = measure(np.array([[0.5, 0.1, 0.7, 0.0]]), mask, 0.5)
    assert result['iou'] == pytest.approx(1 / 3)
    assert result['dice'] == pytest.approx(0.5)
    assert result['coverage'] == pytest.approx(0.5)
    assert measure(np.zeros((1, 4)), mask, 0.5)['dice'] == 0
    negative = measure(np.array([[0.5, 0.0]]), None, 0.5)
    assert negative['active_fraction'] == 0.5 and 'iou' not in negative


def test_summary_is_unweighted_and_counts_empty_masks():
    common = dict(subset='MA', category='positive', cohort='primary', rule='255_only', threshold=0.5)
    rows = [dict(common, iou=0.2, dice=0.3, coverage=0.4),
            dict(common, iou=0.6, dice=0.7, coverage=0.8), dict(common)]
    result = next(x for x in summarize(rows) if x['cohort'] == 'primary')
    assert result['total'] == 3 and result['valid'] == 2 and result['excluded_empty_mask'] == 1
    assert result['iou_mean'] == pytest.approx(0.4)
