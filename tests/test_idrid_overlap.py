import numpy as np
import pytest

from ml.explainability.evaluate_idrid_overlap import overlap


def test_overlap_known_counts():
    activation = np.array([[1, 1], [0, 0]], dtype=bool)
    lesion = np.array([[1, 0], [1, 0]], dtype=bool)
    assert overlap(activation, lesion) == pytest.approx((1 / 3, 1 / 2, 1 / 2))


def test_empty_activation_and_empty_lesion():
    empty = np.zeros((2, 2), dtype=bool)
    assert overlap(empty, ~empty) == (0, 0, 0)
    with pytest.raises(ValueError, match='Empty lesion'):
        overlap(empty, empty)


def test_threshold_after_bilinear_resize_and_empty_cam():
    from ml.explainability.evaluate_idrid_overlap import threshold_cam

    cam = np.array([[0.0, 1.0]], dtype=np.float32)
    assert threshold_cam(cam, (3, 1)).tolist() == [[False, True, True]]
    assert not threshold_cam(np.zeros((2, 2), dtype=np.float32), (5, 5)).any()


def test_aggregate_excludes_incomplete_and_is_unweighted():
    from ml.explainability.evaluate_idrid_overlap import complete_file_mean

    rows = [dict(complete_files=True, iou=0.2, dice=0.4, lesion_coverage=0.6),
            dict(complete_files=True, iou=0.4, dice=0.6, lesion_coverage=0.8),
            dict(complete_files=False, iou=1.0, dice=1.0, lesion_coverage=1.0)]
    assert complete_file_mean(rows) == pytest.approx(dict(iou=0.3, dice=0.5, lesion_coverage=0.7))
    assert complete_file_mean(rows[2:]) is None


def test_comparison_codes_distinguish_overlap():
    from ml.explainability.evaluate_idrid_overlap import comparison_codes

    active = np.array([[False, True], [False, True]])
    lesion = np.array([[False, False], [True, True]])
    assert comparison_codes(active, lesion).tolist() == [[0, 1], [2, 3]]
