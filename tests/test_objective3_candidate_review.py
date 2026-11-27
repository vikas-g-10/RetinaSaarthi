from reports.review_objective3_candidates import crop_bounds


def test_crop_bounds_preserve_size_at_edges():
    assert crop_bounds(0, 0, 1000, 800, 200) == (0, 0, 200, 200)
    assert crop_bounds(1000, 800, 1000, 800, 200) == (800, 600, 1000, 800)
    assert crop_bounds(500, 400, 1000, 800, 200) == (400, 300, 600, 500)
