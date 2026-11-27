from reports.check_objective3_alignment import fits_patch


def test_search_window_boundaries():
    assert fits_patch(36, 36, 100, 100)
    assert fits_patch(64, 64, 100, 100)
    assert not fits_patch(35, 36, 100, 100)
    assert not fits_patch(36, 35, 100, 100)
    assert not fits_patch(65, 64, 100, 100)
    assert not fits_patch(64, 65, 100, 100)
