The files under reports/quality_cache/ are a regenerable intermediate cache for
the Phase 2 image-quality analyses. They contain derived arrays and masks, not
original dataset files. The cache must not be shared or committed. It may be
recreated by running:

    .venv/bin/python ml/quality/followup_phase2.py cache
