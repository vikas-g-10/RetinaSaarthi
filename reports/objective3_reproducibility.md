# Objective 3 local reproducibility and sharing boundary

The original checkpoint and the replacement are distinct: the former has an unverified historical training manifest, whereas the replacement run records its own input manifests and code. Both e-ophtha analyses are exploratory. The evaluation data had been inspected before the replacement run, the image-level screen has an unknown miss rate, and patient/eye separation is unverified. Neither analysis is independent or clinical validation.

The [ADCIS e-ophtha license](https://www.adcis.net/en/third-party/e-ophtha/) permits specified non-commercial research, algorithm development and teaching, requires acknowledgement and citation, and prohibits sharing the databases. It does not expressly prohibit aggregate research findings or manifest hashes, and its acknowledgement requirement refers to outputs and manuscripts. The agreement does not explicitly resolve public sharing of image identifiers or per-image derived metadata. As a precaution, retain e-ophtha image identifiers, image/mask hashes, candidate/decision lists, review sheets and per-image results privately; seek the provider's written clarification before publishing them. This is a sharing precaution, not a legal conclusion that every derived value is prohibited. Each researcher must request access directly from the provider and accept its terms. The provider requires acknowledgement of the TeleOphta Consortium and Decencière et al., IRBM (2013), https://doi.org/10.1016/j.irbm.2013.01.010; see the two study reports for the acknowledgement.

## Local evidence and limits

Keep the following existing ignored folders intact and back them up using an access-controlled location permitted by the licenses; do not place them in Git:

- `outputs/explainability/e_ophtha_inventory/`: image inventory with identifiers, sizes, hashes and mask associations.
- `outputs/explainability/e_ophtha_review/`: the 7,002 first-stage pairs, all second-stage scores, 22-row reviewed shortlist, manual decisions, frozen cohort, control results and comparison sheets. The historic original protocol is also preserved publicly as [a historical copy](e_ophtha_initial_protocol.md); the local copy remains the original record.
- `outputs/classification/replacement_eophtha_v1/`: provisional/final manifests and exclusions, complete validation screening, five-pair shortlist, human decisions and review sheets. The two tracked original APTOS and combined split CSVs are inputs, not proof of the old checkpoint's historical training manifest.
- `outputs/classification/replacement_runs/objective3_replacement_v2/`: selected checkpoint, exact manifests, initialization and code provenance, history, protocol copy, Git diff and source image hash list. This checkpoint and training records must not be published or transferred without resolving the terms governing all underlying data and weights.
- `outputs/explainability/e_ophtha_evaluation_v2/` and `outputs/explainability/e_ophtha_replacement_v2/`: original-model and replacement-model per-image and aggregate results, respectively. These are distinct runs. Generated figures, model checkpoints, source images and masks stay out of Git.

The exact model checkpoint, source data and manual-review lists are not available from a clean public clone. A researcher with authorized access needs a separately authorized copy of the original checkpoint and the exact private evidence, or an approved access-controlled way to inspect them; request provider/rights-holder clarification before sharing. A newly trained model is not the same checkpoint, and new manual judgments or independently rerun approximate screening are not guaranteed to reproduce the old reviewed cohort or results. Without the original checkpoint and exact decision evidence, the published aggregate findings cannot be independently reconstructed exactly from this repository alone. Never substitute a different checkpoint or silently update the frozen protocol hashes.

## Reconstructing locally from preserved evidence

Use the exact dataset versions and original extracted folder layout, the recorded Python and package versions in the [replacement report](objective3_replacement_eophtha_results.md), and the tracked original APTOS manifests. The essential private decision records are `decisions_v2.csv` SHA-256 `e5588f0fe4ce168fd1ee047341a2188c43b0855f6523817f5e66f0b3328c4a85` and `validation_decisions.csv` SHA-256 `e6d63376896b8b6356930a08bb346ea7d80ee20c5bd3893c7114465c1ab0cb8e`. The scripts reject changed decision records rather than generating new human judgments. Screening may vary across OpenCV versions and does not recreate or prove a human review; retain the reviewed originals.

For reconstruction, work in a **separate scratch checkout** with appropriately authorized read-only copies of datasets, private evidence and checkpoint. These scripts write files under ignored outputs and can overwrite files of the same name; do not run them over the original evidence. From the scratch project root, restore the original inventory, two private decision CSVs, and the two tracked source split CSVs at their documented paths, and create the output directories. Then use Fish:

```fish
mkdir -p outputs/explainability/e_ophtha_review outputs/explainability/e_ophtha_inventory outputs/classification/replacement_eophtha_v1
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ml/explainability/create_eophtha_cohort_v2.py
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ml/explainability/e_ophtha_preparation/prepare_manifests.py
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ml/explainability/e_ophtha_preparation/finalize_manifests.py
```

Compare hashes with the [frozen evaluation protocol](objective3_replacement_eophtha_protocol_v1.md) and [training plan](objective3_replacement_training_plan.md). Expected cohort SHA-256 is `5421005fcab4cda3c527618f4b04b7460703adb08b75a8e644bb637cc1fbbac0`; final training and validation manifest hashes are `5081c3b6b3fc1803db0ba80f383e7d5666ca093d48c02603dccffca43068c22b` and `0178b5f4740eee80d2a3a4a3dbb5333f4ed27c28d2ead399b5e821a72f8f0291`. The focused reconstruction test verifies those outputs byte-for-byte from the saved local evidence; it does not perform image screening or independently validate review decisions.

The preparation code also preserves the methods for regenerating the image inventory (`inventory.py`), first screen (`screen.py`), second screen and synthetic controls (`second_stage.py`), and validation screen (`validation_screen.py`) in `ml/explainability/e_ophtha_preparation/`. Run only in the scratch checkout, after creating the corresponding output folders and placing the authorized source datasets and original split manifests as expected. The public second-stage script omits one historical mapped-image control to avoid publishing image identifiers; its original results remain in the local evidence. Comparison-sheet scripts are supplied there as well. These screens generate candidate lists, not ground-truth judgments; manually review before applying any new decisions. Because numerical screening and review may differ, they cannot replace the historical decision records. Do not retune thresholds against the evaluation images.

In that scratch checkout, the documented screening sequence is:

```fish
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ml/explainability/e_ophtha_preparation/inventory.py
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python ml/explainability/e_ophtha_preparation/screen.py
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python ml/explainability/e_ophtha_preparation/second_stage.py
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ml/explainability/e_ophtha_preparation/detail_sheets.py
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python ml/explainability/e_ophtha_preparation/validation_screen.py
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ml/explainability/e_ophtha_preparation/validation_sheets.py
```

`validation_screen.py` needs the provisional validation manifest, generated by `prepare_manifests.py` after restoring the historically reviewed training decisions and inventory. Its regenerated shortlist is not a substitute for the historical decision records. Preserve the earlier shortlist separately before attempting this screen.

With the exact replacement checkpoint and cohort restored and their hashes verified, the replacement report gives the evaluation command. It uses a separate output directory and refuses to overwrite an existing result. The final report and per-image files can then be compared locally; floating-point and library differences can affect numerical reproduction. The tests that load frozen private evidence or checkpoints require those local files and are not expected to pass in a bare Git clone. `tests/test_eophtha_preparation.py` specifically tests reconstruction and rejection of modified shortlist evidence; other focused tests check model loading, metrics, masks and training input isolation.
