# RetinaSaarthi final single-run progress

Prior staged pass completed before this run: backend Python 3.13 environment, parity, and migration 003 were previously verified; this run re-verifies anything claimed in the final report.
[done] 1.1, migration 003 printed and verified; both functions are SECURITY INVOKER, RPC EXECUTE is revoked from public/anon, and select policies match.
[done] 1.2, four new backend closeout tests pass: chunked 413, bad-checkpoint 503/no-write, RPC 502, and returned result_id.
[blocked] 1.3, pgserver install failed because no cp313 wheel exists; no local Postgres shim or RLS verification was possible.
[done] 1.4, parity moved to scripts/verify_parity.py; stage logs and temporary uv/pgserver artifacts removed; .gitignore added.
[done] 1.5, requirements tags stripped and README Python 3.13 setup section added.
[done] 1.6, uv dry-run reported “Checked 18 packages” and “Would make no changes”; full backend suite passed 42 tests.
[done] 2.1, AppPages.tsx was deleted; pages are real per-page files and the final page listing/line counts were captured.
[done] 2.2, History now uses exact-count Supabase range pagination, search, filters, sorting, responsive rows/cards, retry, and database-not-migrated state.
[done] 2.3, client multipart/bearer behavior and distinct classification error/RECAPTURE handling were verified from source output.
[done] 2.4, auth computed-style script passed: #07060d, 22px, 9px, requested gradient, DM Sans, Space Grotesk.
[done] 2.5, identity/prohibited-term greps were run; only legacy unused demo CSS selectors matched and no service_role hit was found.
[done] 2.6, final frontend lint and build passed; build emitted only dependency/chunk-size warnings.
[done] 3.1, Playwright and Chromium/headless-shell were installed and terminal specs save PNG screenshots under frontend/e2e/screenshots.
[done] 3.2, mocks/localStorage injection are confined to e2e specs and every visual screenshot is labeled “layout check with mocked responses”.
[done] 3.3, core visual suite passed 5 tests; expanded empty/result/RECAPTURE/history-state breakpoint suite passed 2 tests, including 390px overflow checks and keyboard/drawer assertions.
[done] 3.4, auth baseline was recorded by computed styles because no Git history or original screenshots exist.

## Objective 3 — exploratory quantitative explainability (2026-11-27)

The entries below are chronological: earlier preparation and blocker notes describe the state at that time and are superseded by later entries. The original model's manifest provenance remains unknown; the separately trained replacement has a recorded training snapshot. Both e-ophtha analyses are exploratory, and independent validation remains pending.

[done] Objective 3 prototype implementation and three-image exploratory study are complete; see the [final study report](reports/objective3_final_report.md).

[pending] Independent validation remains outstanding: checkpoint training inputs and patient- or eye-level independence are not verified. The findings are not clinical validation.

[done] Recorded a fixed protocol and calculated predicted-class Grad-CAM overlap for three mapped IDRiD grade-2 candidates; see [protocol](reports/objective3_protocol.md) and [results and verification](reports/objective3_results.md). Original datasets and checkpoint were not modified. The 09 candidate has an incomplete mask set and is reported separately; the two candidates with four mask files have mean IoU 0.0447, Dice 0.0854, and recorded-lesion coverage 0.2741 at the fixed 0.5 threshold.

[verified] Candidate files, mask dimensions and values, checkpoint metadata, current manifest, duplicate audit and sampled local alignment were checked. The current manifest is not cryptographically linked to the checkpoint; neither the duplicate audit nor the alignment sampling proves independence or perfect alignment.

[remaining] For the original model, establish checkpoint training provenance and patient-level independence, review lesion alignment, and evaluate a larger independently held-out set. Threshold sensitivities were subsequently assessed in the e-ophtha exploratory studies. These overlap scores are not clinical validation and do not establish clinical correctness.

[done] Documented visual observations and inventoried all 81 segmentation-image mappings in [the candidate inventory](reports/objective3_candidate_inventory.csv). Three are existing exploratory candidates; two additional grading-test best matches remain unverified, and the current artifacts do not establish a larger independent evaluation set. See [the results report](reports/objective3_results.md) for evidence and limitations.

[done] Reviewed uncertain 68/034 and 70/006 image pairs and sampled lesion neighborhoods for 70/006. Correspondence remains uncertain for 68/034; 70/006 appears to show the same eye but is displaced. Both remain excluded; patient independence and checkpoint provenance remain unverified. See [the results report](reports/objective3_results.md).

[done] Audited available checkpoint, history, manifest, Git and overlap records. The history and checkpoint match, but no record identified binds the checkpoint to the exact manifest; see [the provenance audit](reports/objective3_results.md).

[done] The separate e-ophtha EX and MA **external exploratory** evaluation is complete under the pre-inference [version 2 protocol](reports/e_ophtha_protocol_v2.md); see the [results and limitations](reports/e_ophtha_exploratory_results.md). The original duplicate gate was not validated and was explicitly revised for exploration. The historical checkpoint training inputs, image-level and patient/eye-level independence remain unverified. Independent validation and clinical validation remain pending; these results do not establish clinical correctness.

[prepared] A [replacement training plan](reports/objective3_replacement_training_plan.md) and a [separate pre-training e-ophtha protocol](reports/objective3_replacement_eophtha_protocol_v1.md) are documented. Derived manifests are provisional pending the validation overlap screen, local initialization-weight verification and an isolated, tested training entry point. No replacement training or evaluation has run; previously viewed e-ophtha results and unknown patient/eye linkage limit any future independence claim.

[prepared] The replacement validation duplicate screen and conservative exclusions are recorded in the [training plan](reports/objective3_replacement_training_plan.md). Frozen manifests contain 2,902 training and 536 validation images; five unresolved validation candidates were excluded. The isolated training entry point and focused tests are in place, but compatible ImageNet initialization weights were not found in the visible local cache and no full file hash could be verified. No replacement training or new e-ophtha evaluation has run; patient/eye independence remains unverified.

[prepared] On 2026-11-28 the replacement ImageNet initialization was obtained from the official torchvision URL and its filename hash prefix verified. Host PyTorch sees the RTX 4050 and a CUDA tensor check passed; the earlier failure was due to the sandbox hiding GPU devices. The isolated model-load and frozen-input preflight passed; see the [training plan](reports/objective3_replacement_training_plan.md) for the exact command and full weight hash. Training has not started, and a replacement-specific evaluation path remains to be verified.

[done] The replacement checkpoint's recorded provenance and best validation epoch were checked, then its separate e-ophtha exploratory evaluation was completed under the pre-training protocol. See [replacement results](reports/objective3_replacement_eophtha_results.md). e-ophtha had previously been inspected and patient/eye independence remains unverified; independent and clinical validation remain pending.

[reviewed] The [reproducibility and sharing review](reports/objective3_reproducibility.md) distinguishes the original and replacement checkpoints and documents private evidence needed for exact reconstruction. Preparation scripts reject changed private decision records; focused tests rebuilt the saved manifests and cohort from preserved local evidence. Authorized dataset access alone is insufficient without the exact private decisions and checkpoint. Independent validation remains pending.
