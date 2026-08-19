# Issue #1 — acceptance record (regression dataset validator)

Traceability for *Harden tabular-regression dataset validation for DIMER*. Each acceptance
criterion maps to the code and test that satisfy it on `main`. Verified 2026-08-19 (53 unit
tests across the pipeline; validator suite green).

| Acceptance criterion | Status | Evidence |
|---|---|---|
| Validator/finetuner use compatible deterministic split rules | ✅ | Shared `DatasetSource` block byte-identical across `validator.py` + the finetuner's `train.py`, enforced by `scripts/check_shared.py` against a cross-repo pinned SHA |
| Duplicate/ambiguous inputs are rejected | ✅ | `DatasetSource.resolve_single` raises on multiple candidates; `no_duplicate_tables` + `no_nested_zip` checks; `tests/test_validator.py::test_duplicate_train_rejected` |
| Minimum-row threshold uses usable target rows | ✅ | `_usable_target_mask` (finite numeric) drives `minimum_rows`; `test_null_targets_reduce_usable_rows` |
| Negative regression targets are accepted | ✅ | No non-negative constraint; `test_happy_path_negative_targets_ok` |
| Mitra feature limits are enforced | ✅ | `feature_limit` check (≤500) with `MITRA_FEATURE_LIMIT`; `test_feature_limit` |
| Archive/resource limits fail with structured DIMER results | ✅ | `_assert_zip_safe` (compression ratio, per-member + total uncompressed bytes) plus per-file byte ceiling and chunked-read row ceiling **before pandas** (also covers directory inputs); `test_zip_bomb_guard`, `test_member_byte_cap_zip`, `test_row_ceiling_rejected_not_truncated`, `test_directory_mode_byte_cap` |
| Malformed config still yields `result.json` + callback attempt | ✅ | `load_config()` parses inside `main()`'s protected path; failure writes `result.json` and attempts the callback |
| CI covers the validator contract | ✅ | `.github/workflows/ci.yml`: compile + `scripts/check_shared.py` + `pytest` |

**Result metadata** (`metadata` block): resolved columns, `rowCount`, `usableRowCount`,
`featureColumnCount`, per-check warnings vs failures, and archive/source summary — sufficient
for the finetuner and diagnostics without leaking row-level data.

No open items: every criterion is satisfied in-repo. `Closes #1`.
