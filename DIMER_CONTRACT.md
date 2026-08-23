# Mitra ↔ DIMER Workbench contract

This repository owns the **Mitra worker side** of the integration contract. The ML implementations remain task-specific; the shared hardening layer only standardizes the DIMER-facing wire contract.

The three layers are: repository `dimer-pipeline.json` for user-facing parameters; `dimer-runtime-contract.json` for DIMER runtime inputs; and versioned `result.json`/artifact outputs.

Exact task identities are `tabular_classification` and `tabular_regression`. Worker results always report their repo-owned task identity; a conflicting platform task is diagnostic only.

Validator and finetuner share one dataset identity: exact selected root ZIP SHA-256, or a deterministic path-framed directory-tree digest. Validator exposes `metadata.datasetIdentity`/`metadata.datasetSha256`; finetuner provenance uses the same convention.

Results are additive-compatible schema v1 with stable machine-facing codes. `message` is human-facing only. Validator `metadata.classNames` remains mandatory: actual names for classification when known and `[]` for regression/unknown.

The logical finetuner model is an AutoGluon `TabularPredictor` directory. `artifacts/best.pt` remains a ZIP compatibility package, not a native PyTorch checkpoint. Its descriptor records logical/package format, exact packaged-byte SHA-256, and `reloadVerified=true` only after unpack→reload→predict succeeds. Secondary evaluation/log/progress JSON is stable schema v1.

Decision: **KEEP_PARITY_COPIES**. Contract bytes are pinned by `scripts/check_contract.py`; task-specific ML logic remains separate.

## DIMER-side requirements — documentation only

**No DIMER/backend code is changed by this work.** External platform requirements remain: first-class tabular task typing; task-neutral generic defaults; typed packaged/directory artifacts; backend validation/derivation from `dimer-pipeline.json`; result/artifact schema validation; preservation of validated dataset identity into finetuning; non-vision AutoGluon serving; and confirmation of Base Model upload/model-id transport (#9). Nested-ZIP behavior remains separately tracked in validator #8 and is not changed here.
