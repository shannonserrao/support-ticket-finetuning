# Spec 01 — Data

## 1. Goal
A deterministic, versioned dataset with train, validation and test splits and a manifest that proves what was used.

## 2. Sources
| Dataset | Config key | Use | To verify in T-02 |
|---|---|---|---|
| Banking77 | `data.dataset_id` | Primary | Hub ID, revision hash, row counts (about 10,003 train and 3,080 test), 77 labels |
| CLINC150 | `data.oos_dataset_id` | Stretch: out-of-scope | Hub ID, revision, out-of-scope label name |

Licences must be checked and written in `docs/data_card.md` before any result is shared.

## 3. Processing
1. Load the pinned revision.
2. Normalise: strip whitespace, drop empty rows, keep original casing.
3. Remove exact-duplicate texts within train; remove any train or validation text that also appears in test.
4. Split the published train set into train and validation (`data.validation_fraction`, stratified by label, `data.seed`).
5. If the file at `feedback.corrections` exists, apply HF-5 from spec 05 (after the split, so validation and test are unchanged).
6. Apply `data.max_train_samples` and `data.max_eval_samples` when set (smoke runs only).

## 4. Output files
- `data/processed/train.jsonl`, `validation.jsonl`, `test.jsonl`: one object per line, `{"id": str, "text": str, "label": str}`.
- `data/labels.json`: ordered list of label names. Index order is the model's class order.
- `data/manifest.json` (committed).

## 5. Manifest
```json
{
  "data_version": "data-v1",
  "created": "ISO-8601",
  "source": {"dataset_id": "...", "revision": "..."},
  "seed": 13,
  "splits": {"train": {"rows": 0, "sha256": "..."}, "validation": {}, "test": {}},
  "labels_sha256": "...",
  "corrections_applied": 0
}
```
`data_version` increments whenever any hash changes.

## 6. Checks (fail the command)
- No id or text shared between any two splits.
- Every label appears in every split.
- Label set equals `data/labels.json` from the previous version unless `data.allow_label_change` is true.

## 7. Acceptance
- Two consecutive `make data` runs produce identical hashes.
- `tests/test_data.py` covers the split, the leakage check and the corrections merge using a small in-memory fixture, without network access.
