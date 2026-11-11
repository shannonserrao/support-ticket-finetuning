# Data card — Support-Ticket Triage

Fill in the blanks during T-02 and T-03. Do not publish results before the licence row is complete.

| Field | Value |
|---|---|
| Primary dataset | Banking77 (online-banking customer queries, 77 intents) |
| Hub ID and revision | `PolyAI/banking77` @ `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6` (pinned in `configs/base.yaml`) |
| Licence | CC BY 4.0 (from the Hub dataset card; checked 2026-11-11) |
| Stretch dataset | `clinc_oos` @ `155b9c710419136e17307b80d0a13e68cd46b4ec`, config `plus` (15,250 train / 3,100 validation / 5,500 test, 151 labels, out-of-scope label `oos`); licence CC BY 3.0 |
| Base model | `answerdotai/ModernBERT-base` @ `8949b909ec900327062f0ebf497f51aef5e6f0c8`; licence Apache-2.0 |
| Language | English |
| Published splits | Train 10,003; test 3,080; 77 labels (verified by `make verify-sources`) |
| Our splits | Train 8,993, validation 1,000 (10% of train, stratified, seed 13), test 3,080. Cleaning removed 4 exact-duplicate and 6 test-overlapping texts from the published train set. |
| Length (words, p95 / max) | Train 30 / 79; validation 29 / 70; test 24 / 69 |
| Data version | `data-v1` |
| Personal data | None expected; messages are short synthetic-style customer queries. Confirm by sampling. |

## Known limits
- Single domain (retail banking). Results will not transfer to other support domains without new data.
- Messages are short and clean. Real tickets contain signatures, quoted threads and multiple intents.
- Some intents are near-duplicates; expect confusion among card-related labels.
- Published label noise has been reported for this dataset; treat a small share of test errors as label errors and inspect the top confusions.

## Changes
| Data version | Date | Change |
|---|---|---|
| data-v1 | 2026-11-11 | Initial split. Manifest hashes in `data/manifest.json`. |
