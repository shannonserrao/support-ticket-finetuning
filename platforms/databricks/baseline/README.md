# Zero-shot baseline on Databricks (task T-05, feeds T-09)

This folder is uploaded to a Databricks workspace. The notebook `baseline_zero_shot.py` classifies the validation
and test tickets with an expensive model behind a model serving endpoint and writes result files. Those files are
then imported back into this repository, where `make predict eval` uses them to choose the routing threshold and
to evaluate the cascade.

Status: the notebook logic is tested offline against a stand-in endpoint (`tests/test_baseline_notebook.py`).
It has not been run in a Databricks workspace from this repository yet; treat the first run as a check.

## What you need
- A Databricks workspace with a **chat model serving endpoint** (Foundation Model APIs, an external model, or
  your own). Note its name and its price per million input and output tokens.
- A cluster or serverless notebook compute with `mlflow` (any ML runtime).
- About 5,000 requests in total: 1,000 validation, 3,080 test, the rest for prompt development.

## 1. Build the bundle (in the repository)
```bash
python -m triage.cli data              # only if data/processed/ does not exist yet
python -m triage.cli baseline-bundle   # or: make baseline-bundle
```
This writes `platforms/databricks/baseline/inputs/` (git-ignored):

| File | Content |
|---|---|
| `validation.jsonl` | 1,000 validation tickets with labels (for prompt development) |
| `test.jsonl` | 3,080 test tickets, **id and text only, no labels** |
| `labels.json`, `prompt.json` | The label list and the rendered prompt with its hash |
| `src/triage/` | Copies of `baseline.py` and `metrics.py`, so the notebook runs the repository's code |
| `bundle_manifest.json` | Data version, split hashes, prompt hash |

## 2. Upload the folder
With the Databricks CLI:
```bash
databricks workspace import-dir platforms/databricks/baseline /Workspace/Users/<you>/triage-baseline
```
Or in the workspace UI: create a folder, then import or drag in `baseline_zero_shot.py` and the `inputs` folder,
keeping `inputs/` next to the notebook. If the notebook cannot find `inputs/`, set its `base_dir` widget to the
folder path, for example `/Workspace/Users/<you>/triage-baseline`.

## 3. Run the notebook
Fill in the widgets, then **Run all**.

| Step | `splits` | `limit` | `prompt_frozen` | Purpose |
|---|---|---|---|---|
| a | `validation` | `50` | `no` | Check the endpoint, the answer format and the token counts |
| b | `validation` | `0` | `no` | Full validation run; look at accuracy, confusions, unparseable answers |
| c | `validation,test` | `0` | `yes` | Final run. Validation answers come from the cache, test is new |

- To change the prompt after step b, edit `configs/baseline/prompt.yaml` or `label_descriptions.yaml` in the
  repository, rebuild the bundle, upload `inputs/` again and repeat step b. Each prompt has its own cache entries.
- **Commit the prompt before step c.** A test run fixes the prompt (spec 03 section 4). The notebook refuses to
  run test with a limit or without `prompt_frozen = yes`.
- If the endpoint rejects `temperature` or needs more than 20 output tokens (reasoning models), change the
  `inputs` in the "model call" cell. For the token limit you can instead set `baseline.max_output_tokens` in
  `configs/base.yaml` and rebuild the bundle.
- The notebook shows a results table. Test accuracy is deliberately not shown there: the notebook has no test labels.

## 4. Bring the results back
Download the `outputs` folder:
```bash
databricks workspace export-dir /Workspace/Users/<you>/triage-baseline/outputs ./baseline_outputs
```
(or download the three files below from the workspace UI into one local folder). Then in the repository:
```bash
python -m triage.cli baseline-import --from ./baseline_outputs   # or: make baseline-import FROM=./baseline_outputs
```
The command checks that the prompt hash and data version match this checkout and that every split is complete,
then puts the files in place and logs a tracked run tagged `baseline=true`.

| File in `outputs/` | Where it goes | How |
|---|---|---|
| `baseline_predictions_test.jsonl` | `artifacts/baseline_predictions.jsonl` | `baseline-import` |
| `baseline_predictions_validation.jsonl` | `artifacts/baseline_predictions_validation.jsonl` | `baseline-import` |
| `baseline_summary.json` | `artifacts/baseline_summary.json` (with the Databricks and local run IDs) | `baseline-import` |
| `cache/responses.jsonl` | Stays on Databricks. Keep it: reruns are free while it exists | — |

Nothing under `artifacts/` or `baseline_outputs/` is committed to git.

## 5. Finish in the repository
1. In `configs/base.yaml` set `baseline.model` (the endpoint name), `baseline.usd_per_1m_input_tokens`,
   `baseline.usd_per_1m_output_tokens`, and `cost.baseline_usd_per_ticket` (printed by `baseline-import` as
   `usd_per_ticket` of the test split). Also set `cost.small_tickets_per_second` from the measured throughput of
   the predict run. Commit these with the task tick for T-05.
2. Run `python -m triage.cli predict` and `python -m triage.cli evaluate` (`make predict eval`).
   - `predict` fits the routing threshold on validation and writes `artifacts/threshold.json`.
   - `evaluate` writes `artifacts/report.json` with baseline accuracy, the cascade at the chosen threshold and
     the cascade curve.
3. Copy the baseline numbers into `docs/model_card.md` with the run ID of the evaluate run (project rule: no
   number without a tracked run).
