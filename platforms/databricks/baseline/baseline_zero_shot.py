# Databricks notebook source
# MAGIC %md
# MAGIC # Zero-shot baseline for Support-Ticket Triage (task T-05)
# MAGIC
# MAGIC Classifies the validation and test tickets with an expensive model behind a Databricks model serving
# MAGIC endpoint, shows the results here and writes result files to `outputs/` next to this notebook.
# MAGIC
# MAGIC **Before running:** this folder must contain `inputs/`, created in the repository with
# MAGIC `python -m triage.cli baseline-bundle`. See `README.md` in this folder for the full procedure.
# MAGIC
# MAGIC **How to use it**
# MAGIC 1. Fill in the widgets at the top. Start with `splits = validation` and `limit = 50` to check the endpoint.
# MAGIC 2. Run all cells. Look at the validation results and the unparseable answers.
# MAGIC 3. When the prompt is final and committed in the repository, set `limit = 0` and run validation in full.
# MAGIC 4. Then set `splits = validation,test` and `prompt_frozen = yes` and run again. Answers are cached, so
# MAGIC    the validation tickets are not paid for twice.
# MAGIC
# MAGIC Test tickets are sent **without labels**. Test accuracy is computed only by `make eval` in the repository.

# COMMAND ----------

dbutils.widgets.text("endpoint", "", "1. Serving endpoint name")
dbutils.widgets.text("usd_per_1m_input_tokens", "", "2. USD per 1M input tokens")
dbutils.widgets.text("usd_per_1m_output_tokens", "", "3. USD per 1M output tokens")
dbutils.widgets.dropdown("splits", "validation", ["validation", "validation,test"], "4. Splits to run")
dbutils.widgets.text("limit", "50", "5. Rows per split (0 = all)")
dbutils.widgets.dropdown("prompt_frozen", "no", ["no", "yes"], "6. Prompt frozen (needed for test)")
dbutils.widgets.text("max_workers", "4", "7. Parallel requests")
dbutils.widgets.text("base_dir", "", "8. Folder of this notebook (blank = current)")
dbutils.widgets.text("output_dir", "", "9. Output folder (blank = <folder>/outputs)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the bundle and check the settings

# COMMAND ----------

import html
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

BASE_DIR = dbutils.widgets.get("base_dir").strip() or os.getcwd()
INPUTS = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR = dbutils.widgets.get("output_dir").strip() or os.path.join(BASE_DIR, "outputs")
if not os.path.exists(os.path.join(INPUTS, "bundle_manifest.json")):
    raise FileNotFoundError(
        f"{INPUTS}/bundle_manifest.json not found. Build the bundle in the repository with "
        "`python -m triage.cli baseline-bundle`, upload the whole folder, and if this notebook is not in that "
        "folder set the `base_dir` widget to the folder's path."
    )
sys.path.insert(0, os.path.join(INPUTS, "src"))
from triage import baseline as B  # noqa: E402  (the same module the repository uses)


def _load(name):
    with open(os.path.join(INPUTS, name), encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(name):
    with open(os.path.join(INPUTS, name), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _price(name):
    raw = dbutils.widgets.get(name).strip()
    return float(raw) if raw else None


BUNDLE, LABELS, PROMPT = _load("bundle_manifest.json"), _load("labels.json"), _load("prompt.json")
ROWS = {"validation": _load_jsonl("validation.jsonl"), "test": _load_jsonl("test.jsonl")}
ENDPOINT = dbutils.widgets.get("endpoint").strip()
PRICE_IN, PRICE_OUT = _price("usd_per_1m_input_tokens"), _price("usd_per_1m_output_tokens")
SPLITS = [s for s in dbutils.widgets.get("splits").split(",") if s]
LIMIT = int(dbutils.widgets.get("limit") or 0)
MAX_WORKERS = int(dbutils.widgets.get("max_workers") or 4)
MAX_TOKENS = int(BUNDLE.get("max_output_tokens", 20))

if not ENDPOINT:
    raise ValueError("Set the `endpoint` widget to the name of your model serving endpoint.")
if PROMPT["hash"] != BUNDLE["prompt_hash"]:
    raise ValueError("inputs/prompt.json does not match inputs/bundle_manifest.json; rebuild and re-upload the bundle.")
if "test" in SPLITS and (dbutils.widgets.get("prompt_frozen") != "yes" or LIMIT):
    raise ValueError("The test split runs only in full (limit = 0) and with prompt_frozen = yes. "
                     "Develop the prompt on validation first; a test run fixes the prompt for good.")
if PRICE_IN is None or PRICE_OUT is None:
    print("WARNING: token prices are not set, so no cost will be reported. Fill in widgets 2 and 3.")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"endpoint      : {ENDPOINT}")
print(f"data version  : {BUNDLE['data_version']}  (validation {len(ROWS['validation'])}, test {len(ROWS['test'])})")
print(f"labels        : {len(LABELS)}")
print(f"prompt hash   : {PROMPT['hash'][:16]}...")
print(f"splits / limit: {SPLITS} / {LIMIT or 'all rows'}")
print(f"output folder : {OUTPUT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The model call
# MAGIC This is the only cell that depends on the endpoint. It assumes a chat endpoint (Foundation Model APIs,
# MAGIC external models, or a custom chat model). If your endpoint rejects `temperature` or needs more output
# MAGIC tokens (reasoning models), change `inputs` here; nothing else needs to change.

# COMMAND ----------

from mlflow.deployments import get_deploy_client  # noqa: E402

_client = get_deploy_client("databricks")


def call_model(messages):
    response = _client.predict(
        endpoint=ENDPOINT,
        inputs={"messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0.0},
    )
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, list):  # some models return a list of content parts
        content = "".join(part.get("text", "") for part in content
                          if isinstance(part, dict) and part.get("type", "text") == "text")
    usage = response.get("usage") or {}
    return {"text": content, "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0)}


_probe = call_model(B.build_messages(PROMPT, ROWS["validation"][0]["text"]))
print("message :", ROWS["validation"][0]["text"])
print("answer  :", repr(_probe["text"]), "->", B.parse_label(_probe["text"], LABELS))
print("tokens  :", _probe["input_tokens"], "in /", _probe["output_tokens"], "out")
if not _probe["input_tokens"]:
    print("WARNING: the endpoint returned no token usage, so cost will be reported as zero.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run
# MAGIC Answers are cached in `outputs/cache/responses.jsonl` by endpoint, prompt hash and ticket id.
# MAGIC Rerunning this cell costs nothing for tickets that already have an answer.

# COMMAND ----------

CACHE = B.ResponseCache(os.path.join(OUTPUT_DIR, "cache", "responses.jsonl"))
RECORDS, SUMMARY = {}, {}


def _progress(done, total):
    if done % 100 == 0 or done == total:
        print(f"  {done}/{total}")


for split in SPLITS:
    rows = ROWS[split][:LIMIT] if LIMIT else ROWS[split]
    print(f"{split}: {len(rows)} tickets")
    RECORDS[split] = B.run_baseline(rows, LABELS, PROMPT, ENDPOINT, call_model, CACHE, PRICE_IN, PRICE_OUT,
                                    max_workers=MAX_WORKERS, on_progress=_progress)
    SUMMARY[split] = B.summarise(RECORDS[split], rows if split == "validation" else None)
    name = B.predictions_file(split) + (".partial" if LIMIT else "")
    with open(os.path.join(OUTPUT_DIR, name), "w", encoding="utf-8") as handle:
        for record in RECORDS[split]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  wrote {name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Results

# COMMAND ----------


def _table(title, header, rows):
    cells = "".join(f"<th style='text-align:left;padding:4px 12px'>{html.escape(str(h))}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td style='padding:4px 12px'>{html.escape(str(c))}</td>" for c in row) + "</tr>"
                   for row in rows)
    displayHTML(f"<h4>{html.escape(title)}</h4><table><tr>{cells}</tr>{body}</table>")


def _fmt(value, digits=4):
    return "n/a" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


_table(
    f"Baseline `{ENDPOINT}` - prompt {PROMPT['hash'][:12]} - {BUNDLE['data_version']}"
    + ("  (PARTIAL RUN, not importable)" if LIMIT else ""),
    ["split", "tickets", "accuracy", "macro F1", "unparseable", "failed calls", "input tokens", "output tokens",
     "USD per ticket", "USD total", "USD spent this run"],
    [[split, s["n"], _fmt(s.get("accuracy")) if split == "validation" else "computed by `make eval`",
      _fmt(s.get("macro_f1")) if split == "validation" else "computed by `make eval`",
      f"{s['unparseable']} ({_fmt(s['unparseable_rate'])})", s["failed_calls"], s["input_tokens"],
      s["output_tokens"], _fmt(s["usd_per_ticket"], 6), _fmt(s["cost_usd_total"]), _fmt(s["spent_usd_this_run"])]
     for split, s in SUMMARY.items()],
)

if "validation" in RECORDS:
    _gold = {r["id"]: r for r in ROWS["validation"]}
    _wrong = Counter((_gold[r["id"]]["label"], r["label"] or "(unparseable)") for r in RECORDS["validation"]
                     if r["label"] != _gold[r["id"]]["label"])
    _table("Validation: most frequent confusions (use these to improve the label descriptions)",
           ["gold label", "baseline answer", "count"], [[g, p, n] for (g, p), n in _wrong.most_common(15)])
    _bad = [r for r in RECORDS["validation"] if r["unparseable"]][:10]
    if _bad:
        _table("Validation: examples of unparseable answers", ["id", "raw answer or error"],
               [[r["id"], r["raw"] if r["raw"] is not None else r.get("error")] for r in _bad])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save the summary and log the run

# COMMAND ----------

RESULT = {
    "model": ENDPOINT, "prompt_hash": PROMPT["hash"], "data_version": BUNDLE["data_version"],
    "usd_per_1m_input_tokens": PRICE_IN, "usd_per_1m_output_tokens": PRICE_OUT,
    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "complete": not LIMIT, "limit": LIMIT, "splits": SUMMARY, "mlflow_run_id": None,
}
SUMMARY_PATH = os.path.join(OUTPUT_DIR, B.SUMMARY_FILE)


def _save_summary():
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(RESULT, handle, indent=2)


_save_summary()
try:
    import mlflow

    with mlflow.start_run(run_name=f"baseline-{ENDPOINT}") as _run:
        RESULT["mlflow_run_id"] = _run.info.run_id
        _save_summary()
        mlflow.set_tags({"baseline": "true", "stage": "dev", "complete": str(not LIMIT).lower()})
        mlflow.log_params({"baseline.model": ENDPOINT, "baseline.prompt_hash": PROMPT["hash"],
                           "data_version": BUNDLE["data_version"], "limit": LIMIT,
                           "usd_per_1m_input_tokens": PRICE_IN, "usd_per_1m_output_tokens": PRICE_OUT})
        for _split, _stats in SUMMARY.items():
            mlflow.log_metrics({f"baseline_{_split}_{k}": float(v) for k, v in _stats.items()
                                if isinstance(v, (int, float))})
        mlflow.log_artifact(SUMMARY_PATH)
    print("MLflow run:", RESULT["mlflow_run_id"])
except Exception as exc:  # the result files matter more than the tracking entry
    print(f"WARNING: could not log to MLflow ({type(exc).__name__}: {exc}). The result files are still complete.")

print(f"\nResult files in {OUTPUT_DIR}:")
for _name in sorted(os.listdir(OUTPUT_DIR)):
    print("  ", _name)
if LIMIT:
    print("\nThis was a PARTIAL run (limit > 0). It cannot be imported. Set limit = 0 for the real run.")
else:
    _missing = [s for s in ("validation", "test") if s not in SUMMARY]
    print("\nNext steps (details in README.md):")
    if _missing:
        print(f"  - Still to run: {_missing}. Set splits = validation,test and prompt_frozen = yes.")
    print("  1. Download this `outputs` folder to your machine.")
    print("  2. In the repository: python -m triage.cli baseline-import --from <downloaded outputs folder>")
    print("  3. Then: python -m triage.cli predict  and  python -m triage.cli evaluate")
