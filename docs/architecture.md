# Architecture — Support-Ticket Triage

The diagrams are Mermaid and render on GitHub and GitLab.

## 1. Training and evaluation pipeline
What `make data train predict eval` does, and what each step writes.

```mermaid
flowchart LR
    subgraph Sources
        HF[(Banking77<br/>pinned revision)]
        COR[(feedback/<br/>corrections.jsonl)]
    end
    subgraph Data["make data"]
        D1[Download] --> D2[Stratified split<br/>train / validation / test]
        D2 --> D3[Checks: leakage,<br/>label coverage]
        D3 --> MAN[data/manifest.json<br/>data-vN + SHA-256]
    end
    subgraph Train["make train"]
        T1[Fine-tune encoder] --> T2[Early stopping on<br/>validation macro F1]
        T2 --> T3[Temperature scaling<br/>on validation]
    end
    subgraph Eval["make predict / eval"]
        P1[Batch inference on test] --> P2[predictions.jsonl<br/>label, confidence, latency]
        P2 --> E1[Metrics + cascade curve<br/>+ business projection]
        BL[Zero-shot baseline<br/>expensive model, cached] --> E1
    end
    HF --> D1
    COR --> D2
    MAN --> T1
    T3 --> P1
    E1 --> REP[artifacts/report.json]
    T1 -. params, loss .-> ML[(MLflow)]
    E1 -. metrics, report .-> ML
    T3 -. weights .-> REG[(Model registry)]
```

## 2. Inference path: the cascade
How one ticket is handled at run time. The small model answers when it is confident; otherwise the ticket is escalated.

```mermaid
flowchart TD
    IN[Incoming ticket] --> ENC[Fine-tuned encoder<br/>in-house, milliseconds]
    ENC --> CAL[Calibrated confidence]
    CAL --> Q{"confidence at or above<br/>threshold?"}
    Q -- yes --> OUT1[Return label<br/>route = small]
    Q -- no --> BIG[Expensive model<br/>or human queue]
    BIG --> OUT2[Return label<br/>route = escalate]
    OUT1 --> LOG[(predictions_log.jsonl)]
    OUT2 --> LOG
    LOG --> FB[Human review queue]
```

## 3. Human feedback loop

```mermaid
flowchart LR
    LOG[(predictions_log.jsonl)] --> SEL[Select: all low-confidence<br/>+ random audit sample]
    SEL --> CSV[review_queue.csv]
    CSV --> HUM[Reviewer fills verdict<br/>and correction]
    HUM --> ING[make feedback-ingest<br/>validates rows]
    ING --> COR[(corrections.jsonl<br/>committed to git)]
    COR --> DATA[make data<br/>new data version]
    DATA --> TRAIN[Retrain: challenger]
    ING --> AUD[Audit agreement rate]
    AUD --> GATE{Promotion gate met?}
    TRAIN --> GATE
    GATE -- yes --> CH[Promote to champion]
    GATE -- no --> KEEP[Keep current champion]
```

## 4. Versioning and release flow

```mermaid
flowchart LR
    DEV[Feature branch] --> MR[Merge request]
    MR --> CI[CI on GitHub / GitLab:<br/>lint + unit tests]
    CI --> MAIN[main]
    MAIN --> RUN[Tracked release run<br/>git SHA + data-vN + config]
    RUN --> GATES{Release gates<br/>SPEC section 5}
    GATES -- pass --> TAG[git tag vX.Y.Z]
    TAG --> REG[Register model-X.Y.Z<br/>alias: challenger]
    REG --> REV[Human audit] --> CHAMP[alias: champion]
    GATES -- fail --> DEV
```

## 5. Where it runs

```mermaid
flowchart TB
    REPO[(Git repo:<br/>code, configs, specs, manifest)]
    REPO --> L[Local<br/>configs/base.yaml]
    REPO --> C[Colab<br/>platform/colab.yaml<br/>artifacts on Drive]
    REPO --> K[Kaggle<br/>platform/kaggle.yaml<br/>artifacts in kernel output]
    REPO --> B[Databricks<br/>platform/databricks.yaml<br/>managed MLflow]
    L & C & K & B --> SAME[Same CLI:<br/>data, train, predict, evaluate]
```
