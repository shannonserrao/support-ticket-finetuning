# Support-Ticket Triage

Route customer-support messages to the right intent with a fine-tuned small encoder, escalating only uncertain tickets to an expensive LLM.

**Status:** scaffold. Specs, plan, tasks, diagrams and the hardware-independent modules are in place; data, training and inference are stubs to implement from `TASKS.md`. No results exist yet.

## What this proves
A proof of concept with three measurable claims, each tested against a zero-shot large-model baseline on the same held-out test set:
1. **Accuracy:** a fine-tuned small encoder matches or beats the baseline on intent classification.
2. **Cost and speed:** it does so at a small fraction of the cost per 1,000 tickets and of the latency.
3. **Routing:** a confidence threshold sends only the hard tickets to the expensive model, and the accuracy-versus-cost curve shows the trade-off.

Targets are in `SPEC.md` section 5. They are hypotheses until a tracked run confirms them.

## Start here
| Document | Purpose |
|---|---|
| `SPEC.md` | Requirements, metrics, release gates |
| `PLAN.md` | Architecture and technical choices |
| `docs/architecture.md` | Diagrams |
| `TASKS.md` | Ordered implementation tasks |
| `BUILD.md` | Run on local, Colab, Kaggle, Databricks |
| `CLAUDE.md` | Rules for AI-assisted development |
| `CONTRIBUTING.md` | Git, GitHub and GitLab, versioning, releases |
| `specs/` | Component specs |
| `docs/` | Data card, model card, tracking, reviewer guide, decisions |

## Quick start
```bash
make setup && make lint test     # works now
make data train predict eval     # works after T-03 to T-08
```

## Layout
```
SPEC.md PLAN.md TASKS.md BUILD.md CLAUDE.md CONTRIBUTING.md CHANGELOG.md
configs/      base, smoke and per-platform YAML
specs/        01-data ... 06-serving-and-routing
docs/         architecture, data card, model card, guides, decisions
src/triage/   config, data, train, predict, metrics, evaluate, tracking, feedback, cli
tests/        unit tests for the implemented modules
platforms/    Colab, Kaggle and Databricks launchers
data/ artifacts/ feedback/
```
