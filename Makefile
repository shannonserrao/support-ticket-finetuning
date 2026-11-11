PY ?= python
CONFIG ?= configs/base.yaml
CLI = $(PY) -m triage.cli

.PHONY: setup lint test verify-sources datatrain eval predict baseline-bundle baseline-import feedback-export feedback-ingest smoke release clean

setup:
	$(PY) -m pip install -e ".[dev,train]"

lint:
	$(PY) -m ruff check src tests

test:
	$(PY) -m pytest -q

verify-sources:
	$(PY) scripts/verify_sources.py --config $(CONFIG)

data:
	$(CLI) data --config $(CONFIG)

train:
	$(CLI) train --config $(CONFIG)

predict:
	$(CLI) predict --config $(CONFIG)

eval:
	$(CLI) evaluate --config $(CONFIG)

baseline-bundle:
	$(CLI) baseline-bundle --config $(CONFIG)

baseline-import:
	@test -n "$(FROM)" || (echo "usage: make baseline-import FROM=path/to/downloaded/outputs" && exit 1)
	$(CLI) baseline-import --config $(CONFIG) --from $(FROM)

feedback-export:
	$(CLI) feedback-export --config $(CONFIG)

feedback-ingest:
	$(CLI) feedback-ingest --config $(CONFIG) --reviewed feedback/reviewed.csv

smoke:
	$(CLI) data --config configs/smoke.yaml
	$(CLI) train --config configs/smoke.yaml
	$(CLI) predict --config configs/smoke.yaml
	$(CLI) evaluate --config configs/smoke.yaml

release:
	@test -n "$(VERSION)" || (echo "usage: make release VERSION=x.y.z" && exit 1)
	git tag -a v$(VERSION) -m "release v$(VERSION)"
	@echo "Tagged v$(VERSION). Push with: git push --tags. Then register the model (TASKS.md T-12)."

clean:
	rm -rf artifacts/* .pytest_cache .ruff_cache
