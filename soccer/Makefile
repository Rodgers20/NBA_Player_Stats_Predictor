# FUTBALL — common tasks.
# Everything runs against the T7 data volume; each target checks it is mounted.

VENV := ./.venv/bin
DATA := /Volumes/T7/FUTBALL-data

.PHONY: data help mount unmount api web dev refresh refresh-now quality train ingest build test check snapshot

help:
	@echo "make data      check that the T7 data folder is available"
	@echo "make api       start the FastAPI data API on :8008"
	@echo "make web       start the Next.js dashboard on :3000"
	@echo "make refresh   pull latest results, rebuild lake, retrain models"
	@echo "make refresh-now  same, but ignore the football-data.org cache TTL"
	@echo "make quality   run the data quality suite"
	@echo "make train     refit the prediction models from the tuned config"
	@echo "make ablate    feature-group ablation (validation only)"
	@echo "make tune      full hyperparameter search, then refit"
	@echo "make check     typecheck + build + quality gate"

data:
	$(VENV)/python -c "from futball.config import require_volume; print(require_volume())"

mount: data

# Compatibility command: stop servers, then eject T7 in Finder.
unmount:
	@./scripts/unmount-t7.sh

api: data
	$(VENV)/python -m uvicorn futball.api.app:app --host 127.0.0.1 --port 8008 --reload

web:
	cd web && npm run dev

refresh: data
	$(VENV)/python -m futball.cli refresh --retrain

refresh-now: data
	$(VENV)/python -m futball.cli refresh --force

quality: data
	$(VENV)/python -m futball.quality

train: data
	$(VENV)/python -m futball.predict.train

ablate: data
	$(VENV)/python -u -m futball.predict.ablate

tune: data
	$(VENV)/python -u -m futball.predict.tune_all
	$(VENV)/python -u -m futball.predict.train

ingest: data
	$(VENV)/python -m futball.cli ingest --season 55:282

# Static snapshot for the deployed section: web/snapshot/ (see docs/snapshot-contract.md).
# Needs the T7 and provider keys in the environment (set -a; . .env; set +a).
snapshot: data
	$(VENV)/python scripts/export_snapshot.py $(SNAPSHOT_ARGS)

build:
	cd web && npm run build

test: data
	$(VENV)/python -m pytest -q

check: quality test
	cd web && npm run lint && npx tsc --noEmit && npm run build
