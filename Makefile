PYTHON ?= python3
API_PORT ?= 8000
FRONTEND_PORT ?= 3001
EXPORT_DIR ?= frontend/public/data

.PHONY: install api frontend dev app export build test
install:
	$(PYTHON) -m pip install -r requirements.txt -r api/requirements.txt
	cd frontend && npm ci
api:
	$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port $(API_PORT) --reload
frontend:
	cd frontend && API_BACKEND_URL=http://127.0.0.1:$(API_PORT) npm run dev -- --hostname 127.0.0.1 --port $(FRONTEND_PORT)
app: dev
dev:
	$(MAKE) -j2 api frontend
export:
	$(PYTHON) scripts/export_api.py --out $(EXPORT_DIR) --api http://127.0.0.1:$(API_PORT)
	cd frontend && NEXT_EXPORT=1 npm run build
build:
	cd frontend && npm run build
test:
	$(PYTHON) -m pytest -q
	cd frontend && npm test && npm run lint && npx tsc --noEmit
