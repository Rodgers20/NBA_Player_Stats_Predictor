API_PORT   := 8000
EXPORT_DIR := frontend/public/data

.PHONY: api dev export build deploy test

# Run the FastAPI backend
api:
	uvicorn api.main:app --port $(API_PORT) --reload

# Run frontend dev server (expects API already running)
frontend:
	cd frontend && npm run dev

# Run both in parallel
dev:
	$(MAKE) -j2 api frontend

# Export all API responses to static JSON, then build the static site
export:
	@echo "→ Exporting API to $(EXPORT_DIR)..."
	@mkdir -p $(EXPORT_DIR)
	python scripts/export_api.py --out $(EXPORT_DIR) --api http://localhost:$(API_PORT)
	@echo "→ Building static Next.js site..."
	cd frontend && NEXT_EXPORT=1 npm run build
	@echo "✓ Static site ready in frontend/out/"

# Build without exporting (uses rewrites to live API)
build:
	cd frontend && npm run build

# Deploy to Vercel (assumes export was run)
deploy: export
	cd frontend && npx vercel --prod

# Run all tests
test:
	pytest -q
	cd frontend && npm test
