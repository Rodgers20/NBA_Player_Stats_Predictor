#!/usr/bin/env bash
# Compatibility wrapper: no disk image is needed.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -c "from futball.config import require_volume; print(require_volume())"
