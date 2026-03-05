#!/usr/bin/env bash
# Terminal 2 — Launch C++ GEAR-SONIC deployment (sim mode by default)
# Usage: bash deploy_sonic.sh [sim|real]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/gear_sonic_deploy"

MODE="${1:-sim}"
shift 2>/dev/null || true

bash deploy.sh "$MODE" "$@"
