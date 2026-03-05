#!/usr/bin/env bash
# Terminal 2 — Launch C++ GEAR-SONIC deployment with ZMQ manager input (for Quest 3 teleop)
# Usage: bash deploy_sonic_with_zmq.sh [sim|real] [extra deploy.sh flags...]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/gear_sonic_deploy"

MODE="${1:-sim}"
shift 2>/dev/null || true

bash deploy.sh "$MODE" --input-type zmq_manager "$@"
