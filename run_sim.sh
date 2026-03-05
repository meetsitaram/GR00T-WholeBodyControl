#!/usr/bin/env bash
# Terminal 1 — Launch MuJoCo simulator for G1 29DOF
# Usage: bash run_sim.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure multicast is enabled on loopback (needed for CycloneDDS)
if ! ip link show lo | grep -q MULTICAST; then
    echo "Enabling multicast on loopback interface..."
    sudo ip link set lo multicast on
fi

source .venv/bin/activate
python gear_sonic/scripts/run_sim_loop.py "$@"
