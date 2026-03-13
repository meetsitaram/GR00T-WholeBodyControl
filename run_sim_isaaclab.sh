#!/usr/bin/env bash
# Terminal 1 — Launch Isaac Lab simulator for G1 29DOF (GEAR-SONIC)
# Usage: bash run_sim_isaaclab.sh [extra args...]
#
# This is the Isaac Lab equivalent of run_sim.sh (MuJoCo).
# Terminal 2 is unchanged: bash deploy_sonic.sh sim

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda env with Isaac Lab
eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# CycloneDDS + iceoryx libraries needed for unitree_sdk2py DDS
export CYCLONEDDS_HOME=/home/stickbot/projects/g1-pick-n-place/cyclonedds/install
ICEORYX_LIB=/home/stickbot/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib
export LD_LIBRARY_PATH="${ICEORYX_LIB}:${CYCLONEDDS_HOME}/lib:${LD_LIBRARY_PATH:-}"

# Restore terminal on exit (the sim uses raw mode for key input)
cleanup() { stty sane 2>/dev/null; echo; }
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  Isaac Lab GEAR-SONIC Simulator"
echo "  (drop-in replacement for MuJoCo run_sim.sh)"
echo "============================================================"
echo ""
echo "  Controls (this terminal):"
echo "    9         = toggle elastic band on/off"
echo "    7 / 8     = lower / raise band attachment"
echo "    Backspace = reset robot + re-enable band"
echo "    Ctrl+C    = quit"
echo ""

DISPLAY=:1 python gear_sonic/scripts/run_sim_isaaclab.py "$@"
