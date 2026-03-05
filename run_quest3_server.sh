#!/usr/bin/env bash
# Terminal 3 — Launch Quest 3 VR teleop manager
# Usage: bash run_quest3_server.sh [--vis-vr3pt] [--no-ssl] [...]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IP=$(hostname -I | awk '{print $1}')

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Quest 3 Teleop Server"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  PREREQUISITE — Quest 3 Guardian Setup (one-time):"
echo ""
echo "    If you haven't set up the play boundary yet:"
echo "    1. On Quest 3: Press Meta button > Settings > Physical Space"
echo "    2. Tap 'Set Floor' — point controller at the floor, confirm"
echo "    3. Tap 'Create Boundary' — draw your play area, confirm"
echo "    Without this, 'Start VR' will fail with a reference space error."
echo ""
echo "  ──────────────────────────────────────────────────────────"
echo ""
echo "  On the Quest 3 headset:"
echo ""
echo "    1. Make sure Quest 3 is on the SAME Wi-Fi as this machine"
echo "    2. Open Meta Quest Browser"
echo "    3. Go to:  https://${IP}:8443"
echo "    4. Accept the self-signed certificate (Advanced -> Proceed)"
echo "    5. Also visit https://${IP}:8765 and accept that cert too"
echo "    6. Go back to https://${IP}:8443"
echo "    7. Tap 'Connect WS' (status turns green)"
echo "    8. Tap 'Start VR' to begin streaming tracking data"
echo ""
echo "  ──────────────────────────────────────────────────────────"
echo ""
echo "  Controls (Quest 3 controllers):"
echo "    A+B+X+Y    = Start / Emergency Stop"
echo "    A+X        = Toggle VR 3PT (upper-body tracking)"
echo "    Left Stick  = Move direction"
echo "    Right Stick = Yaw / heading"
echo "    A+B / X+Y  = Next / Previous locomotion mode"
echo "    Triggers   = Hand grasp (in VR 3PT mode)"
echo "    Grips      = Hand grip (in VR 3PT mode)"
echo ""
echo "  Workstation IP: ${IP}"
echo ""
echo "══════════════════════════════════════════════════════════════"
echo ""

source .venv/bin/activate
python gear_sonic/scripts/quest3_manager_thread_server.py "$@"
