#!/bin/bash
set -e

# ============================================================================
# X2 Ultra Deploy - Deployment Script
# ============================================================================
# Companion to deploy.sh (which targets G1). End-to-end build + pre-flight +
# launch for the agi_x2_deploy_onnx_ref ROS 2 package on AgiBot X2 Ultra.
#
# Topologies (per docs/source/user_guide/x2_sonic_deploy_real.md):
#   local  - Build and run on this machine (your laptop). DDS auto-discovers
#            the robot via the wired SDK ethernet (laptop NIC at 10.0.1.2/24,
#            PC2 dev unit at 10.0.1.41).
#   onbot  - rsync this package + sonic_common to PC2, build and run there.
#            Eliminates ethernet jitter; recommended for production.
#   sim    - Build + run locally against a MuJoCo physics sim, on isolated
#            loopback DDS (ROS_LOCALHOST_ONLY=1 + private ROS_DOMAIN_ID).
#            The sibling Python bridge scripts/x2_mujoco_ros_bridge.py is
#            launched in the background; it steps physics at 1 kHz, publishes
#            joint state + IMU on the same /aima/* topics the deploy
#            subscribes to, and applies the deploy's PD commands as MuJoCo
#            torques. This is the X2 analogue of G1's sim mode (which
#            selects a loopback DDS interface and pairs with the
#            unitree_sdk2py MuJoCo bridge).
#
# Local/onbot pre-flight:
#   1. Ping PC2 to confirm we can reach the robot.
#   2. Verify ROS 2 topics are visible (joint state for each group + IMU).
#   3. Stop the MC module by POSTing stop_app to PC1's Environment Manager
#      HTTP API (the same mechanism `aima em stop-app mc` uses underneath),
#      so HAL joint commands take effect. Use --no-stop-mc to skip.
#
# Sim pre-flight:
#   1. Verify the bridge script + MJCF + Python deps are importable.
#   2. Isolate DDS to loopback so a sim run on the SDK subnet cannot fight a
#      real robot (sets ROS_LOCALHOST_ONLY=1 and ROS_DOMAIN_ID).
#
# Usage:
#   ./deploy_x2.sh [OPTIONS] [local|onbot|sim]
#
# Examples:
#   ./deploy_x2.sh --model /opt/x2_models/model_step_016000_g1.onnx --dry-run
#   ./deploy_x2.sh --model ./model.onnx --motion ./standing.x2m2 onbot
#   ./deploy_x2.sh sim --model ./model.onnx --sim-viewer --autostart-after 5
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PKG_NAME="agi_x2_deploy_onnx_ref"
PKG_DIR_REL="src/x2/agi_x2_deploy_onnx_ref"
SONIC_COMMON_REL="src/common"

# ============================================================================
# Defaults
# ============================================================================

MODE_DEFAULT="local"
ROBOT_HOST_DEFAULT="10.0.1.41"
ROBOT_USER_DEFAULT="agi"
ONBOT_WS_DEFAULT="\$HOME/x2_deploy_ws"
ONNXRUNTIME_ROOT_DEFAULT="/opt/onnxruntime"
SIM_DOMAIN_ID_DEFAULT="73"
# PC1 (10.0.1.40) Environment Manager HTTP API. This is the underlying
# mechanism `aima em start-app/stop-app` uses; talking to it directly avoids
# requiring an ssh key into the robot and works from inside docker as long
# as the host has a route to 10.0.1.40 (true once enp10s0 is on 10.0.1.2/24
# and `network_mode: host` is in effect). Vetted in
# agitbot-x2-record-and-replay/src/x2_recorder/mc_control.py.
MC_EM_URL_DEFAULT="http://10.0.1.40:50080"

MODE="$MODE_DEFAULT"
ROBOT_HOST="$ROBOT_HOST_DEFAULT"
ROBOT_USER="$ROBOT_USER_DEFAULT"
ONBOT_WS="$ONBOT_WS_DEFAULT"
ONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT_DEFAULT"
MC_EM_URL="$MC_EM_URL_DEFAULT"
# Set to true once we've successfully POSTed stop_app, so the cleanup trap
# knows it has to POST start_app on exit. Never set true if MC was already
# down when we started.
MC_STOPPED_BY_US=false

# Deploy CLI passthrough flags
MODEL=""
MOTION=""
LOG_DIR=""
AUTOSTART=""
# Auto-shutdown N seconds after entering CONTROL state. Empty string =
# unbounded (run until Ctrl-C). Useful for bounded dry-run smoke tests where
# you don't want to babysit Ctrl-C.
MAX_DURATION=""
TILT_COS=""
RAMP_SECONDS=""
# Per-joint hard clamp on |target - default_angles|, in radians. Empty string
# = leave it disabled (legacy behaviour). Recommended for first powered runs:
# --max-target-dev 0.05 (about 3 deg). See policy_parameters.hpp for the
# trained standing pose this clamps around.
MAX_TARGET_DEV=""
# Symmetric clip on the raw ONNX action (action_il) BEFORE x2_action_scale.
# Empty string = let the C++ binary use its compiled-in default of 20.0,
# which matches IsaacLab training-time config.action_clip_value. Set to a
# negative number to disable (only useful for parity-vs-old-behavior tests).
ACTION_CLIP=""
# Soft-EXIT ramp duration. When --max-duration trips, lerp target_pos back
# from the last policy command to default_angles over this many seconds
# before shutting down. Empty = let the C++ binary use its compiled-in
# default of 2.0s; set "0" to disable (legacy immediate-shutdown). See
# x2_deploy_onnx_ref.cpp::CliArgs::return_seconds for the gory details.
RETURN_SECONDS=""
IMU_TOPIC=""
INTRA_OP_THREADS=""
# Optional one-shot debug capture: write the first CONTROL-tick obs (tokenizer
# 680 + proprioception 990 + raw policy output + robot state) to PATH and
# exit immediately. See compare_deploy_vs_isaaclab_obs.py for analysis.
OBS_DUMP=""
DRY_RUN=false

# Behaviour toggles
NO_STOP_MC=false
NO_CONFIRM=false
NO_BUILD=false
BUILD_ONLY=false

# Sim mode (MuJoCo bridge is the only sim driver)
SIM_DOMAIN_ID="$SIM_DOMAIN_ID_DEFAULT"
SIM_BRIDGE_REL="scripts/x2_mujoco_ros_bridge.py"
SIM_PYTHON="${SIM_PYTHON:-python3}"
SIM_MJCF=""
SIM_MOTION=""
SIM_INIT_FRAME=""
SIM_VIEWER=false
SIM_IMU_FROM=""
SIM_HOLD_STIFFNESS_MULT=""
SIM_NO_ELASTIC_BAND=false
SIM_BAND_RELEASE_AFTER_S=""
SIM_DT=""
SIM_PRINT_SCENE=false
SIM_RECORD_COMMANDS=""

# Bookkeeping for child PIDs we must clean up on exit
SIM_BRIDGE_PID=""
SIM_RECORD_PID=""

# ============================================================================
# Usage
# ============================================================================

show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [local|onbot|sim]

Deploy the agi_x2_deploy_onnx_ref ROS 2 node onto an AgiBot X2 Ultra.

Modes:
  local           Build + run on this machine; talk to robot via DDS over
                  the wired SDK ethernet (default).
  onbot           rsync + build + run on the robot's PC2 development unit
                  via ssh. Recommended for production.
  sim             Build + run locally on isolated loopback DDS, paired with
                  the MuJoCo bridge (scripts/x2_mujoco_ros_bridge.py)
                  launched as a background child. Bridge steps physics at
                  1 kHz, publishes joint state + IMU on the same /aima
                  topics the deploy subscribes to, and applies the deploy's
                  PD commands as MuJoCo torques. Closed-loop. Mirrors what
                  G1's sim mode does via unitree_sdk2py_bridge.

Required:
  --model PATH                Fused g1+g1_dyn ONNX (e.g. *_g1.onnx)

Optional deploy flags (forwarded to ros2 run):
  --motion PATH               X2M2 reference motion file (default: StandStill)
  --log-dir PATH              Per-tick CSV log directory
  --autostart-after SECONDS         Auto-transition WAIT->CONTROL after N seconds
                              (default: -1, wait for stdin 'go')
  --max-duration SECONDS      Auto-shutdown N seconds after entering CONTROL
                              (default: unbounded). Useful for bounded dry-run
                              smoke tests so the operator isn't expected to
                              babysit Ctrl-C.
  --dry-run                   Publish stiffness=0/damping=0 (no torque).
                              MANDATORY for first power-on on the real robot.
  --tilt-cos COS              Tilt watchdog threshold (default: -0.3)
  --ramp-seconds SECONDS      Soft-start ramp duration (default: 2.0)
  --max-target-dev RAD        Per-joint hard clamp on |target - default_angles|,
                              in radians. Negative/omitted = disabled. Use a
                              small value (e.g. 0.05 ~= 3 deg) for first
                              powered bring-up runs so a divergent policy or
                              obs-construction bug cannot drive any joint
                              more than RAD away from the trained standing
                              pose, regardless of what the ONNX session emits.
  --action-clip RAD           Symmetric clip on the raw ONNX action (action_il)
                              BEFORE x2_action_scale. Default in the C++ binary
                              is 20.0, matching the training-time
                              config.action_clip_value in
                              gear_sonic/config/manager_env/base_env.yaml.
                              Pass a negative value to disable (parity tests
                              only). Without this clip a saturated policy
                              produces O(100 rad) targets which the deploy
                              safety stack truncates -- silently breaking
                              parity with what training observed.
  --return-seconds SECONDS    Soft-EXIT ramp duration (default 2.0). When
                              --max-duration trips, lerp target_pos from the
                              last policy command back to default_angles over
                              SECONDS (deploy-mode kp/kd active) before
                              shutdown. Prevents MC from snapping joints back
                              at handoff (which can red-fault the X2 Ultra
                              MC unit if the policy left limbs mid-motion --
                              e.g. arm extended for take_a_sip). Set 0 to
                              disable (legacy immediate-shutdown).
  --imu-topic NAME            Override IMU topic; use this if the firmware
                              ships with the SDK-example typo
                              (/aima/hal/imu/torse/state)
  --intra-op-threads N        ONNX session threads (default: 1)
  --obs-dump PATH             DEBUG: capture the first CONTROL-tick inference
                              payload (tokenizer + proprioception + raw action
                              + robot state) to PATH and exit. Pair with
                              --dry-run + --autostart-after for a deterministic
                              snapshot from a known robot pose. Diff against
                              IsaacLab GT with
                              gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py

Robot connection (onbot mode):
  --robot-host HOST           PC2 IP/hostname (default: $ROBOT_HOST_DEFAULT)
  --robot-user USER           PC2 ssh user (default: $ROBOT_USER_DEFAULT)
  --onbot-ws PATH             colcon workspace on PC2 (default: $ONBOT_WS_DEFAULT)

MC stop / restart (local + onbot modes):
  --mc-em-url URL             PC1 Environment Manager HTTP API URL.
                              We POST {stop,start}_app to {URL}/json/...
                              instead of using ssh. Reachable from inside
                              docker_x2/ as long as the host can route to
                              10.0.1.40. Default: $MC_EM_URL_DEFAULT

Sim mode (only applies when 'sim' is selected; all optional):
  --sim-mjcf PATH             Override MJCF path (default: x2_ultra.xml).
  --sim-motion PATH           RSI from a motion-lib PKL (default: stand pose).
  --sim-init-frame N          Motion frame to RSI from (default 0).
  --sim-viewer                Open the MuJoCo passive viewer window.
  --sim-imu-from {pelvis,torso}
                              Body to read IMU from (default: pelvis,
                              matches MJCF live sensor at imu_0).
  --sim-hold-stiffness-mult X
                              Multiplier on policy_parameters.kps used to
                              hold the default standing pose BEFORE the
                              deploy connects (default: 1.0).
  --sim-no-elastic-band       Disable the virtual ElasticBand. The band is
                              ON by default and hangs the pelvis from world
                              [0,0,1] so the robot stays upright while the
                              policy spins up. With viewer, press 9 to drop;
                              headless, see --sim-band-release-after-s.
  --sim-band-release-after-s SECS
                              When headless, auto-release the band SECS
                              seconds after the deploy's first command.
                              Default 1.0. Negative = never (unsafe).
  --sim-dt SECS               Physics step (default: 0.001 = 1 kHz).
  --sim-print-scene           Dump bodies/joints/actuators/sensors on start.
  --sim-python PATH           Python interpreter for the bridge
                              (default: \$SIM_PYTHON or python3).
  --sim-record-commands PATH  Record the deploy's command topics to this
                              rosbag2 directory (handy for diffing sim/real).
  --sim-domain-id N           ROS_DOMAIN_ID to isolate the sim from any real
                              robot on the same subnet (default: $SIM_DOMAIN_ID_DEFAULT)

Pre-flight + behaviour toggles:
  --no-stop-mc                Skip the stop_app POST (assume MC is already
                              stopped, or you're using JOINT_DEFAULT mode).
                              The cleanup trap that restarts MC on exit is
                              also skipped in that case.
  --no-confirm                Skip the final "proceed?" prompt (for CI)
  --no-build                  Skip the colcon build step (use the existing
                              install/ tree as-is)
  --build-only                Build, don't run

ONNX Runtime:
  --onnxruntime-root PATH     ORT install prefix (default: $ONNXRUNTIME_ROOT_DEFAULT)

  -h, --help                  Show this help

Examples:
  # Bring-up dry run from your laptop (most common first command):
  $0 --model /opt/x2_models/model_step_016000_g1.onnx --dry-run --autostart-after 5

  # On-bot powered run with operator gate (after dry-run looks clean):
  $0 onbot \\
      --model /opt/x2_models/model_step_016000_g1.onnx \\
      --motion /opt/x2_motions/standing.x2m2 \\
      --log-dir /tmp/x2_powered_\$(date +%Y%m%d_%H%M%S)

  # Closed-loop sim in MuJoCo with viewer (no robot needed):
  $0 sim \\
      --model /opt/x2_models/model_step_016000_g1.onnx \\
      --sim-viewer --autostart-after 5

  # Just rebuild, no robot, no sim:
  $0 --model dummy --build-only --no-stop-mc local

For full background, see:
  docs/source/user_guide/x2_sonic_deploy_real.md
  docs/source/user_guide/x2_first_real_robot.md
  docs/source/references/x2_deployment_code.md
EOF
}

# ============================================================================
# Parse arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_usage; exit 0 ;;
        --model)              MODEL="$2"; shift 2 ;;
        --motion)             MOTION="$2"; shift 2 ;;
        --log-dir)            LOG_DIR="$2"; shift 2 ;;
        --autostart-after)          AUTOSTART="$2"; shift 2 ;;
        --max-duration)       MAX_DURATION="$2"; shift 2 ;;
        --tilt-cos)           TILT_COS="$2"; shift 2 ;;
        --ramp-seconds)       RAMP_SECONDS="$2"; shift 2 ;;
        --max-target-dev)     MAX_TARGET_DEV="$2"; shift 2 ;;
        --action-clip)        ACTION_CLIP="$2"; shift 2 ;;
        --return-seconds)     RETURN_SECONDS="$2"; shift 2 ;;
        --imu-topic)          IMU_TOPIC="$2"; shift 2 ;;
        --intra-op-threads)   INTRA_OP_THREADS="$2"; shift 2 ;;
        --obs-dump)           OBS_DUMP="$2"; shift 2 ;;
        --dry-run)            DRY_RUN=true; shift ;;
        --robot-host)         ROBOT_HOST="$2"; shift 2 ;;
        --robot-user)         ROBOT_USER="$2"; shift 2 ;;
        --onbot-ws)           ONBOT_WS="$2"; shift 2 ;;
        --onnxruntime-root)   ONNXRUNTIME_ROOT="$2"; shift 2 ;;
        --mc-em-url)          MC_EM_URL="$2"; shift 2 ;;
        --no-stop-mc)         NO_STOP_MC=true; shift ;;
        --no-confirm)         NO_CONFIRM=true; shift ;;
        --no-build)           NO_BUILD=true; shift ;;
        --build-only)         BUILD_ONLY=true; shift ;;
        --sim-mjcf)               SIM_MJCF="$2"; shift 2 ;;
        --sim-motion)             SIM_MOTION="$2"; shift 2 ;;
        --sim-init-frame)         SIM_INIT_FRAME="$2"; shift 2 ;;
        --sim-viewer)             SIM_VIEWER=true; shift ;;
        --sim-imu-from)           SIM_IMU_FROM="$2"; shift 2 ;;
        --sim-hold-stiffness-mult) SIM_HOLD_STIFFNESS_MULT="$2"; shift 2 ;;
        --sim-no-elastic-band)    SIM_NO_ELASTIC_BAND=true; shift ;;
        --sim-band-release-after-s) SIM_BAND_RELEASE_AFTER_S="$2"; shift 2 ;;
        --sim-dt)                 SIM_DT="$2"; shift 2 ;;
        --sim-print-scene)        SIM_PRINT_SCENE=true; shift ;;
        --sim-python)             SIM_PYTHON="$2"; shift 2 ;;
        --sim-record-commands)    SIM_RECORD_COMMANDS="$2"; shift 2 ;;
        --sim-domain-id)          SIM_DOMAIN_ID="$2"; shift 2 ;;
        local|onbot|sim)      MODE="$1"; shift ;;
        *)
            echo -e "${RED}Error: unknown argument: $1${NC}" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

if [[ -z "$MODEL" ]]; then
    echo -e "${RED}Error: --model is required${NC}" >&2
    echo "Run '$0 --help' for usage." >&2
    exit 1
fi

if [[ "$MODE" != "local" && "$MODE" != "onbot" && "$MODE" != "sim" ]]; then
    echo -e "${RED}Error: mode must be one of: local, onbot, sim (got '$MODE')${NC}" >&2
    exit 1
fi

# Resolve absolute paths for local artefacts so onbot rsync works.
abspath() {
    local p="$1"
    if [[ -z "$p" ]]; then
        echo ""
    elif [[ "$p" = /* ]]; then
        echo "$p"
    else
        # Resolve relative to caller's CWD (which is now SCRIPT_DIR).
        echo "$(cd "$(dirname "$p")" 2>/dev/null && pwd)/$(basename "$p")"
    fi
}

if [[ "$MODE" == "local" || "$MODE" == "sim" ]]; then
    [[ -n "$MODEL" ]] && MODEL="$(abspath "$MODEL")"
    [[ -n "$MOTION" ]] && MOTION="$(abspath "$MOTION")"
    [[ -n "$LOG_DIR" ]] && LOG_DIR="$(abspath "$LOG_DIR")"
fi

if [[ "$MODE" == "sim" ]]; then
    [[ -n "$SIM_MJCF" ]] && SIM_MJCF="$(abspath "$SIM_MJCF")"
    [[ -n "$SIM_MOTION" ]] && SIM_MOTION="$(abspath "$SIM_MOTION")"
    [[ -n "$SIM_RECORD_COMMANDS" ]] && SIM_RECORD_COMMANDS="$(abspath "$SIM_RECORD_COMMANDS")"

    BRIDGE_PATH="$SCRIPT_DIR/$SIM_BRIDGE_REL"
    if [[ ! -f "$BRIDGE_PATH" ]]; then
        echo -e "${RED}Error: MuJoCo bridge script not found: $BRIDGE_PATH${NC}" >&2
        exit 1
    fi
    if [[ -n "$SIM_MJCF" ]] && [[ ! -f "$SIM_MJCF" ]]; then
        echo -e "${RED}Error: --sim-mjcf does not exist: $SIM_MJCF${NC}" >&2
        exit 1
    fi
    if [[ -n "$SIM_MOTION" ]] && [[ ! -f "$SIM_MOTION" ]]; then
        echo -e "${RED}Error: --sim-motion does not exist: $SIM_MOTION${NC}" >&2
        exit 1
    fi
    if [[ -n "$SIM_IMU_FROM" ]] && \
            [[ "$SIM_IMU_FROM" != "pelvis" && "$SIM_IMU_FROM" != "torso" ]]; then
        echo -e "${RED}Error: --sim-imu-from must be 'pelvis' or 'torso'${NC}" >&2
        exit 1
    fi
fi

# ============================================================================
# Header
# ============================================================================

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                       X2 ULTRA DEPLOY LAUNCHER                       ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}[Mode]${NC}                $MODE"
echo -e "${BLUE}[Package]${NC}             $PKG_NAME"
if [[ "$MODE" == "onbot" ]]; then
    echo -e "${BLUE}[Robot host]${NC}          $ROBOT_USER@$ROBOT_HOST"
    echo -e "${BLUE}[On-bot workspace]${NC}    $ONBOT_WS"
fi
echo ""

# ============================================================================
# Build command-line argument list for ros2 run
# ============================================================================

ROS2_ARGS=("--model" "$MODEL")
[[ -n "$MOTION" ]]            && ROS2_ARGS+=("--motion" "$MOTION")
[[ -n "$LOG_DIR" ]]           && ROS2_ARGS+=("--log-dir" "$LOG_DIR")
[[ -n "$AUTOSTART" ]]         && ROS2_ARGS+=("--autostart-after" "$AUTOSTART")
[[ -n "$MAX_DURATION" ]]      && ROS2_ARGS+=("--max-duration" "$MAX_DURATION")
[[ -n "$TILT_COS" ]]          && ROS2_ARGS+=("--tilt-cos" "$TILT_COS")
[[ -n "$RAMP_SECONDS" ]]      && ROS2_ARGS+=("--ramp-seconds" "$RAMP_SECONDS")
[[ -n "$MAX_TARGET_DEV" ]]    && ROS2_ARGS+=("--max-target-dev" "$MAX_TARGET_DEV")
[[ -n "$ACTION_CLIP" ]]       && ROS2_ARGS+=("--action-clip" "$ACTION_CLIP")
[[ -n "$RETURN_SECONDS" ]]    && ROS2_ARGS+=("--return-seconds" "$RETURN_SECONDS")
[[ -n "$IMU_TOPIC" ]]         && ROS2_ARGS+=("--imu-topic" "$IMU_TOPIC")
[[ -n "$INTRA_OP_THREADS" ]]  && ROS2_ARGS+=("--intra-op-threads" "$INTRA_OP_THREADS")
[[ -n "$OBS_DUMP" ]]          && ROS2_ARGS+=("--obs-dump" "$OBS_DUMP")
$DRY_RUN                      && ROS2_ARGS+=("--dry-run")

# ============================================================================
# MC (Motion Control) HTTP helpers + cleanup trap
# ----------------------------------------------------------------------------
# `aima em start-app/stop-app mc` is a thin wrapper around an HTTP POST to
# PC1's Environment Manager. Talking to it directly avoids requiring an ssh
# key into the robot and makes the whole flow reachable from inside the
# docker_x2/ container. Verified against
# agitbot-x2-record-and-replay/src/x2_recorder/mc_control.py.
# ============================================================================

mc_em_post() {
    # $1 = action ("stop_app" or "start_app")
    local action="$1"
    local url="$MC_EM_URL/json/$action"
    if command -v curl &>/dev/null; then
        curl -fsS -X POST -H 'Content-Type: application/json' \
            --connect-timeout 3 --max-time 5 \
            -d '{"app_name":"mc"}' \
            "$url" >/dev/null
    else
        # Fallback: python3 (always present in our docker image, and on any
        # ROS 2 install). Avoids a hard dependency on curl.
        python3 - "$url" <<'PY'
import json, sys, urllib.request
url = sys.argv[1]
req = urllib.request.Request(
    url,
    data=json.dumps({"app_name": "mc"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=5) as r:
    r.read()
PY
    fi
}

restart_mc_on_exit() {
    # Always called via the trap once we've stopped MC. Idempotent + safe to
    # call multiple times. Preserves the original exit code so a failing
    # deploy run still surfaces its non-zero status to the caller / CI.
    local rc=$?
    if $MC_STOPPED_BY_US; then
        # Mark first so a second SIGINT doesn't double-fire.
        MC_STOPPED_BY_US=false
        echo ""
        echo -e "${BLUE}[cleanup]${NC} restarting MC on $MC_EM_URL ..."
        if mc_em_post start_app; then
            echo -e "${GREEN}[cleanup]${NC} MC start_app POSTed."
        else
            echo -e "${RED}[cleanup]${NC} MC start_app HTTP failed."
            echo -e "${YELLOW}  Manually restart with:${NC}"
            echo -e "  curl -X POST $MC_EM_URL/json/start_app \\"
            echo -e "       -H 'Content-Type: application/json' -d '{\"app_name\":\"mc\"}'"
        fi
    fi
    exit $rc
}

# ============================================================================
# Step 1: Pre-flight (robot for local/onbot, MuJoCo bridge + DDS for sim)
# ============================================================================

if [[ "$MODE" == "sim" ]]; then
    echo -e "${BLUE}[Step 1/4]${NC} Sim pre-flight"

    echo "  Bridge script:     $SCRIPT_DIR/$SIM_BRIDGE_REL"
    echo -n "  Python interpreter ($SIM_PYTHON) ... "
    if command -v "$SIM_PYTHON" &>/dev/null; then
        PYV="$($SIM_PYTHON -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown)"
        echo -e "${GREEN}found ($PYV)${NC}"
    else
        echo -e "${RED}not found${NC}"
        echo -e "${YELLOW}  Install Python 3 or pass --sim-python /path/to/python${NC}"
        exit 1
    fi

    echo -n "  Bridge import smoke test ... "
    if "$SIM_PYTHON" - <<'PY' >/dev/null 2>&1
import importlib
for m in ("mujoco", "rclpy", "aimdk_msgs.msg", "sensor_msgs.msg", "numpy"):
    importlib.import_module(m)
PY
    then
        echo -e "${GREEN}ok${NC}"
    else
        echo -e "${YELLOW}missing one of mujoco / rclpy / aimdk_msgs / sensor_msgs / numpy${NC}"
        echo -e "${YELLOW}  The bridge will fail on launch. Source your ROS 2 install/setup.bash"
        echo -e "  and pip install mujoco numpy if needed.${NC}"
    fi

    echo "  Isolating DDS to loopback (no traffic to/from real robot):"
    echo "    ROS_LOCALHOST_ONLY=1   ROS_DOMAIN_ID=$SIM_DOMAIN_ID"
    export ROS_LOCALHOST_ONLY=1
    export ROS_DOMAIN_ID="$SIM_DOMAIN_ID"
    echo ""
else
    echo -e "${BLUE}[Step 1/4]${NC} Robot pre-flight"

    echo -n "  Pinging $ROBOT_HOST ... "
    if ping -c 1 -W 2 "$ROBOT_HOST" &>/dev/null; then
        echo -e "${GREEN}reachable${NC}"
    else
        echo -e "${RED}unreachable${NC}"
        echo -e "${YELLOW}  Check the SDK ethernet cable and your laptop NIC IP."
        echo -e "  Per dev/quick_start/prerequisites.html: laptop should be"
        echo -e "  static 10.0.1.2/24, robot dev unit (PC2) at 10.0.1.41.${NC}"
        exit 1
    fi

    if command -v ros2 &>/dev/null; then
        # Cache the topic list once; the joint and IMU checks both grep it.
        TOPIC_LIST="$(timeout 5 ros2 topic list 2>/dev/null || true)"

        echo -n "  Checking ROS 2 joint topic visibility ... "
        if echo "$TOPIC_LIST" | grep -q "/aima/hal/joint/leg/state"; then
            echo -e "${GREEN}visible${NC}"
        else
            echo -e "${YELLOW}not visible (DDS may need a moment to discover)${NC}"
        fi

        echo -n "  Checking ROS 2 IMU topic visibility ... "
        if echo "$TOPIC_LIST" | grep -q "/aima/hal/imu/torso/state"; then
            echo -e "${GREEN}visible (torso)${NC}"
        elif echo "$TOPIC_LIST" | grep -q "/aima/hal/imu/torse/state"; then
            echo -e "${YELLOW}firmware uses 'torse' typo${NC}"
            echo -e "${YELLOW}    -> auto-adding --imu-topic /aima/hal/imu/torse/state${NC}"
            if [[ -z "$IMU_TOPIC" ]]; then
                IMU_TOPIC="/aima/hal/imu/torse/state"
                ROS2_ARGS+=("--imu-topic" "$IMU_TOPIC")
            fi
        else
            echo -e "${YELLOW}not visible (DDS may need a moment to discover)${NC}"
        fi
    else
        echo -e "${YELLOW}  ros2 not in PATH; skipping topic visibility check${NC}"
    fi

    if ! $NO_STOP_MC; then
        # ────────────────────────────────────────────────────────────────
        # SAFETY GATE 1/2 -- before we silence the PD-hold controller.
        # ────────────────────────────────────────────────────────────────
        echo ""
        echo -e "${RED}═══════════════════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  SAFETY GATE 1/2 -- STOPPING MC RELEASES THE PD-HOLD CONTROLLER${NC}"
        echo -e "${RED}═══════════════════════════════════════════════════════════════════════${NC}"
        echo ""
        echo -e "${YELLOW}  About to POST stop_app to ${MC_EM_URL}.${NC}"
        echo -e "${YELLOW}  Once MC stops, motors go passive -- gravity wins.${NC}"
        echo ""
        echo -e "${YELLOW}  Confirm BEFORE proceeding:${NC}"
        echo -e "${YELLOW}    [ ] Robot is firmly supported (gantry / harness / hand-held)${NC}"
        echo -e "${YELLOW}    [ ] No personnel within arm or leg reach of robot${NC}"
        echo -e "${YELLOW}    [ ] You are ready for slight settling motion when MC releases${NC}"
        if ! $DRY_RUN; then
            echo -e "${RED}    [ ] E-stop is within reach (this is NOT a dry-run)${NC}"
        fi
        echo ""
        echo -e "${BLUE}  The cleanup trap will POST start_app on script exit (Ctrl-C is safe).${NC}"
        echo ""
        if ! $NO_CONFIRM; then
            read -p "$(echo -e ${RED}Stop MC now? [y/N]: ${NC})" mc_confirm
            if [[ ! "$mc_confirm" =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}  Cancelled before MC stop. Robot is unchanged.${NC}"
                exit 0
            fi
        else
            echo -e "${YELLOW}  --no-confirm: skipping safety gate 1/2${NC}"
        fi
        echo ""
        echo "  Stopping MC module via PC1 EM HTTP API ($MC_EM_URL) ..."
        if ! mc_em_post stop_app; then
            echo -e "${RED}  POST $MC_EM_URL/json/stop_app failed${NC}"
            echo -e "${YELLOW}  Possible causes:${NC}"
            echo -e "${YELLOW}    - host has no route to 10.0.1.40 (check enp10s0 IP / SDK cable)${NC}"
            echo -e "${YELLOW}    - PC1 EM is not running${NC}"
            echo -e "${YELLOW}    - MC is already stopped from a previous run (use --no-stop-mc)${NC}"
            exit 1
        fi
        MC_STOPPED_BY_US=true
        # Install the cleanup trap *only after* a successful stop, so we
        # never accidentally start MC that we did not stop. Covers normal
        # exit, Ctrl-C, SIGTERM. NOTE: this only works if the script does
        # NOT exec the deploy binary (we drop the exec further down).
        trap restart_mc_on_exit EXIT INT TERM

        # Best-effort verify: MC publishes on /aima/hal/joint/arm/command
        # at ~50 Hz; after stop the publisher count should drop to 0 within
        # ~1s. Skip silently if ros2 isn't available (e.g. host shell with
        # no ROS sourced; container shell will have it).
        if command -v ros2 &>/dev/null; then
            sleep 1
            ARM_CMD_PUBS=$(timeout 3 ros2 topic info /aima/hal/joint/arm/command 2>/dev/null \
                | awk '/Publisher count:/ {print $NF}')
            if [[ "$ARM_CMD_PUBS" == "0" ]]; then
                echo -e "${GREEN}  Verified: 0 publishers on /aima/hal/joint/arm/command.${NC}"
            else
                echo -e "${YELLOW}  Could not confirm MC stop (publisher count='${ARM_CMD_PUBS:-?}', expected 0).${NC}"
                echo -e "${YELLOW}  Proceeding anyway -- the deploy will fight MC if it's still up.${NC}"
            fi
        fi
        echo -e "${GREEN}  MC module stopped.${NC}"
    else
        echo -e "${YELLOW}  Skipping MC stop (--no-stop-mc).${NC}"
        echo -e "${YELLOW}  Make sure MC is either stopped or in JOINT_DEFAULT mode.${NC}"
    fi
    echo ""
fi

# ============================================================================
# Step 2: Asset checks (--model and --motion exist where this script runs)
# ============================================================================

echo -e "${BLUE}[Step 2/4]${NC} Asset checks"

check_local_file() {
    if [[ -e "$1" ]]; then
        echo -e "  ${GREEN}✅${NC} $2: $1"
        return 0
    else
        echo -e "  ${RED}❌${NC} $2 not found: $1"
        return 1
    fi
}

if [[ "$MODE" == "onbot" ]]; then
    echo "  (asset existence will be checked on $ROBOT_HOST after rsync)"
else
    MISSING=0
    check_local_file "$MODEL" "Model"      || MISSING=$((MISSING+1))
    [[ -n "$MOTION" ]]  && { check_local_file "$MOTION"  "Motion"  || MISSING=$((MISSING+1)); }
    if [[ "$MODE" == "sim" ]]; then
        check_local_file "$SCRIPT_DIR/$SIM_BRIDGE_REL" "MuJoCo bridge" \
            || MISSING=$((MISSING+1))
        [[ -n "$SIM_MJCF" ]] && { check_local_file "$SIM_MJCF" "MJCF override" \
            || MISSING=$((MISSING+1)); }
        [[ -n "$SIM_MOTION" ]] && { check_local_file "$SIM_MOTION" "Sim RSI motion" \
            || MISSING=$((MISSING+1)); }
    fi
    if [[ $MISSING -gt 0 ]]; then
        echo -e "${RED}  $MISSING asset(s) missing. Aborting.${NC}"
        exit 1
    fi
fi
echo ""

# ============================================================================
# Step 3: Build (colcon)
# ============================================================================

echo -e "${BLUE}[Step 3/4]${NC} Build"

build_local() {
    if $NO_BUILD; then
        echo -e "${YELLOW}  --no-build set; skipping colcon build.${NC}"
        return 0
    fi
    if ! command -v colcon &>/dev/null; then
        echo -e "${RED}  colcon not in PATH. Source your ROS 2 setup.bash first.${NC}"
        exit 1
    fi
    echo "  Building $PKG_NAME locally with colcon ..."
    # The root gear_sonic_deploy/CMakeLists.txt ('g1_deploy') is itself
    # discoverable by colcon and shadows everything beneath it (colcon does
    # not descend into a directory once it finds a package). Restrict the
    # search to the X2 package's tree so colcon actually finds it.
    colcon build --packages-select "$PKG_NAME" \
        --base-paths "$PKG_DIR_REL" \
        --cmake-args -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
    echo -e "${GREEN}  Local build OK.${NC}"
}

build_onbot() {
    echo "  Syncing package to $ROBOT_USER@$ROBOT_HOST:$ONBOT_WS/src/ ..."
    ssh -o ConnectTimeout=5 "$ROBOT_USER@$ROBOT_HOST" "mkdir -p $ONBOT_WS/src"
    rsync -az --delete \
        "$SCRIPT_DIR/$PKG_DIR_REL/" \
        "$ROBOT_USER@$ROBOT_HOST:$ONBOT_WS/src/$PKG_NAME/"
    rsync -az --delete \
        "$SCRIPT_DIR/$SONIC_COMMON_REL/" \
        "$ROBOT_USER@$ROBOT_HOST:$ONBOT_WS/src/sonic_common/"

    if [[ -n "$MODEL" ]]; then
        echo "  Syncing model to $ROBOT_HOST:/tmp/x2_model.onnx ..."
        rsync -az "$MODEL" "$ROBOT_USER@$ROBOT_HOST:/tmp/x2_model.onnx"
        MODEL_REMOTE="/tmp/x2_model.onnx"
    fi
    if [[ -n "$MOTION" ]]; then
        echo "  Syncing motion to $ROBOT_HOST:/tmp/x2_motion.x2m2 ..."
        rsync -az "$MOTION" "$ROBOT_USER@$ROBOT_HOST:/tmp/x2_motion.x2m2"
        MOTION_REMOTE="/tmp/x2_motion.x2m2"
    fi

    if $NO_BUILD; then
        echo -e "${YELLOW}  --no-build set; skipping colcon build on $ROBOT_HOST.${NC}"
    else
        echo "  Running colcon build on $ROBOT_HOST ..."
        ssh "$ROBOT_USER@$ROBOT_HOST" \
            "source /opt/ros/humble/setup.bash && \
             cd $ONBOT_WS && \
             colcon build --packages-select $PKG_NAME \
                 --cmake-args -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
        echo -e "${GREEN}  Remote build OK.${NC}"
    fi
}

if [[ "$MODE" == "onbot" ]]; then
    build_onbot
    # Rewrite ROS2_ARGS to point at the rsync'd remote paths.
    NEW_ARGS=()
    skip_next=false
    for a in "${ROS2_ARGS[@]}"; do
        if $skip_next; then skip_next=false; continue; fi
        case $a in
            --model)  NEW_ARGS+=("--model" "$MODEL_REMOTE"); skip_next=true ;;
            --motion) [[ -n "$MOTION_REMOTE" ]] && { NEW_ARGS+=("--motion" "$MOTION_REMOTE"); skip_next=true; } ;;
            *)        NEW_ARGS+=("$a") ;;
        esac
    done
    ROS2_ARGS=("${NEW_ARGS[@]}")
else
    build_local
fi
echo ""

if $BUILD_ONLY; then
    echo -e "${GREEN}--build-only set; exiting without running.${NC}"
    exit 0
fi

# ============================================================================
# Step 4: Display configuration + confirm + run
# ============================================================================

echo -e "${BLUE}[Step 4/4]${NC} Ready to launch"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                       DEPLOYMENT CONFIGURATION                        ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Mode:               ${GREEN}$MODE${NC}"
echo -e "  Model:              ${GREEN}${ROS2_ARGS[1]}${NC}"
[[ -n "$MOTION" ]]      && echo -e "  Motion:             ${GREEN}$MOTION${NC}"
[[ -n "$LOG_DIR" ]]     && echo -e "  Log dir:            ${GREEN}$LOG_DIR${NC}"
[[ -n "$AUTOSTART" ]]   && echo -e "  Autostart (s):      ${GREEN}$AUTOSTART${NC}"
[[ -n "$MAX_DURATION" ]] && echo -e "  Max duration (s):   ${GREEN}$MAX_DURATION${NC}"
[[ -n "$TILT_COS" ]]    && echo -e "  Tilt cos thresh:    ${GREEN}$TILT_COS${NC}"
[[ -n "$RAMP_SECONDS" ]] && echo -e "  Ramp (s):           ${GREEN}$RAMP_SECONDS${NC}"
[[ -n "$MAX_TARGET_DEV" ]] && echo -e "  Max target dev (rad): ${GREEN}$MAX_TARGET_DEV${NC}"
[[ -n "$ACTION_CLIP" ]]    && echo -e "  Action clip (rad):    ${GREEN}$ACTION_CLIP${NC}"
[[ -n "$RETURN_SECONDS" ]] && echo -e "  Return ramp (s):      ${GREEN}$RETURN_SECONDS${NC}"
[[ -n "$IMU_TOPIC" ]]   && echo -e "  IMU topic:          ${GREEN}$IMU_TOPIC${NC}"
[[ -n "$OBS_DUMP" ]]    && echo -e "  ${YELLOW}OBS-DUMP${NC} -> ${GREEN}$OBS_DUMP${NC} (will exit after first tick)"
$DRY_RUN                && echo -e "  ${YELLOW}DRY-RUN${NC} (stiffness=damping=0; no torque)"
if [[ "$MODE" == "sim" ]]; then
    echo -e "  Sim driver:         ${GREEN}MuJoCo bridge (closed-loop)${NC}"
    echo -e "  Bridge:             ${GREEN}$SCRIPT_DIR/$SIM_BRIDGE_REL${NC}"
    [[ -n "$SIM_MJCF" ]]   && echo -e "  MJCF override:      ${GREEN}$SIM_MJCF${NC}"
    [[ -n "$SIM_MOTION" ]] && echo -e "  RSI motion:         ${GREEN}$SIM_MOTION${NC}"
    [[ -n "$SIM_INIT_FRAME" ]] && echo -e "  RSI frame:          ${GREEN}$SIM_INIT_FRAME${NC}"
    [[ -n "$SIM_IMU_FROM" ]] && echo -e "  IMU from:           ${GREEN}$SIM_IMU_FROM${NC}"
    [[ -n "$SIM_HOLD_STIFFNESS_MULT" ]] && \
        echo -e "  Hold stiffness x:   ${GREEN}$SIM_HOLD_STIFFNESS_MULT${NC}"
    if $SIM_NO_ELASTIC_BAND; then
        echo -e "  ElasticBand:        ${YELLOW}disabled${NC}"
    elif $SIM_VIEWER; then
        echo -e "  ElasticBand:        ${GREEN}ON (viewer: 9 toggle, 7/8 raise/lower)${NC}"
    else
        echo -e "  ElasticBand:        ${GREEN}ON (auto-release ${SIM_BAND_RELEASE_AFTER_S:-1.0}s after 1st cmd)${NC}"
    fi
    [[ -n "$SIM_DT" ]] && echo -e "  Physics dt:         ${GREEN}${SIM_DT}s${NC}"
    $SIM_VIEWER         && echo -e "  Viewer:             ${GREEN}yes${NC}"
    $SIM_PRINT_SCENE    && echo -e "  Print scene:        ${GREEN}yes${NC}"
    [[ -n "$SIM_RECORD_COMMANDS" ]] && \
        echo -e "  Record commands:    ${GREEN}$SIM_RECORD_COMMANDS${NC}"
    echo -e "  DDS isolation:      ${GREEN}ROS_LOCALHOST_ONLY=1, ROS_DOMAIN_ID=$SIM_DOMAIN_ID${NC}"
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}The following command will be executed:${NC}"
echo ""
if [[ "$MODE" == "onbot" ]]; then
    echo -e "${BLUE}ssh $ROBOT_USER@$ROBOT_HOST 'source /opt/ros/humble/setup.bash && \\"
    echo -e "    source $ONBOT_WS/install/setup.bash && \\"
    echo -e "    ros2 run $PKG_NAME x2_deploy_onnx_ref ${ROS2_ARGS[*]}'${NC}"
else
    echo -e "${BLUE}source install/setup.bash${NC}"
    echo -e "${BLUE}ros2 run $PKG_NAME x2_deploy_onnx_ref \\"
    for ((i=0; i<${#ROS2_ARGS[@]}; i++)); do
        if [[ $((i+1)) -lt ${#ROS2_ARGS[@]} ]]; then
            echo -e "${BLUE}    ${ROS2_ARGS[i]} \\"
        else
            echo -e "${BLUE}    ${ROS2_ARGS[i]}${NC}"
        fi
    done
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

if [[ "$MODE" != "sim" ]]; then
    if $DRY_RUN; then
        echo -e "${YELLOW}📋 DRY-RUN: pipeline runs but no torque will be applied.${NC}"
    else
        echo -e "${RED}⚠️  WARNING: this will issue REAL torque commands to the X2 Ultra.${NC}"
        echo -e "${RED}    Robot must be on a gantry / supported. E-stop within reach.${NC}"
    fi
else
    echo -e "${YELLOW}📋 SIM mode: closed-loop MuJoCo. Bridge applies the deploy's PD${NC}"
    echo -e "${YELLOW}    commands as torques. Robot may fall if the policy isn't well-${NC}"
    echo -e "${YELLOW}    behaved -- that's the point. Use --sim-viewer to watch.${NC}"
fi
echo ""

if ! $NO_CONFIRM; then
    if [[ "$MODE" != "sim" ]]; then
        echo -e "${RED}═══════════════════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  SAFETY GATE 2/2 -- LAUNCH POLICY${NC}"
        echo -e "${RED}═══════════════════════════════════════════════════════════════════════${NC}"
        if $DRY_RUN; then
            echo -e "${YELLOW}  Dry-run: pipeline runs but stiffness/damping are zero --${NC}"
            echo -e "${YELLOW}  no torque commanded. CSVs will log what WOULD have been sent.${NC}"
        else
            echo -e "${RED}  Powered: PD targets WILL be published to /aima/hal/joint/*/command${NC}"
            echo -e "${RED}  at ~500 Hz. Soft-start ramp blends policy in over --ramp-seconds.${NC}"
        fi
        echo ""
    fi
    read -p "$(echo -e ${GREEN}Proceed with launch? [Y/n]: ${NC})" confirm
    if [[ -n "$confirm" && ! "$confirm" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Cancelled before launch. MC will be restarted by the cleanup trap.${NC}"
        exit 0
    fi
fi

echo ""
echo -e "${GREEN}🚀 Launching ...${NC}"
echo ""

cleanup_sim() {
    local rc=$?
    if [[ -n "$SIM_BRIDGE_PID" ]] && kill -0 "$SIM_BRIDGE_PID" 2>/dev/null; then
        echo ""
        echo -e "${BLUE}[cleanup]${NC} stopping MuJoCo bridge (pid $SIM_BRIDGE_PID) ..."
        kill -INT "$SIM_BRIDGE_PID" 2>/dev/null || true
        wait "$SIM_BRIDGE_PID" 2>/dev/null || true
    fi
    if [[ -n "$SIM_RECORD_PID" ]] && kill -0 "$SIM_RECORD_PID" 2>/dev/null; then
        echo -e "${BLUE}[cleanup]${NC} stopping command bag recorder (pid $SIM_RECORD_PID) ..."
        kill -INT "$SIM_RECORD_PID" 2>/dev/null || true
        wait "$SIM_RECORD_PID" 2>/dev/null || true
    fi
    exit $rc
}

if [[ "$MODE" == "onbot" ]]; then
    REMOTE_CMD="source /opt/ros/humble/setup.bash && \
        source $ONBOT_WS/install/setup.bash && \
        ros2 run $PKG_NAME x2_deploy_onnx_ref"
    for a in "${ROS2_ARGS[@]}"; do
        REMOTE_CMD+=" $(printf %q "$a")"
    done
    # NOT `exec` -- we want the restart_mc_on_exit trap to fire after ssh
    # returns. Capture the exit code, propagate via the trap.
    ssh -t "$ROBOT_USER@$ROBOT_HOST" "$REMOTE_CMD"
elif [[ "$MODE" == "sim" ]]; then
    if [[ -f install/setup.bash ]]; then
        source install/setup.bash
    fi

    # Background the MuJoCo bridge first so it's ready before the deploy
    # starts polling /aima topics.
    BRIDGE_ARGS=()
    [[ -n "$SIM_MJCF" ]]                 && BRIDGE_ARGS+=("--mjcf" "$SIM_MJCF")
    [[ -n "$SIM_MOTION" ]]               && BRIDGE_ARGS+=("--motion" "$SIM_MOTION")
    [[ -n "$SIM_INIT_FRAME" ]]           && BRIDGE_ARGS+=("--init-frame" "$SIM_INIT_FRAME")
    [[ -n "$SIM_IMU_FROM" ]]             && BRIDGE_ARGS+=("--imu-from" "$SIM_IMU_FROM")
    [[ -n "$SIM_HOLD_STIFFNESS_MULT" ]]  && BRIDGE_ARGS+=("--hold-stiffness-mult" "$SIM_HOLD_STIFFNESS_MULT")
    $SIM_NO_ELASTIC_BAND                 && BRIDGE_ARGS+=("--no-elastic-band")
    [[ -n "$SIM_BAND_RELEASE_AFTER_S" ]] && BRIDGE_ARGS+=("--band-release-after-s" "$SIM_BAND_RELEASE_AFTER_S")
    [[ -n "$SIM_DT" ]]                   && BRIDGE_ARGS+=("--sim-dt" "$SIM_DT")
    BRIDGE_ARGS+=("--ros-domain-id" "$SIM_DOMAIN_ID")
    $SIM_VIEWER                          && BRIDGE_ARGS+=("--viewer")
    $SIM_PRINT_SCENE                     && BRIDGE_ARGS+=("--print-scene")

    echo -e "${BLUE}[sim]${NC} backgrounding: $SIM_PYTHON $SIM_BRIDGE_REL ${BRIDGE_ARGS[*]}"
    "$SIM_PYTHON" "$SCRIPT_DIR/$SIM_BRIDGE_REL" "${BRIDGE_ARGS[@]}" &
    SIM_BRIDGE_PID=$!
    trap cleanup_sim INT TERM EXIT

    if [[ -n "$SIM_RECORD_COMMANDS" ]]; then
        echo -e "${BLUE}[sim]${NC} backgrounding: ros2 bag record -> $SIM_RECORD_COMMANDS"
        ros2 bag record -o "$SIM_RECORD_COMMANDS" \
            /aima/hal/joint/leg/command \
            /aima/hal/joint/waist/command \
            /aima/hal/joint/arm/command \
            /aima/hal/joint/head/command &
        SIM_RECORD_PID=$!
    fi

    # Give the bridge a moment to load MuJoCo + start publishing so the
    # deploy doesn't time out in INIT before the first state arrives.
    sleep 2
    if ! kill -0 "$SIM_BRIDGE_PID" 2>/dev/null; then
        echo -e "${RED}[sim] MuJoCo bridge exited immediately; aborting${NC}" >&2
        echo -e "${RED}      Re-run with --sim-print-scene for diagnostics.${NC}" >&2
        exit 1
    fi

    echo -e "${BLUE}[sim]${NC} starting deploy (cleanup trap installed; Ctrl-C to stop)"
    ros2 run "$PKG_NAME" x2_deploy_onnx_ref "${ROS2_ARGS[@]}"
else
    if [[ -f install/setup.bash ]]; then
        source install/setup.bash
    fi
    # NOT `exec` -- we need the EXIT trap (restart_mc_on_exit) to fire after
    # the deploy returns / Ctrl-C. The trap re-exit()s with the deploy's
    # status so callers / CI still see the right code.
    ros2 run "$PKG_NAME" x2_deploy_onnx_ref "${ROS2_ARGS[@]}"
fi
