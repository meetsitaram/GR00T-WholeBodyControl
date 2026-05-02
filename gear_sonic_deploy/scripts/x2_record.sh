#!/usr/bin/env bash
# x2_record.sh -- thin wrapper around x2_record_real_run.py that auto-relaunches
# itself inside the docker_x2/x2sim container so you can run it from any host
# shell, without needing ROS 2 / aimdk_msgs installed on the host.
#
# Usage (from any host shell):
#
#   ./gear_sonic_deploy/scripts/x2_record.sh \
#       --out scratch/manual_arm_test_$(date +%Y%m%d_%H%M%S).npz \
#       --note "manual arm wiggle"
#
# All flags after this script's own (just --no-docker) are forwarded verbatim
# to x2_record_real_run.py. See `x2_record_real_run.py --help` for the full
# list (--out, --duration, --imu-topic, --status-period, --moving-threshold,
# --note, --summarize, --quiet).
#
# Mirrors the auto-docker pattern of deploy_x2.sh:maybe_relaunch_in_docker.
# In particular it mounts $HOME -> $HOME at the same absolute path so any
# host paths you pass (e.g. --out scratch/foo.npz, --summarize ~/x/y.npz)
# resolve identically inside the container -- you don't have to translate
# them yourself.
#
# Real-mode env: x2sim's compose default is sim DDS isolation
# (ROS_LOCALHOST_ONLY=1, ROS_DOMAIN_ID=73). The recorder needs the real
# robot's bus, so we override to ROS_LOCALHOST_ONLY=0 + ROS_DOMAIN_ID=0
# (or X2_REAL_DOMAIN_ID if set).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
USER_CWD="$(pwd)"

NC='\033[0m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'

maybe_relaunch_in_docker() {
    if [[ -d /workspace/sonic ]] || [[ -n "${X2_RECORD_IN_DOCKER:-}" ]]; then
        return 0
    fi

    # Bypass docker for invocations that don't need rclpy / aimdk_msgs:
    #   --help / -h        : prints usage and exits
    #   --no-docker        : explicit user opt-out
    #   --summarize PATH   : reanalyses a saved .npz; only needs numpy
    for a in "$@"; do
        case "$a" in
            --no-docker|-h|--help|--summarize) return 0 ;;
        esac
    done

    if ! command -v docker &>/dev/null; then
        echo -e "${YELLOW}Note: 'docker' not in PATH and ROS doesn't look sourced;${NC}" >&2
        echo -e "${YELLOW}      x2_record.sh is going to fail on rclpy/aimdk_msgs imports.${NC}" >&2
        echo -e "${YELLOW}      Either install docker + run the docker_x2 container,${NC}" >&2
        echo -e "${YELLOW}      or source ROS 2 + aimdk_msgs and re-run with --no-docker.${NC}" >&2
        return 0
    fi

    local compose_dir="$DEPLOY_DIR/docker_x2"
    if [[ ! -f "$compose_dir/docker-compose.yml" ]]; then
        echo -e "${YELLOW}Note: $compose_dir/docker-compose.yml not found; aborting auto-docker.${NC}" >&2
        return 0
    fi

    local domain_id="${X2_REAL_DOMAIN_ID:-0}"
    local tty_args=("-T")  # default no TTY (works in CI / nohup)
    if [[ -t 0 && -t 1 ]]; then
        tty_args=()
    fi

    # Forward $DISPLAY etc. so any matplotlib popping out of --summarize works
    # (currently x2_record_real_run.py doesn't open a window, but future flags
    # might; cheap to forward).
    echo -e "${BLUE}[auto-docker]${NC} re-exec inside docker_x2/x2sim (recorder)"
    echo -e "${BLUE}[auto-docker]${NC} mounting \$HOME ($HOME) at $HOME inside container"
    echo -e "${BLUE}[auto-docker]${NC} real-DDS env: ROS_LOCALHOST_ONLY=0, ROS_DOMAIN_ID=$domain_id"
    echo -e "${BLUE}[auto-docker]${NC} use --no-docker to bypass (requires rclpy + aimdk_msgs on host)"
    echo ""

    # Re-exec the *container* path to this script so SCRIPT_DIR inside the
    # nested invocation resolves to /workspace/sonic/...; the host's $HOME
    # mount makes the file equivalent both ways but the container path
    # interacts cleanly with /ros2_ws/install/ setup.bash and the colcon
    # install volumes attached at /workspace/sonic/gear_sonic_deploy/install.
    local script_in_container="/workspace/sonic/gear_sonic_deploy/scripts/$(basename "${BASH_SOURCE[0]}")"

    cd "$compose_dir"
    export X2_RECORD_IN_DOCKER=1
    exec docker compose run --rm \
        "${tty_args[@]}" \
        -e "ROS_LOCALHOST_ONLY=0" \
        -e "ROS_DOMAIN_ID=$domain_id" \
        -e "X2_RECORD_IN_DOCKER=1" \
        -v "$HOME:$HOME:rw" \
        -w "$USER_CWD" \
        x2sim \
        bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec "$@"' \
        bash "$script_in_container" "$@"
}

maybe_relaunch_in_docker "$@"

# Inside the container (or on a host that already has ROS sourced + the
# --no-docker flag): strip --no-docker from the args (record_real_run.py
# doesn't know about it) and exec the recorder.
RECORDER="$DEPLOY_DIR/scripts/x2_record_real_run.py"
if [[ ! -f "$RECORDER" ]]; then
    # Try the canonical container path as a fallback.
    if [[ -f "/workspace/sonic/gear_sonic_deploy/scripts/x2_record_real_run.py" ]]; then
        RECORDER="/workspace/sonic/gear_sonic_deploy/scripts/x2_record_real_run.py"
    else
        echo "ERROR: cannot find x2_record_real_run.py near $SCRIPT_DIR" >&2
        exit 1
    fi
fi

forward_args=()
for a in "$@"; do
    case "$a" in
        --no-docker) ;;        # consumed
        *) forward_args+=("$a") ;;
    esac
done

exec python3 "$RECORDER" "${forward_args[@]}"
