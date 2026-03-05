# Dev Notes

## Local Development Environment Setup

**Date:** 2026-03-04
**Host OS:** Ubuntu 24.04 (x86_64, kernel 6.17.0)
**GPU:** NVIDIA (driver 580.126.09, CUDA 13.0)

---

### Prerequisites Installed

| Component | Version | Location |
|-----------|---------|----------|
| NVIDIA Driver | 580.126.09 | system |
| CUDA Toolkit | 13.0.88 | `/usr/local/cuda` (via apt: `cuda-compiler-13-0`, `cuda-cudart-dev-13-0`) |
| TensorRT | 10.15.1 | apt packages (`libnvinfer-dev`, etc.) with symlink tree at `~/TensorRT` |
| ONNX Runtime | 1.16.3 | `/opt/onnxruntime` |
| Python | 3.10.19 | uv-managed, in `.venv/` |
| uv | 0.10.0 | `~/.local/bin/uv` |
| just | 1.43.0 | `/usr/local/bin/just` |
| cmake | system | `/usr/bin/cmake` |
| clang | system | `/usr/bin/clang` |
| git-lfs | 3.4.1 | system |

### TensorRT Setup (apt-based)

The docs recommend downloading the TensorRT TAR package, but on this system TensorRT was installed via apt from the NVIDIA CUDA repository. Because `FindTensorRT.cmake` expects a `TensorRT_ROOT/include/` + `TensorRT_ROOT/lib/` layout, a symlink directory was created:

```bash
mkdir -p ~/TensorRT/include ~/TensorRT/lib
ln -sf /usr/include/x86_64-linux-gnu/NvInfer*.h ~/TensorRT/include/
ln -sf /usr/include/x86_64-linux-gnu/NvOnnx*.h ~/TensorRT/include/
ls /usr/include/x86_64-linux-gnu/Nv*.h | xargs -I {} ln -sf {} ~/TensorRT/include/
ln -sf /usr/lib/x86_64-linux-gnu/libnvinfer*.so* ~/TensorRT/lib/
ln -sf /usr/lib/x86_64-linux-gnu/libnvonnxparser*.so* ~/TensorRT/lib/
ln -sf /usr/lib/x86_64-linux-gnu/libtensorrt*.so* ~/TensorRT/lib/
```

The environment variable is set in `~/.bashrc`:

```bash
export TensorRT_ROOT=$HOME/TensorRT
```

> **Note:** The docs recommend TensorRT **10.13** for x86_64. This system has **10.15** (matched to CUDA 13.0). Different TensorRT versions *may* cause inference issues — test in simulation first before deploying to hardware.

---

### Python Virtual Environment

Created with `uv` using Python 3.10 (required by both `gear_sonic` and `decoupled_wbc` which specify `requires-python = "~=3.10.0"`):

```bash
uv venv --python 3.10 .venv
```

#### Installed packages

```bash
# gear_sonic with teleop and simulation extras
uv pip install -e "gear_sonic/[teleop,sim]"

# decoupled_wbc with full and dev extras
# Requires --no-build-isolation due to lerobot's poetry build backend
# and temporarily patching pyproject.toml paths (see workaround below)
uv pip install setuptools wheel poetry-core poetry
uv pip install --no-build-isolation -e "decoupled_wbc/[full,dev]"
```

#### decoupled_wbc install workaround

`decoupled_wbc/pyproject.toml` references `readme = "../README.md"` and `license = {file = "../LICENSE"}` which newer versions of setuptools reject (cannot access files outside the package directory). To install:

1. Create temporary symlinks inside `decoupled_wbc/`:
   ```bash
   cd decoupled_wbc
   ln -sf ../README.md README.md
   ln -sf ../LICENSE LICENSE
   ```
2. Temporarily edit `decoupled_wbc/pyproject.toml`:
   - Change `readme = "../README.md"` → `readme = "README.md"`
   - Change `license = {file = "../LICENSE"}` → `license = {file = "LICENSE"}`
3. Run the install (see above)
4. Revert `pyproject.toml` and remove symlinks

Additionally, `lerobot` (a dependency) uses `poetry-core` as its build backend, so `poetry-core` and `poetry` must be installed in the venv before running with `--no-build-isolation`.

---

### C++ Build (gear_sonic_deploy)

Built using `just` from inside `gear_sonic_deploy/`:

```bash
cd gear_sonic_deploy
export TensorRT_ROOT=$HOME/TensorRT
source scripts/setup_env.sh
just build
```

#### Built targets (in `gear_sonic_deploy/target/release/`):

| Binary | Description |
|--------|-------------|
| `g1_deploy_onnx_ref` | Main deployment executable |
| `freq_test` | Inference frequency test |
| `run_tests` | Unit tests (GTest) |
| `zmq_pose_subscriber_test` | ZMQ pose subscriber test |
| `zmq_python_sender_test` | ZMQ Python sender test |

#### Build notes

- ROS2 is **not installed** — the build skips `ROS2InputHandler` support (this is optional)
- DLA support is disabled on x86_64 (DLA is Jetson-only)
- cppzmq headers were vendored to `gear_sonic_deploy/third_party/cppzmq/` (not available via apt on Ubuntu 24.04)

---

### Environment Variables (in ~/.bashrc)

```bash
export TensorRT_ROOT=$HOME/TensorRT
```

### Useful Commands

```bash
# Activate Python venv
source .venv/bin/activate

# Set up C++ build environment (CUDA, TensorRT, ONNX Runtime paths)
cd gear_sonic_deploy && source scripts/setup_env.sh

# Build C++ project
just build          # Release build
just build Debug    # Debug build
just clean          # Clean build artifacts
just --list         # Show all available commands

# Run unit tests
./gear_sonic_deploy/target/release/run_tests

# Run inference frequency test
./gear_sonic_deploy/target/release/freq_test policy/release/model_decoder.onnx
```

---

## Quest 3 VR Teleop Setup

**Date:** 2026-03-05

### Overview

Quest 3 VR teleop uses WebXR + WebSocket to stream head + controller tracking data to the GEAR-SONIC pipeline. The sim2sim setup requires **3 terminals**.

### Additional Dependencies

Installed into the existing `.venv` (no separate venv needed):

```bash
source .venv/bin/activate
uv pip install websockets
```

A self-signed TLS certificate is generated at `gear_sonic/utils/teleop/vr/quest3_certs/`:

```bash
mkdir -p gear_sonic/utils/teleop/vr/quest3_certs
openssl req -x509 -newkey rsa:2048 \
    -keyout gear_sonic/utils/teleop/vr/quest3_certs/key.pem \
    -out gear_sonic/utils/teleop/vr/quest3_certs/cert.pem \
    -days 365 -nodes -subj "/CN=quest3-teleop"
```

Firewall ports must be open:

```bash
sudo ufw allow 8443/tcp   # HTTPS for WebXR app
sudo ufw allow 8765/tcp   # WSS for WebSocket data
```

### Running Sim2Sim with Quest 3

```bash
# Terminal 1 — MuJoCo Simulator
bash run_sim.sh

# Terminal 2 — C++ Deployment (with ZMQ manager input)
bash deploy_sonic_with_zmq.sh

# Terminal 3 — Quest 3 Teleop Manager
bash run_quest3_server.sh
```

### Quest 3 Headset Steps

1. Ensure Quest 3 is on the **same Wi-Fi** as the workstation
2. Open **Meta Quest Browser**
3. Navigate to `https://<workstation-ip>:8443`
4. Accept the self-signed certificate warning (Advanced → Proceed)
5. Also visit `https://<workstation-ip>:8765` and accept that cert too
6. Go back to `https://<workstation-ip>:8443`
7. Tap **"Connect WS"** (status turns green)
8. Tap **"Start VR"** to begin streaming

### Engaging the Robot

1. In **Terminal 2**, wait for `Init Done`
2. In **MuJoCo viewer**, press **`9`** to drop the robot
3. On **Quest 3 controllers**, press **A+B+X+Y** together to engage teleop (starts in PLANNER mode)
4. Press **A+B** to cycle from IDLE to SLOW_WALK / WALK / RUN
5. Press **A+X** to toggle VR 3PT (upper body tracking)

### Controls Reference

| Input | Action |
|---|---|
| A+B+X+Y | Start / Emergency Stop |
| A+X | Toggle VR 3PT (upper body tracking) |
| Left Stick | Move direction |
| Right Stick | Yaw / heading |
| A+B | Next locomotion mode |
| X+Y | Previous locomotion mode |
| Triggers | Hand grasp (VR 3PT mode) |
| Grips | Hand grip (VR 3PT mode) |

### Locomotion Modes

| ID | Mode |
|---|---|
| 0 | Idle (default) |
| 1 | Slow Walk |
| 2 | Walk |
| 3 | Run |
| 4 | Squat |
| 5-6 | Kneel |
| 7 | Lying face-down |
| 8 | Crawling |
| 17 | Forward Jump |
| 18 | Stealth Walk |
| 19 | Injured Walk |

---

### Issues Encountered & Fixes

#### 1. CycloneDDS multicast error on loopback

**Symptom:** Sim loop exits immediately with CycloneDDS domain creation failure.

**Fix:** Enable multicast on the loopback interface (already handled by `run_sim.sh`):
```bash
sudo ip link set lo multicast on
```

#### 2. WebXR "request failed" error on Start VR

**Symptom:** Clicking "Start VR" shows "request failed" on the Quest 3 browser.

**Cause:** The self-signed certificate for the WebSocket port (8765) was not accepted. The browser blocks WSS connections to untrusted certs.

**Fix:** Navigate to `https://<workstation-ip>:8765` in the Quest 3 browser, accept the certificate warning, then go back to the main page and retry.

#### 3. WebXR "device does not support requestReferenceSpace"

**Symptom:** Start VR fails with "failed to execute requestReferenceSpace on XRSession — device does not support".

**Cause:** Quest 3 guardian boundary not set up. The `local-floor` reference space requires a floor level and play boundary.

**Fix:** On Quest 3: Settings → Physical Space → tap **"Set Floor"** (point controller at floor, confirm) → tap **"Create Boundary"** (draw play area, confirm). The WebXR app now falls back through `local-floor` → `local` → `viewer` if needed.

#### 4. Controllers not detected (hand tracking mode)

**Symptom:** Head position updates in logs but all buttons/axes/triggers are zeros. Server shows `Input sources (controllers): 0` or `hand-tracking (gamepad=NO)`.

**Cause:** Quest 3 defaults to hand tracking when not holding physical controllers. WebXR hand tracking sources don't have gamepad buttons/axes.

**Fix:** **Pick up the physical Quest 3 controllers.** The headset auto-switches to controller mode. If it doesn't switch, go to Settings → Movement Tracking → Hand and Body Tracking → **turn off Hand Tracking** to force controller mode.

#### 5. Black screen in VR immersive mode

**Symptom:** Clicking Start VR enters immersive mode but shows a black screen.

**Cause:** The `immersive-vr` session renders to a blank WebGL canvas with no 3D scene.

**Fix:** The WebXR app now requests `immersive-ar` mode first (passthrough), falling back to `immersive-vr`. The canvas clears to transparent so the passthrough camera feed is visible.

#### 6. Browser caching old HTML page

**Symptom:** Code changes to the WebXR HTML page don't take effect after restarting the server.

**Cause:** Quest 3 browser caches the page aggressively.

**Fix:** Clear cache in Quest Browser (three dots menu → Settings → Clear Browsing Data → Cached images and files → Clear). The HTTP server now sends `Cache-Control: no-cache` headers to prevent future caching.

#### 7. Terminal 2 doesn't accept keyboard input with `--input-type zmq_manager`

**Symptom:** Pressing `]` in Terminal 2 does nothing.

**Cause:** With `--input-type zmq_manager`, all input comes via ZMQ from the Quest 3 manager — keyboard input is disabled.

**Fix:** This is expected. Use the Quest 3 controllers instead: **A+B+X+Y** to start the policy (replaces `]`), **A+B+X+Y** again for emergency stop (replaces `O`).

#### 8. Robot collapses / doesn't move after engaging

**Symptom:** Robot stands after A+B+X+Y but joystick has no effect.

**Cause:** Locomotion mode starts at **IDLE** (mode 0), which ignores joystick input.

**Fix:** Press **A+B** to cycle to SLOW_WALK (mode 1) or WALK (mode 2), then use the joystick.

#### 9. `websockets.exceptions.InvalidUpgrade: invalid Connection header: keep-alive`

**Symptom:** Error spam in Terminal 3 when accepting certs.

**Cause:** Navigating directly to the WebSocket port (`https://<ip>:8765`) sends a regular HTTPS request instead of a WebSocket upgrade. The `websockets` library (v16) rejects it.

**Fix:** This error is harmless — the certificate is still accepted. The actual WebSocket connection from the WebXR page works fine afterward.
