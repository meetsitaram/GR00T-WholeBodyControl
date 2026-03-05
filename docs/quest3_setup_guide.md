# Quest 3 VR Teleop Setup Guide

End-to-end instructions for connecting a Meta Quest 3 headset to the GEAR-SONIC control pipeline and teleoperating the Unitree G1 robot in MuJoCo sim2sim.

> **Note:** IsaacLab / Isaac Sim integration is listed as "Coming soon!" in this repo. The simulation environment used below is **MuJoCo**. When IsaacLab support lands, Terminal 1 would change to launch the IsaacLab environment instead of `run_sim_loop.py`; Terminals 2 and 3 remain the same.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Workstation** | Linux (x86_64), with a GPU for MuJoCo rendering |
| **Meta Quest 3** | On the same Wi-Fi network as the workstation |
| **Quest 3 browser** | Meta Quest Browser (built-in) |
| **Python** | 3.10 (auto-installed by the setup script via `uv`) |
| **OpenSSL** | For generating TLS certificates (`openssl` CLI) |
| **C++ deployment binary** | Built per the [installation guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html) |

---

## Step 1: Install the MuJoCo Sim Environment

If you haven't already set up the MuJoCo simulation venv, run from the **repo root**:

```bash
bash install_scripts/install_mujoco_sim.sh
```

This creates `.venv_sim` with MuJoCo, Pinocchio, Unitree SDK2, and other sim dependencies.

---

## Step 2: Install the Quest 3 Teleop Environment

From the **repo root**:

```bash
bash install_scripts/install_quest3.sh
```

This script does the following:
1. Installs `uv` (if not already present) and Python 3.10
2. Creates `.venv_teleop` virtual environment
3. Installs `gear_sonic[teleop]` (ZMQ, Pinocchio, PyVista, etc.)
4. Installs `websockets` (for the Quest 3 WebSocket bridge)
5. Generates a self-signed TLS certificate in `gear_sonic/utils/teleop/vr/quest3_certs/`
6. Installs `gear_sonic[sim]` and `unitree_sdk2_python` (desktop only; skipped on Jetson)

---

## Step 3: Build the C++ Deployment Binary

If not already done, build the deployment binary per the repo's [installation guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html):

```bash
cd gear_sonic_deploy
bash scripts/install_deps.sh
# Follow the build instructions for your platform
```

---

## Step 4: Find Your Workstation IP

The Quest 3 connects to the workstation over Wi-Fi. Get your workstation's LAN IP:

```bash
hostname -I | awk '{print $1}'
```

Note this IP (e.g., `192.168.1.100`) -- you'll need it for the Quest 3 browser.

---

## Step 5: Launch the Sim2Sim Loop (3 Terminals)

### Terminal 1 -- MuJoCo Simulator

```bash
cd /path/to/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

A MuJoCo viewer window will open showing the G1 robot.

### Terminal 2 -- C++ Deployment

```bash
cd /path/to/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh sim --input-type zmq_manager
```

Wait until you see **"Init done"** in the terminal output.

> If the teleop script (Terminal 3) runs on a different machine, add `--zmq-host <IP-of-teleop-machine>`.

### Terminal 3 -- Quest 3 Teleop Manager

```bash
cd /path/to/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/quest3_manager_thread_server.py
```

You should see output like:

```
[Quest3Reader] WebSocket server on wss://0.0.0.0:8765
[Quest3Reader] Serving WebXR app at https://0.0.0.0:8443
[Manager] Waiting for Quest 3 client to connect ...
[Manager] Open the WebXR page in the Quest 3 browser.
[Manager]   URL: https://<workstation-ip>:8443
```

**Optional flags:**

| Flag | Purpose |
|---|---|
| `--ws-port 8765` | WebSocket server port (default: 8765) |
| `--http-port 8443` | HTTPS server port for WebXR app (default: 8443) |
| `--vis-vr3pt` | Enable VR 3-point pose visualization window |
| `--no-ssl` | Disable TLS (not recommended; WebXR requires HTTPS) |
| `--no-g1` | Hide G1 robot mesh in visualization |

---

## Step 6: Connect the Quest 3

1. **Put on the Quest 3 headset.**

2. **Open the Meta Quest Browser** (built-in browser).

3. **Navigate to:**

   ```
   https://<workstation-ip>:8443
   ```

   Replace `<workstation-ip>` with the IP from Step 4 (e.g., `https://192.168.1.100:8443`).

4. **Accept the self-signed certificate warning.** Since we're using a self-signed cert, the browser will warn you. Tap "Advanced" then "Proceed" (or equivalent) to continue.

5. **The Quest 3 Teleop page loads.** You'll see:
   - A WebSocket URL field (auto-filled to `wss://<workstation-ip>:8765`)
   - "Connect WS" button
   - "Start VR" button (disabled until WebSocket is connected)

6. **Tap "Connect WS".** The status should change to "connected" (green).

   > If the WebSocket URL is wrong, correct it to `wss://<workstation-ip>:8765` and tap Connect again.

7. **Tap "Start VR".** The browser enters immersive VR mode. Tracking data starts streaming.

8. **Check Terminal 3** -- you should see:

   ```
   [Quest3Reader] Client connected: ...
   [Manager] Quest 3 connected!
   [Manager] Controls: A+B+X+Y = start/stop, A+X = toggle VR 3PT
   ```

---

## Step 7: Start Teleoperating

### Initial Setup in MuJoCo

1. In **Terminal 2** (deploy.sh), press **`]`** to start the policy.
2. Click the **MuJoCo viewer** window, press **`9`** to drop the robot to the ground.

### Engage the Control Policy

1. **Stand in calibration pose:** upright, feet together, upper arms at sides, forearms bent 90-degrees forward (L-shape at elbows), palms facing inward.

2. **Press A + B + X + Y** simultaneously on the Quest 3 controllers. This:
   - Engages the control policy
   - Enters **PLANNER** mode (locomotion only)
   - Runs initial calibration

3. The robot should now respond to **joystick inputs** for locomotion.

### Quest 3 Controller Controls

**Mode Switching:**

| Action | Buttons | Notes |
|---|---|---|
| Start / Emergency Stop | **A + B + X + Y** | First press: engage + calibrate. Again: stop. |
| Toggle VR 3PT | **A + X** | Switches between PLANNER and PLANNER_VR_3PT |

**Joystick Controls (active in all Planner modes):**

| Input | Function |
|---|---|
| Left Stick | Move direction (forward / backward / strafe) |
| Right Stick (horizontal) | Yaw / heading |
| A + B | Next locomotion mode (Idle -> Slow Walk -> Walk -> Run -> ...) |
| X + Y | Previous locomotion mode |

**Hand Controls (active in PLANNER_VR_3PT mode):**

| Input | Function |
|---|---|
| Left/Right Trigger | Corresponding hand grasp |
| Left/Right Grip | Corresponding hand grip |

### Available Modes

| Mode | Description |
|---|---|
| **OFF** | Policy not running |
| **PLANNER** | Locomotion only via joysticks. No upper-body VR tracking. |
| **PLANNER_VR_3PT** | Locomotion via joysticks + upper-body follows VR 3-point tracking (head + 2 hands) |

### Locomotion Modes (cycled with A+B / X+Y)

| ID | Mode |
|---|---|
| 0 | Idle (default) |
| 1 | Slow Walk |
| 2 | Walk |
| 3 | Run |
| 4 | Squat |
| 5 | Kneel (two legs) |
| 6 | Kneel |
| 7 | Lying face-down |
| 8 | Crawling |
| 9-16 | Boxing variants |
| 17 | Forward Jump |
| 18 | Stealth Walk |
| 19 | Injured Walk |

### Switching to VR 3-Point Upper-Body Tracking

1. While in **PLANNER** mode, align your arms roughly with the robot's current pose.
2. Press **A + X** to enter **PLANNER_VR_3PT** mode.
3. The robot's upper body now tracks your head and hand movements.
4. Press **A + X** again to return to **PLANNER** (locomotion only).

---

## Step 8: Emergency Stop

| Method | Action |
|---|---|
| **Quest 3 controllers** | Press **A + B + X + Y** simultaneously |
| **Keyboard** (Terminal 2) | Press **`O`** for immediate stop |
| **Keyboard** (Terminal 3) | Press **Ctrl+C** to shut down the manager |

---

## Step 9: Shutdown

1. Press **A + B + X + Y** on Quest 3 controllers to stop the policy (or press `O` in Terminal 2).
2. Exit VR mode in the Quest 3 browser (tap the Quest menu button).
3. Press **Ctrl+C** in Terminal 3 (Quest 3 manager).
4. Press **Ctrl+C** in Terminal 1 (MuJoCo simulator).

---

## Troubleshooting

### Quest 3 browser shows "connection refused" or can't load the page

- Verify the workstation IP is correct: `hostname -I`
- Check firewall: ports **8443** (HTTPS) and **8765** (WSS) must be open

  ```bash
  sudo ufw allow 8443/tcp
  sudo ufw allow 8765/tcp
  ```

- Ensure both devices are on the **same Wi-Fi network**

### WebSocket connects but "Start VR" is disabled

- WebXR requires a **secure context** (HTTPS). Make sure you're using `https://` not `http://`
- Accept the self-signed certificate warning in the browser
- If issues persist, navigate to `wss://<ip>:8765` directly in the browser and accept that certificate too

### WebXR shows "not available" or "immersive-vr not supported"

- You must be using the **Meta Quest Browser** on the Quest 3 headset itself
- Desktop browsers don't have real WebXR immersive-vr support (without emulators)

### No tracking data / robot doesn't respond

- Check Terminal 3 for `[Quest3Reader] Client connected` message
- Verify Terminal 2 shows "Init done"
- Verify Terminal 2 was started with `--input-type zmq_manager`

### Certificate errors

- Regenerate the certificate by deleting the existing one and re-running the install script:

  ```bash
  rm -rf gear_sonic/utils/teleop/vr/quest3_certs/
  bash install_scripts/install_quest3.sh
  ```

  Or generate manually:

  ```bash
  mkdir -p gear_sonic/utils/teleop/vr/quest3_certs
  openssl req -x509 -newkey rsa:2048 \
      -keyout gear_sonic/utils/teleop/vr/quest3_certs/key.pem \
      -out gear_sonic/utils/teleop/vr/quest3_certs/cert.pem \
      -days 365 -nodes -subj "/CN=quest3-teleop"
  ```

### High latency or low FPS

- The WebXR app targets ~72 Hz. Check the "Tracking FPS" display on the Quest 3 page.
- Ensure the Wi-Fi connection is stable (5 GHz band preferred).
- If latency is high, try reducing network congestion or moving closer to the router.

---

## Real Robot Deployment

Once comfortable in simulation, switch to the real G1 robot:

### Terminal 1 -- C++ Deployment (Real Robot)

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh real --input-type zmq_manager
# Wait for "Init done"
```

> If the teleop script runs on a different machine: `bash deploy.sh real --input-type zmq_manager --zmq-host <teleop-machine-ip>`

### Terminal 2 -- Quest 3 Teleop Manager

```bash
cd /path/to/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/quest3_manager_thread_server.py
```

No MuJoCo terminal is needed for real robot deployment.

---

## Quick Reference: Command Summary

```bash
# === ONE-TIME SETUP ===

# Install MuJoCo sim environment
bash install_scripts/install_mujoco_sim.sh

# Install Quest 3 teleop environment
bash install_scripts/install_quest3.sh

# === SIM2SIM SESSION ===

# Terminal 1: MuJoCo simulator
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2: C++ deployment
cd gear_sonic_deploy && source scripts/setup_env.sh
bash deploy.sh sim --input-type zmq_manager

# Terminal 3: Quest 3 manager
source .venv_teleop/bin/activate
python gear_sonic/scripts/quest3_manager_thread_server.py

# Then open https://<workstation-ip>:8443 in the Quest 3 browser

# === REAL ROBOT SESSION ===

# Terminal 1: C++ deployment
cd gear_sonic_deploy && source scripts/setup_env.sh
bash deploy.sh real --input-type zmq_manager

# Terminal 2: Quest 3 manager
source .venv_teleop/bin/activate
python gear_sonic/scripts/quest3_manager_thread_server.py

# Then open https://<workstation-ip>:8443 in the Quest 3 browser
```
