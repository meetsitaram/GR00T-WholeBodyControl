# Quest 3 vs PICO VR Teleop: Side-by-Side Comparison

This document compares the Quest 3 VR teleop implementation against the existing PICO VR teleop system for GEAR-SONIC control.

---

## 1. VR Data Reader: `Quest3Reader` vs `PicoReader`

| Aspect | PICO (`PicoReader` in pico_manager) | Quest 3 (`Quest3Reader` in quest3_reader.py) |
|---|---|---|
| **Data source** | XRoboToolkit SDK (`xrt` C++ bindings) | WebSocket server (Python `websockets`) |
| **Transport** | Wi-Fi from headset to XRT PC Service to local SDK | Wi-Fi from headset browser (WebXR) to WebSocket |
| **Raw data** | 24-joint SMPL body poses (24, 7) | 3 poses directly: head + 2 controllers |
| **Tracking points** | Full body, then extract 3 | Native 3-point (head + hands) |
| **Background thread** | Polls `xrt.get_body_joints_pose()` in a loop | asyncio WebSocket `serve()` in a thread |

**PICO PicoReader** (`gear_sonic/scripts/pico_manager_thread_server.py`):

```python
class PicoReader:
    """
    Background reader that pulls Pico/XRT data as fast as possible and computes dt/FPS.
    """

    def __init__(self, max_queue_size: int = 15):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_t = None
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if not xrt.is_body_data_available():
                time.sleep(0.001)
                continue
            stamp_ns = xrt.get_time_stamp_ns()
            # ... polls xrt.get_body_joints_pose() ...
            # Returns: {"body_poses_np": np.array(body_poses), ...}
```

**Quest 3 Quest3Reader** (`gear_sonic/utils/teleop/vr/quest3_reader.py`):

```python
class Quest3Reader:
    """Background reader for Quest 3 VR data via WebSocket."""

    def __init__(self, ws_host="0.0.0.0", ws_port=8765, http_port=8443, use_ssl=True):
        self._latest: dict | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._connected = False
        self._fps_ema = 0.0
        # ... WebSocket + HTTP server setup ...

    def start(self):
        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()
        self._http_thread = threading.Thread(target=self._run_http, daemon=True)
        self._http_thread.start()

    def get_latest(self) -> dict | None:
        with self._lock:
            return self._latest

    # Receives JSON via WebSocket, computes 3pt pose inline
    # Returns: {"vr_3pt_pose": (3,7) ndarray, "buttons": {...}, "axes": {...}, ...}
```

`PicoReader._run()` polls a native C++ SDK in a tight loop. `Quest3Reader` runs an async WebSocket server that receives pushed data from the browser.

---

## 2. Controller Input: XRT Functions vs WebSocket JSON

**PICO** -- 6 separate XRT SDK function calls (`pico_manager_thread_server.py`):

```python
def get_controller_inputs():
    """Fetch controller button/trigger states from XRoboToolkit."""
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_grip = xrt.get_left_grip()
    right_grip = xrt.get_right_grip()
    left_menu_button = xrt.get_left_menu_button()
    return left_menu_button, left_trigger, right_trigger, left_grip, right_grip

def get_controller_axes():
    left_axis = xrt.get_left_axis()   # [x, y]
    right_axis = xrt.get_right_axis() # [x, y]
    # ...

def get_abxy_buttons():
    a_pressed = bool(xrt.get_A_button())
    b_pressed = bool(xrt.get_B_button())
    x_pressed = bool(xrt.get_X_button())
    y_pressed = bool(xrt.get_Y_button())
    # ...
```

**Quest 3** -- all from the same `get_latest()` sample (`quest3_reader.py`):

```python
    def get_controller_inputs(self) -> tuple[float, float, float, float]:
        """Returns (left_trigger, right_trigger, left_grip, right_grip)."""
        sample = self.get_latest()
        if sample is None:
            return 0.0, 0.0, 0.0, 0.0
        buttons = sample.get("buttons", {})
        return (
            float(buttons.get("leftTrigger", 0.0)),
            float(buttons.get("rightTrigger", 0.0)),
            float(buttons.get("leftGrip", 0.0)),
            float(buttons.get("rightGrip", 0.0)),
        )

    def get_controller_axes(self) -> tuple[float, float, float, float]:
        sample = self.get_latest()
        # ... reads from sample["axes"]["lx"], etc.

    def get_buttons(self) -> tuple[bool, bool, bool, bool]:
        sample = self.get_latest()
        # ... reads from sample["buttons"]["a"], etc.
```

PICO makes individual C++ SDK calls per button. Quest 3 unpacks everything from a single JSON message that arrived via WebSocket.

---

## 3. Coordinate Transform & 3-Point Pose Extraction

**PICO** -- full SMPL pipeline (`pico_manager_thread_server.py`):

```
xrt.get_body_joints_pose() -> (24, 7) Unity frame
        |
        v
_compute_rel_transform() per joint
Unity Y-up left-handed -> Robot Z-up right-handed
Q = [[-1,0,0],[0,0,1],[0,1,0]]
        |
        v
Extract joints 0, 22, 23, 12 (Root, L-Wrist, R-Wrist, Neck)
        |
        v
Apply OFFSETS per joint (rotation corrections)
        |
        v
Make relative to Root (subtract root pos, root_inv * quat)
        |
        v
Return (3, 7) [L-Wrist, R-Wrist, Neck]
```

**Quest 3** -- direct 3-point (`quest3_reader.py`):

```
WebXR head + left + right (3 poses, WebXR frame)
        |
        v
transform_pose_to_robot()
WebXR right-handed Y-up -> Robot Z-up
Q = [[0,0,-1],[-1,0,0],[0,1,0]]
        |
        v
Estimate root at floor below headset (z=0)
        |
        v
Subtract root position (relative coordinates)
        |
        v
Return (3, 7) [L-Wrist, R-Wrist, Head/Neck]
```

### Why the transform matrices differ

| Axis | Unity (PICO) | WebXR (Quest 3) | Robot |
|------|-------------|-----------------|-------|
| Forward | +Z | -Z | +X |
| Right | +X | +X | -Y |
| Up | +Y | +Y | +Z |
| Handedness | Left | Right | Right |

PICO transform matrix (Unity left-handed Y-up):
```python
Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])  # Unity -> Robot
```

Quest 3 transform matrix (WebXR right-handed Y-up):
```python
WEBXR_TO_ROBOT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])  # WebXR -> Robot
```

Unity is left-handed, so the X-axis mapping is negated compared to WebXR.

---

## 4. Planner Streamer: `PlannerStreamer` vs `Quest3PlannerStreamer`

The joystick-to-locomotion logic is **identical** between both. The difference is only in how inputs are read.

**PICO** (`pico_manager_thread_server.py`):

```python
    def run_once(self, stream_mode: StreamMode):
        try:
            # PICO-specific: check XRT timestamp to detect disconnect
            xrt_timestamp = xrt.get_time_stamp_ns()
            if xrt_timestamp == self.last_xrt_timestamp:
                return
            self.last_xrt_timestamp = xrt_timestamp

            # Read buttons via XRT SDK
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()
            # ...
            lx, ly, rx, ry = get_controller_axes()
            # ...
            # VR 3PT: extract from SMPL body poses
            if stream_mode == StreamMode.PLANNER_VR_3PT:
                sample = self.reader.get_latest()
                if sample is not None:
                    vr_3pt_pose = self.three_point.process(
                        _process_3pt_pose(sample["body_poses_np"])  # SMPL extraction
                    )
                # Hand control via XRT
                (_, left_trigger, right_trigger, left_grip, right_grip) = get_controller_inputs()
```

**Quest 3** (`quest3_manager_thread_server.py`):

```python
    def run_once(self, stream_mode: StreamMode):
        try:
            # No XRT timestamp check needed -- WebSocket is push-based

            # Read buttons via Quest3Reader
            a, b, x, y = self.reader.get_buttons()
            # ...
            lx, ly, rx, ry = self.reader.get_controller_axes()
            # ... (identical joystick math) ...

            # VR 3PT: use pre-computed 3pt directly (no SMPL)
            if stream_mode == StreamMode.PLANNER_VR_3PT:
                raw_3pt = self.reader.get_3pt_pose()  # already (3,7) in robot frame
                if raw_3pt is not None:
                    vr_3pt_pose = self.three_point.process(raw_3pt)  # direct!
                # Hand control via Quest3Reader
                lt, rt, lg, rg = self.reader.get_controller_inputs()
```

The core locomotion math (joystick deadzone, speed mapping, yaw accumulation, movement vector rotation) is **byte-for-byte identical** in both. The only differences are:

1. No XRT timestamp guard in Quest 3 (WebSocket is push-based)
2. `_process_3pt_pose(sample["body_poses_np"])` (PICO/SMPL) vs `self.reader.get_3pt_pose()` (Quest 3/direct)
3. `get_abxy_buttons()` (free function calling XRT) vs `self.reader.get_buttons()` (method)

---

## 5. Manager State Machine: `run_pico_manager` vs `run_quest3_manager`

**PICO** has 6 modes in a complex state machine (`pico_manager_thread_server.py`):

```
Chain 1: POSE <--(B+Y)--> PLANNER_FROZEN_UPPER_BODY <--(left_axis_click)--> VR_3PT
Chain 2: POSE <--(A+X)--> PLANNER <--(left_axis_click)--> VR_3PT
Emergency: A+B+X+Y --> OFF
Pause:     Menu held --> POSE_PAUSE
```

**Quest 3** has 3 modes in a simplified state machine (`quest3_manager_thread_server.py`):

```
OFF --(A+B+X+Y)--> PLANNER --(A+X)--> PLANNER_VR_3PT
         ^                                  |
         +--------(A+B+X+Y)--------<--------+
```

Quest 3 omits `POSE`, `PLANNER_FROZEN_UPPER_BODY`, and `POSE_PAUSE` because those require SMPL full-body tracking (which Quest 3 doesn't provide). The two modes that remain are the ones most useful for the target use case: planner-only locomotion and planner + VR upper body.

---

## 6. Initialization: `run_pico_manager` vs `run_quest3_manager`

**PICO** (`pico_manager_thread_server.py`):

```python
    # Start XRoboToolkit native service
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    # ZMQ socket
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")

    # Reader + ThreePointPose + PoseStreamer + PlannerStreamer
    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()
```

**Quest 3** (`quest3_manager_thread_server.py`):

```python
    # Start WebSocket + HTTPS servers
    reader = Quest3Reader(ws_port=ws_port, http_port=http_port, use_ssl=use_ssl)
    reader.start()

    print("[Manager] Waiting for Quest 3 client to connect ...")
    print(f"[Manager] Open the WebXR page in the Quest 3 browser.")
    while not reader.is_connected:
        time.sleep(0.5)
    print("[Manager] Quest 3 connected!")

    # ZMQ socket (identical)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")

    # ThreePointPose + Quest3PlannerStreamer (no PoseStreamer needed)
```

PICO starts a native system service and waits for body tracking. Quest 3 starts WebSocket/HTTP servers and waits for a browser connection.

---

## 7. WebXR App vs XRoboToolkit App

| Aspect | PICO (XRoboToolkit) | Quest 3 (WebXR) |
|---|---|---|
| **Client runs on** | PICO headset (native APK) | Quest 3 browser (HTML page) |
| **Installation** | Download APK, install in Unknown apps | Open a URL in the browser |
| **SDK dependency** | XRoboToolkit-PICO-1.1.1.apk + PC Service | None (standard Web APIs) |
| **Tracking API** | XRT body tracking (24 SMPL joints) | WebXR `getViewerPose()` + `gripSpace` |
| **Buttons API** | XRT `get_A_button()`, etc. | WebXR `gamepad.buttons[]` |
| **Transport** | XRT SDK JSON over Wi-Fi (proprietary) | WebSocket (JSON) |
| **Extra hardware** | 2x ankle motion trackers (for full body) | None needed |

The WebXR app (`gear_sonic/utils/teleop/vr/quest3_webxr_app/index.html`) is a single ~200-line HTML file that the Quest 3 browser loads from the HTTPS server. It uses standard WebXR APIs to read tracking data each frame:

```javascript
// Head pose (from index.html onXRFrame)
const pose = frame.getViewerPose(refSpace);
const hp = pose.transform.position;
const ho = pose.transform.orientation;

// Controller poses
for (const src of xrSession.inputSources) {
    const gp = frame.getPose(src.gripSpace, refSpace);
    // ... buttons from src.gamepad.buttons[0..5]
    // ... axes from src.gamepad.axes[2..3]
}
```

---

## 8. Install Script: `install_quest3.sh` vs `install_pico.sh`

| Step | `install_pico.sh` | `install_quest3.sh` |
|------|------------------|---------------------|
| Python 3.10 + uv | Same | Same |
| `gear_sonic[teleop]` | Same | Same |
| **XRoboToolkit SDK** | `uv pip install cmake pybind11 ...` + CMake build | **Skipped** |
| **PXREARobotSDK (aarch64)** | Build from source on Jetson | **Skipped** |
| **websockets** | Not installed | `uv pip install websockets` |
| **TLS certificate** | Not needed | `openssl req -x509 ...` auto-generated |
| `gear_sonic[sim]` | Same | Same |
| `unitree_sdk2_python` | Same | Same |

The Quest 3 script is ~30 lines shorter because it skips the XRoboToolkit SDK installation entirely.

---

## 9. Shared Components (`gear_sonic/utils/teleop/common.py`)

The following classes and functions were extracted from `pico_manager_thread_server.py` into a shared module used by both PICO and Quest 3:

| Component | Purpose |
|---|---|
| `LocomotionMode` | Enum for locomotion modes (IDLE, SLOW_WALK, WALK, RUN, etc.) |
| `StreamMode` | Enum for streaming modes (OFF, POSE, PLANNER, PLANNER_VR_3PT, etc.) |
| `JOYSTICK_DEADZONE` | Constant for joystick dead zone threshold |
| `YawAccumulator` | Tracks cumulative yaw from right stick input |
| `generate_finger_data` | Generates finger joint data for ZMQ messages |
| `init_hand_ik_solvers` | Initializes IK solvers for hand open/close |
| `compute_hand_joints_from_inputs` | Maps trigger/grip values to hand joint angles |
| `ThreePointPose` | Device-agnostic calibration for 3-point VR pose data |
| `FeedbackReader` | Reads robot feedback (measured joints) via ZMQ |

Both `pico_manager_thread_server.py` and `quest3_manager_thread_server.py` import these from `gear_sonic.utils.teleop.common`.

---

## File Map

| Quest 3 File | PICO Equivalent | Notes |
|---|---|---|
| `gear_sonic/utils/teleop/vr/quest3_reader.py` | `PicoReader` class in `pico_manager_thread_server.py` | WebSocket vs XRT SDK |
| `gear_sonic/utils/teleop/vr/quest3_webxr_app/index.html` | XRoboToolkit APK (external) | Browser app vs native APK |
| `gear_sonic/scripts/quest3_manager_thread_server.py` | `gear_sonic/scripts/pico_manager_thread_server.py` | Simplified state machine |
| `gear_sonic/utils/teleop/common.py` | Was inline in `pico_manager_thread_server.py` | Extracted shared code |
| `install_scripts/install_quest3.sh` | `install_scripts/install_pico.sh` | No XRT SDK, adds websockets + TLS |
