#!/usr/bin/env python3
"""Isaac Lab simulator for GEAR-SONIC.

Drop-in replacement for run_sim_loop.py (MuJoCo). Same two-terminal workflow:
  Terminal 1: python run_sim_isaaclab.py     (this script)
  Terminal 2: bash deploy_sonic.sh sim       (unchanged)

Mirrors base_sim.py exactly:
  - ElasticBand on pelvis (9=toggle, 7/8=height, Backspace=reset)
  - Torques: tau_ff + kp*(q_cmd-q) + kd*(dq_cmd-dq), clamped to MuJoCo frcrange
  - Zero torques when no LowCmd received (band holds)
  - Auto-reset on fall (pelvis < 0.2m)
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
import select
import signal
import sys
import termios
import threading
import tty
import time

_ORIGINAL_TERM = None
try:
    _ORIGINAL_TERM = termios.tcgetattr(sys.stdin)
except Exception:
    pass


def _restore_term(*_a):
    if _ORIGINAL_TERM is not None:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSANOW, _ORIGINAL_TERM)
        except Exception:
            pass


atexit.register(_restore_term)
for _s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_s, lambda *a: (_restore_term(), sys.exit(1)))

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEAR_SONIC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, GEAR_SONIC_ROOT)

# ── AppLauncher (must precede isaaclab imports) ──────────────────────────

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Isaac Lab GEAR-SONIC Sim")
parser.add_argument("--spawn_height", type=float, default=0.793)
parser.add_argument("--domain_id", type=int, default=0)
parser.add_argument("--interface", type=str, default="lo")
parser.add_argument("--no_realtime", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Imports after AppLauncher ────────────────────────────────────────────

import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

from gear_sonic.utils.isaaclab_sim.g1_cfg import (  # noqa: E402
    G1_SONIC_29DOF_CFG, SONIC_JOINT_ORDER,
    SONIC_DEFAULT_ANGLES, SONIC_DEFAULT_ANGLES_ARRAY,
    SONIC_DEFAULT_KP, SONIC_DEFAULT_KD,
    SONIC_TORQUE_LIMITS, NUM_BODY_MOTORS, SIM_DT, FALL_HEIGHT,
)
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import (  # noqa: E402
    ElasticBand, UnitreeSdk2Bridge,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402


# ── Keyboard listener ───────────────────────────────────────────────────

class KeyListener:
    def __init__(self):
        self._keys, self._lock, self._stop = [], threading.Lock(), False
        self._old = None
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._old = termios.tcgetattr(sys.stdin)
        atexit.register(self._restore)
        tty.setcbreak(sys.stdin.fileno())
        self._t.start()

    def _restore(self):
        if self._old:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSANOW, self._old)
            except Exception:
                pass
            self._old = None

    def stop(self):
        self._stop = True
        self._restore()

    def _run(self):
        while not self._stop:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                with self._lock:
                    self._keys.append(sys.stdin.read(1))

    def pop(self):
        with self._lock:
            k = list(self._keys)
            self._keys.clear()
        return k


# ── Jitter tracker ──────────────────────────────────────────────────────

class Jitter:
    def __init__(self, names, n=200):
        self.names, self.n = names, n
        self.nj = len(names)
        self._pdq = np.zeros(self.nj)
        self._pq = np.zeros(self.nj)
        self._reset()

    def _reset(self):
        self._c = 0
        self._mdq = self._mddq = self._mtau = self._mqe = self._mqd = 0.0
        self._mdq_j = self._mddq_j = self._mtau_j = self._mqe_j = self._mqd_j = ""
        self._clip = 0
        self._rms = 0.0
        self._mrw = self._mrv = 0.0
        self._st = []

    def update(self, q, dq, tau, qc, rw, rv, tw):
        self._c += 1
        a = np.abs(dq); i = int(np.argmax(a))
        if a[i] > self._mdq: self._mdq, self._mdq_j = float(a[i]), self.names[i]
        a = np.abs((dq - self._pdq) / SIM_DT); i = int(np.argmax(a))
        if a[i] > self._mddq: self._mddq, self._mddq_j = float(a[i]), self.names[i]
        a = np.abs(q - self._pq); i = int(np.argmax(a))
        if a[i] > self._mqd: self._mqd, self._mqd_j = float(a[i]), self.names[i]
        if tau is not None:
            a = np.abs(tau); i = int(np.argmax(a))
            if a[i] > self._mtau: self._mtau, self._mtau_j = float(a[i]), self.names[i]
            if np.any(a > SONIC_TORQUE_LIMITS): self._clip += 1
        if qc is not None:
            a = np.abs(qc - q); i = int(np.argmax(a))
            if a[i] > self._mqe: self._mqe, self._mqe_j = float(a[i]), self.names[i]
        self._rms += float(np.mean(dq**2))
        self._mrw = max(self._mrw, float(np.linalg.norm(rw)))
        self._mrv = max(self._mrv, float(np.linalg.norm(rv)))
        self._st.append(tw)
        self._pdq, self._pq = dq.copy(), q.copy()

    def ready(self):
        return self._c >= self.n

    def report(self, step, fps, h, bon, blen, conn):
        c = max(self._c, 1)
        s = np.array(self._st) if self._st else np.array([0.0])
        qe = (f"    max |q_err|  = {self._mqe:8.4f} rad     ({self._mqe_j})"
              if self._mqe > 0 else "    max |q_err|  =      N/A")
        r = "\n".join([
            f"\n{'~'*72}",
            f" step={step} | fps={fps:.0f} | h={h:.2f}m | band={'ON' if bon else 'OFF'}"
            f"(len={blen:+.1f}) | {'connected' if conn else 'waiting...'}",
            f"  JITTER (last {c} steps):",
            f"    max |dq|     = {self._mdq:8.2f} rad/s   ({self._mdq_j})",
            f"    max |ddq|    = {self._mddq:8.1f} rad/s2  ({self._mddq_j})",
            f"    max |dq|/step= {self._mqd:8.4f} rad     ({self._mqd_j})",
            f"    rms  dq      = {math.sqrt(self._rms/c):8.4f} rad/s",
            f"    max |tau|    = {self._mtau:8.1f} Nm PRE-CLAMP ({self._mtau_j})"
            f"  clipped {100*self._clip/c:.0f}%",
            qe,
            f"    root |w|     = {self._mrw:8.2f} rad/s",
            f"    root |v|     = {self._mrv:8.2f} m/s",
            f"    step time    = {np.mean(s)*1e3:5.1f}ms avg,"
            f" {np.std(s)*1e3:5.2f}ms std, {np.max(s)*1e3:5.1f}ms max",
            f"{'~'*72}",
        ])
        self._reset()
        return r


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Isaac Lab GEAR-SONIC Simulator")
    print("=" * 60)

    sim = SimulationContext(SimulationCfg(dt=SIM_DT, render_interval=4))
    sim.set_camera_view(eye=[2, 0, 1.5], target=[0, 0, 0.5])

    # Ground + robot (equivalent to MuJoCo XML loading)
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    rcfg = G1_SONIC_29DOF_CFG.replace(prim_path="/World/G1")
    rcfg.init_state.pos = (0.0, 0.0, args.spawn_height)
    robot = Articulation(rcfg)

    sim.reset()
    nj = robot.num_joints
    print(f"[Robot] {nj} joints, bodies: {robot.body_names}")

    il = {n: i for i, n in enumerate(robot.joint_names)}
    s2il = [il.get(j) for j in SONIC_JOINT_ORDER]
    print(f"[DDS] Mapped {sum(1 for x in s2il if x is not None)}/{NUM_BODY_MOTORS} joints")

    pelvis_idx = next(i for i, n in enumerate(robot.body_names) if n == "pelvis")
    torso_idx = next(i for i, n in enumerate(robot.body_names) if n == "torso_link")

    # Elastic band (reuse from MuJoCo module)
    band = ElasticBand()
    band.point = np.array([0.0, 0.0, 1.0])

    # DDS bridge (reuse from MuJoCo module)
    ChannelFactoryInitialize(args.domain_id, args.interface)
    bridge = UnitreeSdk2Bridge({
        "ROBOT_TYPE": "g1_29dof", "NUM_MOTORS": NUM_BODY_MOTORS,
        "NUM_HAND_MOTORS": 7, "NUM_HAND_JOINTS": 7,
        "USE_SENSOR": False, "USE_JOYSTICK": False,
    })
    print(f"[DDS] Bridge ready (iface={args.interface})")

    ef = torch.zeros(1, 1, 3, dtype=torch.float32, device=robot.device)
    et = torch.zeros(1, 1, 3, dtype=torch.float32, device=robot.device)
    keys = KeyListener()
    keys.start()
    jitter = Jitter(SONIC_JOINT_ORDER)

    spawn_pose = torch.tensor(
        [[*rcfg.init_state.pos, *rcfg.init_state.rot]],
        dtype=torch.float32, device=robot.device)
    zero_vel = torch.zeros((1, 6), dtype=torch.float32, device=robot.device)

    init_jpos = torch.zeros((1, nj), dtype=torch.float32, device=robot.device)
    init_jvel = torch.zeros((1, nj), dtype=torch.float32, device=robot.device)
    for jn, ang in SONIC_DEFAULT_ANGLES.items():
        idx = il.get(jn)
        if idx is not None:
            init_jpos[0, idx] = ang

    prev_lv = np.zeros(3)

    def reset(re_band=False):
        nonlocal prev_lv
        if re_band:
            band.enable = True
            band.length = 0.0
        robot.write_root_pose_to_sim(spawn_pose)
        robot.write_root_velocity_to_sim(zero_vel)
        robot.write_joint_state_to_sim(init_jpos, init_jvel)
        robot.write_data_to_sim()
        prev_lv = np.zeros(3)
        with bridge.low_cmd_lock:
            bridge.low_cmd_received = False
        print(f"\n[SIM] RESET ({'band ON' if band.enable else 'standing'})")

    print("\n  9=toggle band | 7/8=height | Backspace=reset | Ctrl+C=quit\n")

    step = 0
    t0 = time.time()
    try:
        while simulation_app.is_running():
            lt = time.time()

            for ch in keys.pop():
                if ch == '9':
                    band.handle_keyboard_button(ch)
                elif ch == '7':
                    band.length -= 0.1
                    print(f'[Band] length={band.length:+.1f}')
                elif ch == '8':
                    band.length += 0.1
                    print(f'[Band] length={band.length:+.1f}')
                elif ch == '\x7f':
                    reset(re_band=True)

            # ── Read state (single GPU->CPU) ─────────────────────────
            d = robot.data
            sg = torch.cat([
                d.joint_pos[0], d.joint_vel[0],
                d.root_pos_w[0], d.root_quat_w[0],
                d.root_lin_vel_w[0], d.root_ang_vel_b[0],
                d.body_pos_w[0, pelvis_idx], d.body_quat_w[0, pelvis_idx],
                d.body_ang_vel_w[0, pelvis_idx],
                d.body_quat_w[0, torso_idx], d.body_ang_vel_w[0, torso_idx],
            ])
            sc = sg.cpu().numpy().astype(np.float64)
            o = 0
            jp = sc[o:o+nj]; o += nj
            jv = sc[o:o+nj]; o += nj
            rp = sc[o:o+3]; o += 3
            rq = sc[o:o+4]; o += 4
            rlv = sc[o:o+3]; o += 3
            rav = sc[o:o+3]; o += 3
            pp = sc[o:o+3]; o += 3
            pq = sc[o:o+4]; o += 4
            pw = sc[o:o+3]; o += 3
            tq = sc[o:o+4]; o += 4
            tw = sc[o:o+3]; o += 3
            h = rp[2]

            # SONIC-order joint data
            qs = np.zeros(NUM_BODY_MOTORS, dtype=np.float64)
            ds = np.zeros(NUM_BODY_MOTORS, dtype=np.float64)
            for si in range(NUM_BODY_MOTORS):
                idx = s2il[si]
                if idx is not None:
                    qs[si] = jp[idx]
                    ds[si] = jv[idx]

            # ── Torques from LowCmd ──────────────────────────────────
            tau_dbg = q_cmd_dbg = None
            eff = torch.zeros(nj, dtype=torch.float32, device=robot.device)

            with bridge.low_cmd_lock:
                has_cmd = bridge.low_cmd_received
                cmd = bridge.low_cmd if has_cmd else None

            if has_cmd and cmd is not None:
                qc = np.array([cmd.motor_cmd[i].q for i in range(NUM_BODY_MOTORS)])
                dc = np.array([cmd.motor_cmd[i].dq for i in range(NUM_BODY_MOTORS)])
                kp = np.array([cmd.motor_cmd[i].kp for i in range(NUM_BODY_MOTORS)])
                kd = np.array([cmd.motor_cmd[i].kd for i in range(NUM_BODY_MOTORS)])
                tf = np.array([cmd.motor_cmd[i].tau for i in range(NUM_BODY_MOTORS)])
                tau = tf + kp * (qc - qs) + kd * (dc - ds)
                tau_dbg = tau.copy()
                q_cmd_dbg = qc
                tau = np.clip(tau, -SONIC_TORQUE_LIMITS, SONIC_TORQUE_LIMITS)
                for si in range(NUM_BODY_MOTORS):
                    idx = s2il[si]
                    if idx is not None:
                        eff[idx] = float(tau[si])
            elif not band.enable:
                tau = (SONIC_DEFAULT_KP * (SONIC_DEFAULT_ANGLES_ARRAY - qs)
                       + SONIC_DEFAULT_KD * (0.0 - ds))
                tau_dbg = tau.copy()
                tau = np.clip(tau, -SONIC_TORQUE_LIMITS, SONIC_TORQUE_LIMITS)
                for si in range(NUM_BODY_MOTORS):
                    idx = s2il[si]
                    if idx is not None:
                        eff[idx] = float(tau[si])

            robot.set_joint_effort_target(eff.unsqueeze(0))

            # ── Elastic band ─────────────────────────────────────────
            if band.enable:
                pose = np.concatenate([pp, pq, rlv, pw])
                wr = band.Advance(pose)
                ef[0, 0, :] = torch.tensor(wr[:3], dtype=torch.float32)
                et[0, 0, :] = torch.tensor(wr[3:], dtype=torch.float32)
            else:
                ef[0, 0, :] = 0.0
                et[0, 0, :] = 0.0
            robot.set_external_force_and_torque(ef, et, body_ids=[pelvis_idx], is_global=True)

            # ── Step ─────────────────────────────────────────────────
            robot.write_data_to_sim()
            sim.step()
            robot.update(SIM_DT)

            if h < FALL_HEIGHT:
                print(f"\n[SIM] Fall! h={h:.3f}m")
                reset()

            # ── Publish DDS state ────────────────────────────────────
            acc = (rlv - prev_lv) / SIM_DT
            prev_lv = rlv.copy()
            obs = {
                "floating_base_pose": np.concatenate([rp, rq]),
                "floating_base_vel": np.concatenate([rlv, rav]),
                "floating_base_acc": np.concatenate([acc, np.zeros(3)]),
                "secondary_imu_quat": tq,
                "secondary_imu_vel": np.concatenate([np.zeros(3), np.zeros(3), tw]),
                "body_q": qs, "body_dq": ds,
                "body_ddq": np.zeros(NUM_BODY_MOTORS),
                "body_tau_est": np.zeros(NUM_BODY_MOTORS),
                "left_hand_q": np.zeros(7), "left_hand_dq": np.zeros(7),
                "right_hand_q": np.zeros(7), "right_hand_dq": np.zeros(7),
                "time": step * SIM_DT,
            }
            bridge.PublishLowState(obs)

            step += 1
            wt = time.time() - lt
            jitter.update(qs, ds, tau_dbg, q_cmd_dbg, rav, rlv, wt)
            if jitter.ready():
                print(jitter.report(step, step/(time.time()-t0), h,
                                    band.enable, band.length, bridge.low_cmd_received))

            if not args.no_realtime:
                el = time.time() - lt
                if el < SIM_DT:
                    time.sleep(SIM_DT - el)

    except KeyboardInterrupt:
        print("\n[SIM] Stopped.")
    finally:
        keys.stop()

    print(f"\n  {step} steps in {time.time()-t0:.1f}s ({step/(time.time()-t0):.0f} fps)\n")
    simulation_app.close()


if __name__ == "__main__":
    main()
