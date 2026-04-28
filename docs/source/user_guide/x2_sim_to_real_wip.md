# X2 Ultra Sim-to-Real WIP — Status & Hand-off Notes

**Branch:** `deploy_sim_to_real`
**Last update:** 2026-04-27
**Owner:** sitaram

This document is the running notebook for the AgiBot X2 Ultra
sim-to-real bring-up. It is intentionally written as a hand-off so the
work can be paused (currently to chase the IsaacLab → MuJoCo sim-sim
gap) and resumed without re-deriving context.

For the per-shift operator checklist see
[`x2_first_real_robot.md`](x2_first_real_robot.md). For architecture
and CLI reference see
[`../references/x2_deployment_code.md`](../references/x2_deployment_code.md).

---

## TL;DR

- ONNX deploy on a real X2 Ultra is **stable and safe** end-to-end:
  yaw-anchored reference, IsaacLab-matched action clip, soft-start
  ramp in, soft-exit ramp out, MC handoff is clean (no red flashing
  light on the last 4 powered runs of `take_a_sip`).
- Active-balance behaviour (push-recovery, dynamic motion) is **not yet
  visible on hardware**. The 16k checkpoint hits the ±20 rad
  `action_clip` ceiling for >90 % of ticks and the
  `--max-target-dev 0.30 rad` clamp throttles the resulting commanded
  delta. Increasing `--max-target-dev` toward IsaacLab's measured peak
  (~1.30 rad on `take_a_sip`) is the next powered-run task.
- Work paused here to fix the **IsaacLab vs MuJoCo sim-sim parity
  gap**. Mismatch in the sim-sim loop is the most likely upstream
  cause of "policy clips at 20 in IsaacLab too" and "deploy is
  numerically stable but visually unresponsive".

---

## What works today

### Code

| Area | File(s) | Status |
|---|---|---|
| Yaw-anchor (virtual RSI) | `gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/math_utils.hpp`, `include/reference_motion.hpp`, `src/reference_motion.cpp`, `src/x2_deploy_onnx_ref.cpp` | ✅ Unit-tested (`TestPklMotionYawAnchor`); validated on robot via 81.75° anchor on `take_a_sip` |
| IsaacLab-parity action clip | `src/x2_deploy_onnx_ref.cpp` (`--action-clip`, default 20.0) | ✅ Mirrors `ManagerEnvWrapper.step` `torch.clip(env_actions, -20, 20)` |
| Soft-start ramp in | `--ramp-seconds`, `SoftStartRamp` in `safety.hpp` | ✅ Pre-existing |
| Soft-exit ramp out | `src/x2_deploy_onnx_ref.cpp` `State::RAMP_OUT`, `--return-seconds` (default 2.0s) | ✅ Validated 4× on `take_a_sip`, no red light at MC handoff |
| Per-joint hard clamp | `--max-target-dev RAD` | ✅ Working but currently throttling policy authority (see Open Issues §1) |
| Tilt watchdog | `--tilt-cos`, `TiltWatchdog` | ✅ Pre-existing |
| Obs dump + parity tool | `src/x2_deploy_onnx_ref.cpp` `--obs-dump`, `gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py` | ✅ Used to surface the yaw-anchor bug; reusable for any future obs divergence |
| IsaacLab GT capture | `gear_sonic/scripts/dump_isaaclab_step0.py`, `dump_isaaclab_trajectory.py` | ✅ Single-step + per-tick trajectory dumps; the trajectory script wraps `os._exit` so `eval_agent_trl.py` doesn't drop the data |
| MuJoCo eval driver | `gear_sonic/scripts/eval_x2_mujoco_onnx.py` | ✅ Mirrors `eval_x2_mujoco.py` but consumes the deployed ONNX directly |
| MuJoCo ↔ ROS bridge | `gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py` | ✅ Lets the `agi_x2_deploy_onnx_ref` ROS node drive a MuJoCo sim under the same topics it would use on hardware |
| Pre-flight + monitor | `gear_sonic_deploy/scripts/x2_preflight.py`, `x2_action_monitor.py` | ✅ pre-flight green; monitor has a known cosmetic bug (TODO in file header) |
| Codegen for policy params | `gear_sonic_deploy/scripts/codegen_x2_policy_parameters.py` | ✅ Regenerates `policy_parameters.hpp` from the run's `config.yaml` |
| Motion → deploy format | `gear_sonic_deploy/scripts/export_motion_for_deploy.py` (`.pkl` → `.x2m2`) | ✅ Used for `idle_stand`, `take_a_sip` |

### Build & test

- `colcon build --packages-select agi_x2_deploy_onnx_ref` — green inside
  the `docker_x2` container.
- Offline unit suite — green:
  ```
  TestQuatMath
  TestProprioceptionPriming
  TestProprioceptionTermOrderAndAging
  TestStandStillTokenizerSize
  TestPklMotionYawAnchor
  ```

### On-robot validation

| Date | Motion | Duration | Outcome |
|---|---|---|---|
| 2026-04-22 | `idle_stand` | 1 s, 2 s, 5 s | All clean. `--max-target-dev 0.03` saturates but no danger. |
| 2026-04-22 | `take_a_sip` | 5 s, `--max-target-dev 0.30` | Clean run. Original red-light incident at MC handoff — root cause: no return-to-default ramp. |
| 2026-04-27 | `take_a_sip` | 5 s × 4 runs, `--max-target-dev 0.30`, `--return-seconds 2.0` | All clean, no red light, MC happily resumed walking. |

---

## Open issues / known gaps

### 1. Policy authority is gated by `--max-target-dev`

The 16k checkpoint emits `|action_il|` up to ~33 rad per tick during
`take_a_sip` and the IsaacLab-parity clamp truncates to ±20. Even with
the trained `x2_action_scale[i]`, this can produce target-pos
deltas of O(0.5–1.3 rad) per joint. Today we run with
`--max-target-dev 0.30` for safety, which rejects ≥50 % of those
commands. Next powered run should follow the documented ladder:

```
0.30  (current; pre-flight pose check)
0.50  (mid-range; expect visible arm motion on take_a_sip)
1.00  (close to IsaacLab P95)
1.50  (≥ IsaacLab peak; effectively no clamp on this motion)
```

Run each step at least twice with `--return-seconds 2.0` and a `take_a_sip`
or `eat_hot_dog` motion. Promote to the next step only on a clean
handoff.

### 2. Robot doesn't react to pushes

Even after the action-clip patch, on-gantry pushes don't trigger
anything that looks like active balancing. The IsaacLab trajectory dump
shows the policy *does* react to disturbances in sim. Two suspects:

- **(a)** The deploy still under-reports IMU velocity / contact-related
  obs vs IsaacLab. Re-run `compare_deploy_vs_isaaclab_obs.py` with the
  robot held tilted (not just standing) to check. We have only ever
  diffed the standing-pose dump.
- **(b)** Sim-sim gap (see next section): if the IsaacLab policy is
  actually unhealthy and only "looks fine" because IsaacLab has lower
  effective dynamics, deploy stability won't help.

Don't try to debug (2) further until the sim-sim parity story is
closed.

### 3. `x2_action_monitor.py` shows `max|step|=0.000`

500 Hz writer ZOH + `.3f` print precision rounds legitimate steps to
zero. Cosmetic only — `tick.csv` and `target_pos.csv` are still
correct. There's a TODO comment at the top of
`gear_sonic_deploy/scripts/x2_action_monitor.py` describing the fix
(switch to scientific notation or compute over a 50 Hz decimated view).

### 4. Ctrl-C bypasses `RAMP_OUT`

The new `--return-seconds` ramp only fires on `--max-duration`.
Operator Ctrl-C still calls `rclcpp::shutdown()` immediately because
the default rclcpp signal handler doesn't know about our state
machine. Fix: install a custom SIGINT handler that flips
`state_ -> RAMP_OUT` and lets the executor drain. Low risk, ~30 lines.

### 5. C++ test suite is not wired into colcon

`test_obs_builder` is built standalone via the
`build_offline/` CMake config. Need to register it as a `gtest` target
(or even just an `add_executable` test in a colcon `IfBuildingTests`
block) so `colcon test --packages-select agi_x2_deploy_onnx_ref`
actually runs it.

### 6. `eval_agent_trl.py` wrap is fragile

`dump_isaaclab_trajectory.py` monkey-patches `os._exit` to flush
trajectory data because `eval_agent_trl.py` skips Python's `atexit`.
This is brittle — a future refactor of `eval_agent_trl.py` will silently
disable the dump. Move the trajectory hook into `eval_agent_trl.py`
itself behind a `--dump-trajectory` flag.

---

## Suspected root cause (sim-sim gap)

The strongest signal that the current pipeline has an **upstream**
problem is:

> The IsaacLab `ManagerEnvWrapper` clips actions at ±20 rad during
> training, and our `take_a_sip` rollout sits **at the clip** for
> >90 % of ticks. A healthy policy should not be saturated against the
> training-time safety clamp on the very motion it was trained on.

If MuJoCo sim-sim shows the **same** saturation, the ONNX export or
observation pipeline is wrong even before deploy. If MuJoCo shows
healthy actions, the IsaacLab → ONNX export drops something the
`agi_x2_deploy_onnx_ref` proprioception pipeline depends on.

This is the next thing to chase. Suggested entry points:

1. Run `gear_sonic/scripts/eval_x2_mujoco_onnx.py` on the same 16k
   checkpoint and dump per-tick `|action|`. Compare against
   `dump_isaaclab_trajectory.py` output.
2. Run `gear_sonic_deploy/scripts/compare_isaaclab_vs_mujoco_obs.py`
   for the standing pose. If observation diff > 1e-3 anywhere, fix it
   before continuing.
3. Re-export the ONNX with `gear_sonic_deploy/scripts/reexport_x2_onnx.sh`
   if a per-step parity test against the live PyTorch policy fails.

---

## Resuming this branch

When sim-sim is sorted:

1. Rebase `deploy_sim_to_real` onto whatever branch landed the
   sim-sim fixes (likely `x2-deploy` or a sibling).
2. Re-run the offline parity:
   ```bash
   gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py \
     --deploy   logs/x2/obs_dump_idle_yawanchor.bin \
     --isaaclab /tmp/x2_step0_isaaclab_idle.pt \
     --rerun-onnx gear_sonic_deploy/models/x2_sonic_16k.onnx
   ```
   Expect each named slot ≤ 1e-4 from IsaacLab.
3. On-robot smoke test ladder (see `x2_first_real_robot.md`).
4. Walk `--max-target-dev` up the ladder in §1 above.
5. If push-recovery still doesn't appear, schedule a controlled tilt
   test and capture `compare_deploy_vs_isaaclab_obs.py` output for the
   tilted pose (currently we've only ever dumped the standing pose).

---

## Useful commands cheat sheet

Powered run with current safest settings:

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model /workspace/sonic/gear_sonic_deploy/models/x2_sonic_16k.onnx \
    --motion /workspace/sonic/gear_sonic_deploy/data/motions_x2m2/x2_ultra_take_a_sip.x2m2 \
    --autostart-after 5 --max-duration 5 \
    --max-target-dev 0.30 --ramp-seconds 2.0 --tilt-cos -0.3 \
    --return-seconds 2.0 \
    --log-dir /workspace/sonic/logs/x2/takesip_$(date +%Y%m%d_%H%M%S)
```

IsaacLab single-step ground truth:

```bash
python gear_sonic/scripts/dump_isaaclab_step0.py \
    +checkpoint=$HOME/x2_cloud_checkpoints/run-20260420_083925 \
    ++dump_path=/tmp/x2_step0_isaaclab_idle.pt
```

IsaacLab full trajectory (500 steps):

```bash
python gear_sonic/scripts/dump_isaaclab_trajectory.py \
    +checkpoint=$HOME/x2_cloud_checkpoints/run-20260420_083925 \
    ++dump_path=/tmp/x2_traj_takesip.pt \
    ++max_render_steps=500
```

Offline unit tests (in container):

```bash
docker exec docker_x2-x2sim-run-... bash -lc '
    source /opt/ros/humble/setup.bash &&
    source /ros2_ws/install/setup.bash &&
    source /workspace/sonic/gear_sonic_deploy/install/setup.bash &&
    /workspace/sonic/gear_sonic_deploy/build/agi_x2_deploy_onnx_ref/test_obs_builder
'
```
