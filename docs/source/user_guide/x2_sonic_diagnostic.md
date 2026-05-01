# X2 SONIC vs X1 Recipe: File-Based Diagnostic

**Source comparison:** `agibot_x1_train` (cloned `main`) vs your `GR00T-WholeBodyControl` branch `wip/mujoco-experiments-20260420`. All numbers below are taken from actual files, not estimates.

## Headline finding

You've already done substantially more careful sim-to-real engineering than the X1 recipe contains. Your X2 MJCF has per-joint armature classes, frictionloss, foot collision spheres, documented experiments (G13, G14, G16b), and your eval script does proper PD with action_scale derived from effort limits. **The asset is not your problem.**

Your most likely gap is **in the training-side configuration** — specifically, what's in your IsaacSim training recipe vs what AgiBot's X1 uses. The X1 file `humanoid/envs/x1/x1_dh_stand_config.py` is 414 lines of carefully-tuned training config that goes well beyond what most SONIC recipes do.

This doc walks through exactly what the X1 file contains and what your training pipeline likely needs to match.

---

## Critical: the official AgiBot X2 URDF you should download

AgiBot publishes an official URDF paired with their AimDK SDK. URL (per their docs page `https://x2-aimdk.agibot.com/en/latest/get_sdk/index.html`):

```
https://x2-aimdk.agibot.com/en/latest/_downloads/2ffc9785259556f409e385974a7a0461/X2_URDF-v1.3.0.zip
```

### Why this matters

Your committed URDF (`gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra.urdf`, 36KB) has:
- **No `<dynamics damping= friction=>` tags anywhere** — same as the X1 URDF
- **`<limit effort="0" velocity="0">` placeholders** instead of real limits
- **Possibly outdated** vs the v1.3.0 release

These are CAD-export characteristics. You may have been working from an earlier version of the URDF that AgiBot since revised. v1.3.0 is a versioned release, which strongly implies they update inertials, limits, and possibly mesh simplifications across versions.

### What to check inside the zip when you download it

1. **Variant filenames.** Look for `x2_ultra_simple_collision.urdf`, `*_simplified.urdf`, `*_collision.urdf`, `*_visual.urdf`. Vendors typically ship multiple variants — the simplified-collision one would replace mesh collisions with capsules, which is ~10× faster in MuJoCo and produces more stable contact dynamics.

2. **Bundled MJCF.** If the zip includes an `.xml` MJCF alongside the URDF, that's gold — AgiBot's own MuJoCo model with their internal armature/damping/contact tuning. Compare against your `x2_ultra.xml` line by line.

3. **Specs / system ID files.** Any `joint_specs.csv`, motor torque-speed curve PDFs, or measured-inertia tables. These directly resolve the H2-placeholder armature problem in your `x2_ultra.py:11` comment.

4. **Diff against your committed URDF.** Specifically:
   - Are `<dynamics>` tags present with non-zero damping/friction?
   - Are `<limit effort= velocity=>` real values now?
   - Did inertia tensors change?
   - Are there `<mimic>` joints (closed kinematic chains)?

5. **The "simple_collision" variant you mentioned forgetting.** If it exists in this zip, this is likely the one you remember. Compare its collision geometry against your MJCF — if it uses capsules where your MJCF uses meshes, swapping in capsule collisions could meaningfully reduce MuJoCo contact instability.

### Realistic expectations

This is an SDK-bundled URDF intended for ROS-side IK/visualization/planning. CAD-export URDFs without dynamics tags are common for vendor SDK releases — runtime calibration values aren't always exposed via URDF. **Don't be surprised if `<dynamics>` tags are still missing in v1.3.0.** The included MJCF (if any) is the more important artifact because MJCF *requires* explicit damping/armature/friction.

---

## The X1 PD gains (the actual values)

From `x1_dh_stand_config.py:147-150`:

```python
stiffness = {'hip_pitch_joint': 30, 'hip_roll_joint': 40, 'hip_yaw_joint': 35,
             'knee_pitch_joint': 100, 'ankle_pitch_joint': 35, 'ankle_roll_joint': 35}
damping = {'hip_pitch_joint': 3, 'hip_roll_joint': 3.0, 'hip_yaw_joint': 4,
           'knee_pitch_joint': 10, 'ankle_pitch_joint': 0.5, 'ankle_roll_joint': 0.5}
action_scale = 0.5
decimation = 10  # 1000Hz physics, 100Hz policy
```

**Compare with your X2 setup** (`x2_ultra.py:22-41`, computed from armature):

```python
NATURAL_FREQ = 10 * 2 * pi   # ~62.83 rad/s
DAMPING_RATIO = 2.0          # critically damped × 2

# Hip/knee: armature=0.0251 → KP = 0.0251 × 62.83² = 99.1, KD = 2 × 2 × 0.0251 × 62.83 = 6.31
# Ankle:    armature=0.0036 → KP = 0.0036 × 62.83² = 14.2, KD = 2 × 2 × 0.0036 × 62.83 = 0.91
```

| Joint group | X1 KP | X2 (your) KP | X1 KD | X2 (your) KD |
|---|---|---|---|---|
| Hip pitch | 30 | ~99 | 3 | ~6.3 |
| Hip roll | 40 | ~99 | 3 | ~6.3 |
| Hip yaw | 35 | ~99 | 4 | ~6.3 |
| Knee | 100 | ~99 | 10 | ~6.3 |
| Ankle pitch | 35 | ~14 | 0.5 | ~0.9 |
| Ankle roll | 35 | ~14 | 0.5 | ~0.9 |

**Observations:**
1. Your hip-pitch KP (99) is **3.3× X1's** (30). X1 uses *softer* hips. This will dramatically change gait character — your policy learns to depend on stiff hip control, which then needs to transfer through MuJoCo's slightly different solver.
2. X1 uses **per-joint Kd values** that are NOT a fixed ratio of Kp. Your formula `KD = 4 × armature × ω` produces a uniform damping ratio, which is mathematically clean but is not what the working recipe uses.
3. Your ankle KP (14) is **less than half X1's** (35). For motion tracking, soft ankles + stiff PD elsewhere is a reasonable choice — but only if your training matched this assumption.
4. X1 `action_scale = 0.5`. Your `action_scale = 0.25 * effort / kp_train` — this is a *non-uniform* scale, which is fine, but it's a fundamentally different decision than what the proven recipe uses.

**The "armature × ω²" approach is theoretically appealing but practically untested at this scale.** The X1 numbers are empirically tuned over many real-robot iterations. Your formula gives a self-consistent set, but consistency is not the same as "matches reality."

**Highest-leverage experiment:** Try training X2 with X1-style per-joint hand-tuned KP/KD (scaled for X2's larger torques), instead of the formula-derived values. Specifically: Hip KP ~30-40, Knee KP ~100, Ankle KP ~35, with the small Kd values X1 uses.

---

## Domain Randomization: what the X1 file does that yours likely doesn't

This is the section of `x1_dh_stand_config.py` that I suspect is your biggest gap. Lines 176-277 — **100 lines of DR config**. Here's the full inventory:

### Always on:
```python
randomize_friction = True;        friction_range = [0.2, 1.3]
restitution_range = [0.0, 0.4]
push_robots = True;               push_interval_s = 4
push_duration = [0, 0.05, 0.1, 0.15, 0.2, 0.25]   # curriculum
max_push_vel_xy = 0.2;            max_push_ang_vel = 0.2
randomize_base_mass = True;       added_mass_range = [-3, 3]   # ±3 kg
randomize_com = True;             com_displacement_range = [±0.05, ±0.05, ±0.05]   # 5cm in all axes
randomize_gains = True;           stiffness_multiplier_range = [0.8, 1.2]
                                  damping_multiplier_range = [0.8, 1.2]
randomize_torque = True;          torque_multiplier_range = [0.8, 1.2]
randomize_link_mass = True;       added_link_mass_range = [0.9, 1.1]   # per-link
randomize_motor_offset = True;    motor_offset_range = [-0.035, 0.035]   # 2°
randomize_joint_friction = True;  joint_friction_range = [0.01, 1.15]
randomize_joint_damping = True;   joint_damping_range = [0.3, 1.5]
randomize_joint_armature = True;  joint_armature_range = [0.0001, 0.05]
randomize_coulomb_friction = True
joint_coulomb_range = [0.1, 0.9]
joint_viscous_range = [0.05, 0.1]
```

### The latency suite (this is the killer):
```python
add_lag = True
randomize_lag_timesteps = True
lag_timesteps_range = [5, 40]      # action lag: 5-40 timesteps at 1000Hz = 5-40ms

add_dof_lag = True                 # observation (DOF) lag separate from action lag
randomize_dof_lag_timesteps = True
dof_lag_timesteps_range = [0, 40]

add_imu_lag = False                # IMU lag separate again
randomize_imu_lag_timesteps = True
imu_lag_timesteps_range = [1, 10]
```

**Why this matters specifically for SONIC + MuJoCo failure:**

If your training has zero or minimal latency randomization, your policy assumes instantaneous control. MuJoCo has different effective latency than IsaacSim because of its CPU-based stepping. The real X2 has 5–40ms of communication+control latency. A policy that wasn't trained against latency will oscillate or go unstable on anything other than the simulator it trained in.

### Action items:

1. **Open your training config** for the X2 SONIC run. Search for `randomize_`, `domain_rand`, `lag`, `delay`, `noise`. Count how many are enabled.
2. **If the count is fewer than ~10 randomization knobs**, that's your gap. Even more critical: **action lag and observation lag.** SONIC training pipelines often skip these because they complicate transformer training, but they are mandatory for sim-to-sim and sim-to-real.
3. **Per-joint armature randomization range `[0.0001, 0.05]`.** Yours is fixed per joint. The X1 recipe randomizes armature 500× (across the range) during training. This is essentially "the policy learns to handle any reasonable rotor inertia," which makes it robust to MuJoCo's slightly different effective armature handling.

---

## Observation noise (X1 does this, the values matter)

From `x1_dh_stand_config.py:111-122`:

```python
add_noise = True
noise_level = 1.5    # multiplies all values below

dof_pos = 0.02       # 0.02 × 1.5 = 0.03 rad noise
dof_vel = 1.5        # 1.5 × 1.5 = 2.25 rad/s noise — NOTE: this is huge
ang_vel = 0.2        # 0.2 × 1.5 = 0.3 rad/s
lin_vel = 0.1
quat = 0.1
gravity = 0.05
```

The `dof_vel = 1.5 × 1.5 = 2.25` value is striking — that's enormous noise on joint velocity. It reflects the reality that real-robot joint velocity (computed from finite differences of position) is extremely noisy at high frequencies. **A SONIC training pipeline that uses clean joint velocities from IsaacSim and then deploys against MuJoCo's slightly-different velocity computation will fail hard.**

Verify: does your training observation pipeline add velocity noise of this magnitude?

---

## Reward regularization (the unsexy stuff that matters)

From `x1_dh_stand_config.py:347-356`, the X1 regularization weights:

```python
action_smoothness = -0.002
torques = -8e-9
dof_vel = -2e-8
dof_acc = -1e-7
collision = -1.0
dof_vel_limits = -1
dof_pos_limits = -10.0   # very strong
dof_torque_limits = -0.1
```

These tiny negative weights are not optional. They prevent the policy from learning behaviors that *only* work in the lax PhysX solver: high-frequency action chattering, near-limit joint excursions, large transient torques. SONIC's tracking-focused loss will let these slide unless explicit regularization is added.

**Action item:** Check your SONIC reward/loss specification. If it's almost entirely tracking error with little action_rate, dof_acc, or torque penalty — that's a likely contributor to the IsaacSim → MuJoCo gap.

---

## The X1 MJCF: what's actually in it

From `xyber_x1_serial.xml` (which is shorter than your X2 MJCF — 358 vs 474 lines):

```xml
<option timestep="0.001" />

<!-- Per-joint damping is set INSIDE each joint, not in defaults: -->
<joint name="left_hip_pitch" type='hinge' damping='1' range="-3.14 3.14" />
<!--                                       ↑ uniform damping=1 for ALL joints -->

<!-- Default contact behavior: -->
<geom contype="0" conaffinity="0" solref="0.005 1" condim="3" friction="1 1" />

<!-- Foot is just 4 small spheres (size 0.002): -->
<geom type="sphere" size="0.002" pos="0.03 0.0408 0.07" class="collision"/>
<!-- Plus 4 visual-only spheres of size 0.02 -->

<!-- No armature anywhere. -->
<!-- No frictionloss anywhere. -->
<!-- No solref/solimp tuning per geom. -->
```

**The X1 MJCF is dramatically simpler than yours.** Just `damping=1` per joint, no armature, no frictionloss, basic foot spheres. This works because **everything sophisticated is in the training-side DR**, not in the MJCF.

**Architectural insight:** Two equally valid philosophies exist:
- **X1 / Humanoid-Gym philosophy:** simple MJCF, sophisticated training-side DR randomizes armature/damping/friction over wide ranges so policy is robust to *any* reasonable contact/joint model.
- **Your philosophy:** richly-modeled MJCF with system-identified armature, fixed gains, less DR — bet on accurate modeling.

Both can work, but the second requires the system identification to actually be correct. From your `x2_ultra.py:11`: *"Exact rotor inertia (armature) values are not available from the vendor; using H2 motor constants as starting estimates."* So your armature values are placeholders, not measured. **Without measured values, the X1 philosophy (wide armature DR) is much safer than the modeled philosophy.** And the v1.3.0 SDK URDF download may finally give you the measured values to make the modeled philosophy actually work.

---

## The X1 sim2sim.py: the critical PD line

From `humanoid/scripts/sim2sim.py:129-133`:

```python
def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    torque_out = (target_q + cfg.robot_config.default_dof_pos - q) * kp - dq * kd
    return torque_out

# Used as:
target_q = action * env_cfg.control.action_scale
tau = pd_control(target_q, q, kps, target_dq=zeros, dq, kds, cfg)
tau = np.clip(tau, -tau_limit, tau_limit)
data.ctrl = tau
mujoco.mj_step(model, data)
```

**Note carefully:** they use `<motor>` actuators (no built-in PD), and PD is computed in Python. `data.ctrl = tau` is direct torque commanding.

**Compare to your eval_x2_mujoco.py** — your `_compute_gains_and_scales()` produces per-joint KP/KD and action_scale, and you do PD in Python too. Structurally similar. The key thing to verify:

1. Your `action_scale` formula: `0.25 * effort / kp_train`. For a hip joint: `0.25 * 120 / 99 = 0.30`. For an ankle: `0.25 * 36 / 14 = 0.64`. **This means policy actions of [-1, 1] map to joint target offsets of [-0.30, 0.30] rad on hips and [-0.64, 0.64] rad on ankles — over 36° on the ankle!** That's a huge command range and may explain why the policy can produce stable behavior in IsaacSim (where PhysX dampens it) but unstable behavior in MuJoCo (where the same large commands amplify through different contact dynamics).

2. X1 uses uniform `action_scale = 0.5`. Combined with their lower KPs (30-100), policy actions of [-1, 1] map to [-0.5, 0.5] rad target offsets, then PD with KP=30 gives max torque demand of `0.5 × 30 = 15 N·m` on a hip — *far below* the joint's 150 N·m capability. This leaves enormous PD headroom for disturbance rejection.

3. Your config: hip action [-1, 1] × 0.30 = ±0.30 rad target, KP=99 → max correction torque `0.30 × 99 = ~30 N·m`. Still well below 120 N·m limit, but the *ratio* is much higher than X1. **Your PD has less headroom for disturbance rejection.**

**Hypothesis:** When MuJoCo introduces small contact-dynamics disagreements, your stiffer-PD-with-tighter-action-scale system has less margin to absorb them than X1's softer-PD-with-wider-action-scale system. The policy hits torque saturation or destabilizes.

**Experiment to try:** Halve all your KPs and double the action_scale. This produces the same nominal behavior (same target trajectories) but with more PD headroom. If MuJoCo behavior improves, this is your gap.

---

## What I'd do, in order

1. **Download the AgiBot X2 v1.3.0 URDF zip** from the URL above. Diff against your committed `x2_ultra.urdf`. Look for: a `*_simple_collision.urdf` variant, a bundled MJCF, system-ID specs files. This is a couple of hours that could resolve the placeholder-armature problem at the source.

2. **Look at your training-side DR config.** This is the highest-probability gap. Your asset/eval code already shows sophistication; if your training was done with default SONIC DR (which is typically narrow), that's the smoking gun. Specifically check:
   - Action-side latency (5–40ms range)
   - Observation-side latency
   - Joint armature randomization (wide range, especially while your armature values are still placeholders)
   - Coulomb friction randomization
   - Torque multiplier randomization

3. **Try the softer-PD experiment.** Train X2 with X1-style PD: Hip KP=40, Knee KP=100, Ankle KP=35, action_scale=0.5. If this trains and transfers to MuJoCo better, the issue was PD/scale interaction with contact solver, not asset fidelity.

4. **Add observation noise if not present.** Particularly `dof_vel` noise of order 1.0+ rad/s. Real robots have this, MuJoCo's velocity computation has different numerical character than IsaacSim's, and policies trained without this noise overfit to clean state.

5. **Verify regularization rewards exist in your SONIC loss.** action_smoothness, dof_acc, torque magnitude penalties.

6. **Last resort: drop the modeled armature, use X1-style uniform damping=1.** If your placeholder armature values are wrong, fixed-armature MJCF is worse than no-armature with wide DR. This is a one-line MJCF change to test.

---

## What this DOESN'T explain

If after all of the above you still see the gap, then the remaining suspects are:
- The actual joint axis directions or link inertias in your X2 USD don't match the real X2 (verifiable when you have hardware via the gravity-comp test from earlier).
- The reference motion retargeting produces motions infeasible for X2's specific kinematics.
- SONIC's transformer architecture is doing something specific to its training distribution that doesn't transfer with the same DR ranges that worked for an MLP policy.

But given how much careful work is already in your branch, I'd bet against asset issues and toward training-side DR being incomplete.
