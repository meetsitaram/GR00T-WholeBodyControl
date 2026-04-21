# Sim-to-Sim: Deploying a Trained Policy in MuJoCo

This guide documents how to take a SONIC checkpoint trained in IsaacLab
and run it in MuJoCo for closed-loop evaluation, plus the collection of
subtle IsaacLab↔MuJoCo mismatches we hit doing exactly that for the
Agibot X2 Ultra. Every gotcha listed here was a real bug that produced
visually-plausible but completely broken behavior; check this list first
when porting a new embodiment.

The reference implementation is `gear_sonic/scripts/eval_x2_mujoco.py`.
The script is X2-specific in the sense that it loads X2's MJCF, default
pose, gain table, and DOF mappings — but the structure (RSI →
proprioception buffer → tokenizer obs → actor → PD → auto-reset) is
identical for any embodiment. Use it as the template.

## Quick Start

```bash
conda run -n env_isaaclab --no-capture-output python gear_sonic/scripts/eval_x2_mujoco.py \
    --checkpoint logs_rl/<run>/model_step_NNNNNN.pt \
    --motion gear_sonic/data/motions/<robot>_<motion>.pkl
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--init-frame N` | 0 | RSI motion frame to teleport into at every reset |
| `--fall-height H` | 0.4 m | Pelvis-z below this triggers an auto-reset |
| `--fall-tilt-cos C` | -0.3 | `gravity_body[z]` above this triggers an auto-reset (~72° tilt) |
| `--max-episode T` | 0 | If > 0, force-reset every T simulated seconds |
| `--speed S` | 1.0 | Motion phase speed multiplier |
| `--device` | cpu | Inference device for the actor |

Viewer controls: `SPACE` pause/resume, `R` manual reset, `V` toggle
tracking/free camera.

## How the Script Mirrors IsaacLab

The deployment loop reproduces what IsaacLab does on every episode:

1. **Reference State Initialization (RSI)** — at every reset, the robot
   is teleported to the motion's `--init-frame` state. This sets root
   pos/quat/lin_vel/ang_vel and joint pos/vel from the motion library.
2. **Closed-loop control** — each control tick the script reads MuJoCo
   state, builds the IsaacLab-format proprioception (history of the
   last `HISTORY_LEN` ticks) and tokenizer (next `NUM_FUTURE_FRAMES`
   reference frames in the body frame) observations, runs the actor
   (encoder + FSQ + decoder), and applies explicit PD control to track
   the policy's joint targets.
3. **Auto-reset on fall** — when the pelvis drops below
   `--fall-height` or the body tilts past `--fall-tilt-cos` the
   episode resets back to RSI. This is a coarser, height-based trigger
   than IsaacLab's tracking-error terminations used during training,
   so episode lengths in the viewer are **not** comparable to
   training/eval episode lengths — the deployment script lets the
   policy survive bad-tracking states that training would have
   terminated.

## Critical Gotchas

### G1. MuJoCo free-joint `qvel` is NOT all in world frame

This is the single most important pitfall. For a free joint:

| Slot | Frame |
|---|---|
| `qvel[0:3]` (linear) | **World** frame |
| `qvel[3:6]` (angular) | **Body-local** frame (NOT world!) |

This bites in two places:

1. **RSI write** — motion-lib stores `root_ang_vel_w` in world frame.
   You must convert to body before writing:
   ```python
   mj_data.qvel[3:6] = quat_rotate_inverse(root_quat_w_wxyz, root_ang_vel_w)
   ```
2. **Per-step proprioception** — IsaacLab's `base_ang_vel` term is
   already in the body frame. Read `mj_data.qvel[3:6]` directly and
   do **not** rotate by `quat_rotate_inverse(base_quat, ...)`. The
   wrong-assumption rotation produces a *double* rotation, corrupting
   the angular-velocity history every step. Symptom: policy looks
   confused, drifts off balance after a stride, never recovers — but
   the same checkpoint walks fine in IsaacLab.

Verification recipe: zero-gravity single-body MJCF, set `qvel[3:6]`,
let the sim integrate one timestep, and check how `qpos[3:7]` evolves
relative to body-frame and world-frame hypotheses.

### G2. Quaternion convention: wxyz everywhere, xyzw for SciPy only

- IsaacLab and MuJoCo both use **wxyz** (scalar-first) quaternions.
- `scipy.spatial.transform.Rotation.from_quat` / `as_quat` use **xyzw**
  (scalar-last) — a constant source of off-by-90° rotations.
- Use a `quat_rotate_inverse` that matches IsaacLab's `quat_apply_inverse`
  exactly. Don't substitute a SciPy-derived helper without converting
  the quaternion order, and double-check the sign convention. A sign-flip
  bug will make gravity in the body frame point *up* instead of down;
  symptom: the robot tries to stand on its head.

### G3. DOF reordering is required and the gather-index naming is confusing

IsaacLab and MuJoCo traverse the kinematic tree in different orders.
Maintain two index permutations in your embodiment file (e.g.
`gear_sonic/envs/manager_env/robots/<robot>.py`):

```python
# Gather index: result[il_pos] = source[IL_TO_MJ_DOF[il_pos]]
IL_TO_MJ_DOF = [...]   # used to convert MJ→IL
MJ_TO_IL_DOF = [...]   # used to convert IL→MJ
```

The naming describes "where to *gather from*", not "where to map to":

- **MJ → IL (state read):** `dof_pos_il = dof_pos_mj[IL_TO_MJ_DOF]`
- **IL → MJ (action write):** `action_mj = action_il[MJ_TO_IL_DOF]`

We had these swapped early on. The symptom was huge correlated
joint motion — the wrong permutation is still a meaningful permutation,
just with totally different per-joint meanings. When in doubt set up a
sentinel test: write a known per-joint pattern, round-trip it, and
verify you get back what you started with.

### G4. Default joint pose, action scale, and PD gains all matter

Pull these from the embodiment file:

- **`DEFAULT_JOINT_POS`** is the home pose. Joint targets are
  `target = DEFAULT_DOF + action * ACTION_SCALE`, **not** raw
  `action * ACTION_SCALE`. Forgetting the offset gives the policy a
  zero-initialized pose where (e.g.) knees are straight, leading to
  instant collapse on contact.
- **`ACTION_SCALE`** is per-joint. Typical humanoid configs use
  large values for hip/knee (~0.5) and smaller values for arms
  (~0.25). A single global scalar will keep the robot alive but
  produces noticeably stiff arms and over-driven legs.
- **`KP` / `KD`** must be derived the same way IsaacLab's implicit
  actuator computes them internally (stiffness + armature
  contribution). Do **not** copy raw `effort_limits` or `damping`
  from the actuator config and use them directly as PD gains.

### G5. Implicit (IsaacLab) vs explicit (MuJoCo) actuator

IsaacLab's PD runs implicitly with armature on the inertia matrix.
MuJoCo's `ctrl`-driven PD is explicit. With low arm/waist KP
(weak by design — many SONIC robots have ~14 N·m/rad on upper-body
joints) the difference is observable: explicit MuJoCo cannot hold
the upper body up against gravity by itself, and the policy is
expected to add the gravity-comp torque.

This is **not something you "fix"** in deployment — it's a known
sim gap. Practical implications:

- Do not expect the robot to hold its default pose under a zero
  action. IsaacLab can; MuJoCo cannot.
- Once the policy is trained well, sag disappears (the policy adds
  the needed torque). With a weakly-trained checkpoint you may see
  the upper body lean even though the legs walk.

### G6. Proprioception layout follows the dataclass, not the YAML

The flat proprioception vector is **not** in YAML order — it's in
`PolicyCfg` dataclass attribute order
(`gear_sonic/envs/manager_env/mdp/observations.py`):

```
[base_ang_vel, joint_pos_rel, joint_vel, last_action, gravity_dir]
```

Each term is concatenated across `history_length` frames,
**oldest-first** within the term. For an embodiment with `N` DOF and
`history_length=10` the layout is:

| Term | Per-frame | × history | Block size |
|---|---|---|---|
| `base_ang_vel` | 3 | 10 | 30 |
| `joint_pos_rel` | N | 10 | 10·N |
| `joint_vel` | N | 10 | 10·N |
| `last_action` | N | 10 | 10·N |
| `gravity_dir` | 3 | 10 | 30 |

Total: `60 + 30·N`. (For X2: 60 + 30·31 = **990**.)

A wrong term order produces nicely-numerically-ranged garbage. Use
`gear_sonic/scripts/dump_isaaclab_step0.py` to dump IsaacLab's
proprioception at step 0, then slice each block in your MuJoCo build
and verify the values match the corresponding `env_state` quantity
within the configured noise tolerance.

### G7. CircularBuffer reset broadcast-fills with the first observation

IsaacLab's `CircularBuffer` does **not** initialize history to zeros.
On reset it *broadcast-fills* every history slot with whatever the
first appended observation is. Replicate this:

```python
def append(self, ...):
    if not self._primed:
        for _ in range(HISTORY_LEN):
            self._hist.append(obs)   # for each term
        self._primed = True
    else:
        self._hist.append(obs)       # for each term
```

Wrong behavior (zero-initialized history): the policy makes huge
corrections in the first few control ticks because the `last_action`
history is artificially zero, even though the at-rest action would
have produced a finite value. The robot wobbles for the first
~200 ms of every episode then either settles or explodes.

### G8. FSQ quantization layer in the actor forward path

The universal-token actor's tokenizer encoder output is passed
through a Finite Scalar Quantizer (FSQ) — a `tanh`-then-round-to-bin
step. The checkpoint stores the FSQ levels but a naive forward
through encoder + decoder does **not** apply it automatically. You
must insert it explicitly:

```python
def fsq_quantize(z, levels):
    # z: (B, num_tokens * token_dim)
    bound = (levels - 1) * 0.5
    z_bounded = bound * torch.tanh(z)
    z_quant = torch.round(z_bounded) / bound
    return z_quant
```

Read the FSQ levels from the checkpoint config (look for
`quantizer.levels`). Without quantization the policy works
marginally — actions are continuous and slightly off, the robot
wobbles but doesn't catastrophically fail. With it, the policy
matches its trained behavior.

### G9. Tokenizer observation: future reference in the *current* body frame

The tokenizer encoder consumes a flattened sequence of the next
`num_future_frames` reference frames. Each frame's data
(`command_multi_future_nonflat`, `motion_anchor_ori_b_mf_nonflat`,
etc. — depends on the encoder config) is expressed in the **current**
body frame, not world frame. The reference frame is captured at the
*current* control tick, so each tokenizer obs is a fresh body-frame
projection of the same world-frame motion clip — repeating the
projection every tick is correct.

For X2 with 10 future frames at 0.1 s spacing the total dim is
`10 × 68 = 680`. Frame width depends on the tokenizer observation
group; see `gear_sonic/config/manager_env/observations/tokenizer/`.

### G10. RSI is the proper reset, not a clean spawn

IsaacLab does **not** start episodes with the robot standing in its
default pose at the origin. It teleports the entire robot — joints
+ root — to a sampled motion frame's state. This means:

- The first proprioception observation already has motion-correct
  joint velocities and base angular velocity.
- The "first action" the policy ever sees comes from a mid-stride
  state, not a stationary one.

If you spawn the robot at the default pose and let physics settle,
the first ~10 frames of proprioception will be wildly out-of-distribution
relative to training, and the policy will produce huge actions trying
to "fix" the imagined error. The deployment script uses RSI by default;
keep it that way unless you have a specific reason not to.

### G11. Foot collision rewrite — what we tried, and why we reverted

When standing-manipulation rollouts started visibly stumbling on the
heels/toes (the policy could not *recover* once one edge of a foot lost
contact), we suspected the MJCF foot contact model. The X2 Ultra MJCF
ships with **12 small spheres** per foot (`type="sphere" size="0.005"`)
and the global default `condim=3` — i.e. only sliding friction, no
torsional/rolling friction.

We tried the obvious "physically nicer" rewrite:

- Replace the 12 spheres with a single **box** collider per foot
  (`size="0.102 0.060 0.005"`), height-aligned to match the original
  spheres' lowest contact point.
- Switch the foot default to `condim=6` and
  `friction="1 0.05 0.005"` (sliding + torsional + rolling).

**Result: no improvement on average and clear regressions on several
motions.** Across the 15-motion top-standing benchmark, mean survival
moved by less than the per-motion noise floor at both step-2000 and
step-4000 checkpoints, while specific motions (`take_a_sip_270`,
`eat_hotdog_both_hands`) got *worse*. The change was reverted.

Best guess as to why: the policy was trained against IsaacSim PhysX
contact (rigid mesh + patch friction), not MuJoCo box-with-`condim=6`,
so swapping the contact model in deployment just trades one out-of-
distribution contact regime for another. Closing this gap likely needs
either (a) training-time domain randomization over foot contact
parameters, or (b) a mesh-based foot collider matched against the URDF
used in IsaacSim — both of which are out of scope for this guide and
tracked in the Open Work appendix below.

### G12. Joint armature must match IsaacLab per-joint

This was the one foot/contact investigation finding that survived. The
shipped MJCF used a single uniform `armature="0.03"` on **every** joint
plus `damping="0.0"`. IsaacLab's `ArticulationCfg` for X2 Ultra uses
**per-joint armature** values on the order of 1–2 of magnitude smaller,
grouped by motor torque class (see
`gear_sonic/envs/manager_env/robots/x2_ultra.py`):

| Class (joint regex) | Isaac armature | MJCF default class |
|---|---:|---|
| `.*_hip_pitch_joint`, `.*_hip_roll_joint`, `.*_hip_yaw_joint`, `.*_knee_joint` | 0.025101925 | `hipknee` |
| `.*_waist_yaw_joint` | 0.010177520 | `waistyaw` |
| `.*_wrist_pitch_joint`, `.*_wrist_roll_joint`, `.*_head_*_joint` | 0.00425 | `smallmotor` |
| ankle, waist_pr, shoulder, elbow, `wrist_yaw` (most joints) | 0.003609725 | (base default) |

Effect of the mismatch: armature shows up on the *diagonal of the joint-
space inertia matrix*, so an order-of-magnitude wrong value changes how
hard the same PD torque accelerates the joint. The policy was trained
expecting Isaac's values; running against MuJoCo's much-stiffer
effective inertia produces a robot that "feels" sluggish exactly where
it needs to be quickest (ankles, wrists), and over-responsive elsewhere.

The fix is encoded as nested `<default>` classes in the MJCF, plus
`class="…"` tags on the relevant `<joint>` elements:

```xml
<default class="x2">
  <joint damping="0.0" armature="0.003609725" frictionloss="0.3"/>
  <default class="hipknee">    <joint armature="0.025101925"/> </default>
  <default class="waistyaw">   <joint armature="0.010177520"/> </default>
  <default class="smallmotor"> <joint armature="0.00425"/>     </default>
</default>
…
<joint class="hipknee" name="left_hip_pitch_joint" .../>
<joint class="waistyaw" name="waist_yaw_joint" .../>
<joint class="smallmotor" name="head_pitch_joint" .../>
```

#### Why `damping="0.0"` is intentional, NOT a bug

Isaac's `ImplicitActuatorCfg` provides both `armature` *and* a
`stiffness`/`damping` pair derived from a natural-frequency / damping-
ratio model:

```python
NATURAL_FREQ  = 10 * 2 * pi          # 10 Hz
DAMPING_RATIO = 2.0
KP = armature * NATURAL_FREQ**2
KD = 2 * DAMPING_RATIO * armature * NATURAL_FREQ
```

In MuJoCo deployment, this stiffness/damping is **already** applied
**explicitly** by the Python PD loop in `eval_x2_mujoco.py` /
`benchmark_motions_mujoco.py` (`torque = KP·(target - q) - KD·qdot`).
If you *also* set MJCF `<joint damping="…">`, MuJoCo's solver will add
a *second* damping term to `qfrc_passive`, double-counting it and
crushing the policy's authority. **The MJCF must keep `damping="0.0"`**
for this deployment path to be correct. Only `armature` (which
contributes to the inertia matrix, not to passive forces) and
`frictionloss` belong in the MJCF.

#### Benchmark numbers

15-motion `top15_standing.pkl` benchmark, `--max-seconds 15`, headless,
fall = pelvis < 0.4 m or tilt > 72°. Mean survival seconds (higher is
better):

| Checkpoint | MJCF baseline (`armature=0.03`, no class) | MJCF + per-joint armature | Δ |
|---|---:|---:|---:|
| `model_step_002000.pt` | 3.81 s | 3.98 s | **+0.17 s** |
| `model_step_004000.pt` | 3.19 s | 3.25 s | +0.06 s |

Mean is mildly positive at both checkpoints; per-motion variance is
high (some motions improve by >2 s, others regress by ~1 s, with the
direction even flipping per checkpoint), confirming the residual
sim2sim gap is broader than this single MJCF axis. The patch is shipped
because it (a) is a correctness fix — MJCF should not lie about the
robot's rotor inertia — and (b) does not regress mean survival at any
checkpoint we tested.

To revert: `git checkout HEAD -- gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml`
restores the uniform-armature MJCF. No training-side change is needed
(armature lives in the deployment MJCF, not in the URDF that IsaacLab
loads for training).

### G16. Per-joint-group PD scaling — FIRST positive post-G11 result

After G11/G13/G14/G15 (foot geometry, foot compliance, joint frictionloss,
solver knobs + LPF) all reverted with no measurable gain, we noticed that
the working **G1 sim2sim deployment**
(`gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`)
ships *two* PD sets — `JOINT_KP/KD` (training-equivalent) and `MOTOR_KP/KD`
(deployed in MuJoCo, hand-tuned). Only the `MOTOR_*` values reach
`mj_data.ctrl` via `gear_sonic/utils/mujoco_sim/robot.py`. The asymmetry is
the standard fix for G5: IsaacLab integrates PD against joint-space inertia
+ armature **implicitly**, so the same numerical KP behaves stiffer than the
explicit `ctrl`-driven torque in MuJoCo. Bumping deployed PD recovers
authority without retraining.

The X2-specific `gear_sonic/scripts/eval_x2_mujoco.py` was using
`KP = armature × ω²` directly as the deployed PD — i.e. the analogue of
running G1 with its `JOINT_KP` instead of its `MOTOR_KP`. G16 added per-
joint-group `--kp-scale-{leg,knee,ankle,waist,arm,wrist,head}` and
`--kd-scale-*` flags to `benchmark_motions_mujoco.py` and swept seven
configurations on the same n=50 seed=0 standing subsample as G15.

Δmean survival vs the n=50 baseline:

| Config             | Δmean 4k | Δmean 6k | avg Δ   | 4k W/L/T | 6k W/L/T |
|--------------------|---------:|---------:|--------:|---------:|---------:|
| baseline           |    +0.00 |    +0.00 |  +0.00  |  —       |  —       |
| **`ankle_kp2`**    | **+0.85** | **+0.20** | **+0.52** | **38/10/2** | **29/20/1** |
| `global_kp15`      |    +0.49 |    +0.24 |  +0.37  | 35/11/4  | 28/21/1  |
| `global_kp20`      |    +0.15 |    +0.14 |  +0.15  | 29/20/1  | 30/19/1  |
| `g1_full`          |    +0.27 |    −0.12 |  +0.07  | 33/16/1  | 21/27/2  |
| `ankle_kp2_kd20`   |    +0.14 |    −0.03 |  +0.06  | 30/18/2  | 22/26/2  |
| `ankle_kd20`       |    +0.14 |    −0.10 |  +0.02  | 25/19/6  | 24/24/2  |
| `g1_legs`          |    −0.08 |    −0.21 |  −0.15  | 18/25/7  | 20/27/3  |

(W/L/T = per-motion delta thresholded at ±0.05 s. Baselines: 4k mean
2.276 s 0/50 survived; 6k mean 2.392 s 0/50 survived.)

Key takeaways:

- **Just doubling ankle KP is the headline lever.** +0.85 s @ 4k (38W/10L)
  and +0.20 s @ 6k. First single-axis post-G11 deployment knob to clear the
  per-motion noise band by a wide margin and win on both checkpoints. The
  largest per-motion movers are concentrated on idle / standing-still /
  body-check clips, exactly where foot-floor interaction dominates.
- **Bumping ankle KD destroys the gain.** `ankle_kp2_kd20` collapses from
  +0.85 → +0.14 at 4k. The G1 ankle KD ratio (×20) is **overdamped for
  X2** — our ankle armature is small enough that high deployed KD chokes
  the corrective response. Keep ankle KD at ×1.
- **Wholesale "copy G1 ratios" is wrong for X2.** `g1_legs` is the worst
  config in the sweep (avg −0.15 s). The waist down-scale, knee KD
  down-scale, and ankle KD up-scale that work for G1 do not transfer.
- **`global_kp15` is a real second-best** (+0.37 s avg), independently
  corroborating the implicit-vs-explicit gap; X2's deployed PD is uniformly
  weak. **`global_kp20` already overshoots** (+0.15 s avg), so the right
  uniform multiplier sits around 1.3–1.5.
- The X2 sim2sim foot-floor failure mode is **dominantly an ankle-authority
  shortfall**, not a contact-physics mismatch (G11/G13/G14/G15 all moved
  contact physics and lost) and not a policy bandwidth issue (G15 LPF lost).

The flags are kept in `benchmark_motions_mujoco.py` and the per-cell CSVs +
SUMMARY are in `/home/stickbot/sim2sim_armature_eval/g16_pd_scaling/`.

#### G16b. Fine-grain ankle-KP + KD + global combos on 6k

Follow-up on the 6k checkpoint to pin the optimum. 10 cells:

| Config              | Δmean 6k | W/L/T   | Notes |
|---------------------|---------:|---------|-------|
| **`ankle_kp1.5`**   | **+0.51** | **38/10/2** | best surgical knob at 6k |
| `global13_ankle1`   | +0.47    | 32/16/2 | global ×1.3 alone (1 motion fully survives) |
| `ankle_kp2_kd2`     | +0.46    | 33/16/1 | KP×2 with modest KD×2 |
| `global13_ankle2`   | +0.46    | 33/16/1 | ankle ×2 on top of global ×1.3 — **no extra gain** |
| `ankle_kp1.75`      | +0.40    | 36/11/3 |       |
| `ankle_kp2.5`       | +0.40    | 31/17/2 |       |
| `ankle_kp2_kd1.5`   | +0.34    | 35/12/3 |       |
| `ankle_kp3`         | +0.22    | 29/19/2 | non-monotone bounce → looks like ringing |
| `ankle_kp2`         | +0.20    | 29/20/1 | (G16 reference)  |
| `ankle_kp2.25`      | +0.14    | 32/18/0 |       |
| `ankle_kp2_kd20`    | −0.03    | 22/26/2 | (G16 reference; G1-style ×20 KD) |

The 6k picture inverts the 4k one. At 4k, `ankle_kp2` was the headline
(+0.85 s); at 6k it's only +0.20 s and the optimum has shifted **down** to
`ankle_kp1.5` (+0.51 s). The 6k policy has learned to compensate part of
the ankle authority gap on its own — so the right deployment-side bump
shrinks with training. Two takeaways for the recommended deployment PD:

- **Surgical recommendation (kept as default)**: `--kp-scale-ankle 1.5` for
  6k (or 2.0 for earlier checkpoints). Mechanistically clean, smallest
  delta from the training-equivalent PD. **This is now baked into
  `eval_x2_mujoco.py` via the `DEPLOYMENT_KP_SCALE` table** — calling the
  eval or benchmark scripts with no CLI flags applies ankle KP×1.5 by
  default, reproduces the +0.51 s 6k survival gain bit-exactly, and leaves
  every other joint group at its training-equivalent PD. Override per-run
  by passing `--kp-scale-*` / `--kd-scale-*` flags (the benchmark scales
  multiply on top of the table).
- **Broadband alternative**: `--kp-scale 1.3` produces an equivalent
  +0.47 s gain at 6k. Adding `--kp-scale-ankle 2.0` on top yields *no
  extra benefit* (+0.46 s vs +0.47 s). The implicit-vs-explicit gap (G5)
  and the ankle gap are **the same effect** at 6k — once you've stiffened
  globally by 30%, the ankle is no longer the bottleneck.

Two unrelated corrections to G16:

- **Modest ankle KD does help.** `ankle_kp2_kd2` and `ankle_kp2_kd1.5`
  both improve over `ankle_kp2` alone. The "KD ruins it" claim in G16
  applies specifically to the ×20 G1-style bump; the optimal ankle KD
  scale lives in [1.5, 2.0].
- **Single full survive at 6k** appears for the first time at
  `ankle_kp2.5` and `global13_ankle1`. We are still 1/50 on a 6 s cap,
  i.e. survival caps ~5–6 s on standing clips, but 70–75% per-motion
  improvements are now reproducible.

### G17. Walking-targeted waist/knee PD sweep — visual diagnosis didn't transfer

Background: with the G16b `ankle_kp1.5` baked into `eval_x2_mujoco.py` we ran
the two walking PKLs (`x2_ultra_walk_forward.pkl`,
`x2_ultra_relaxed_walk_postfix.pkl`) on the 6k checkpoint in the viewer. The
visual read was that the **waist looked weak** (torso couldn't hold yaw under
counter-rotation) and the **knee looked too stiff** (standing leg wouldn't
flex on impact). Both reads cleanly matched the X2-vs-G1 deployed PD ratio
table:

| group           | X2 deployed KP/KD       | G1 MOTOR_KP/KD       | G1/X2 KP × KD ratio |
|-----------------|------------------------:|---------------------:|--------------------:|
| hip P/R/Y       | 99.1 / 6.31             | 150 / 2              | ×1.5 / ×0.32        |
| knee            | 99.1 / 6.31             | 200 / 4              | ×2.0 / ×0.63        |
| ankle P/R       | 21.4 / 0.91 (×1.5 baked)| 40 / 2               | ×1.9 / ×2.2         |
| **waist yaw**   | **40.2 / 2.56**         | **250 / 5**          | **×6.2 / ×1.95**    |
| **waist P/R**   | **14.25 / 0.91**        | **250 / 5**          | **×17.5 / ×5.5**    |
| shoulder P/R    | 14.25 / 0.91            | 100 / 5              | ×7.0 / ×5.5         |

The waist gap is the largest in the whole table — X2's deployed waist roll/
pitch is **17.5× weaker** than G1's. So we ran a 6-cell sweep on both walks
to see if closing that gap (and softening the over-damped knee) improves
walking survival.

| Config                                    | walk_forward | relaxed_walk | mean   |
|-------------------------------------------|-------------:|-------------:|-------:|
| **A. baseline (ankle ×1.5 baked)**        |    2.78 s    |  **3.22 s**  | **3.00 s** |
| B. waist KP × 3                           |    1.68 s    |    2.60 s    | 2.14 s ↓ |
| C. waist KP × 5                           |    1.56 s    |    2.62 s    | 2.09 s ↓ |
| D. knee KD × 0.5                          |    2.98 s    |    1.98 s    | 2.48 s ↓ |
| E. knee KP × 1.5, knee KD × 0.5           |  **3.98 s**  |    2.02 s    | 3.00 s ⇄ |
| F. waist KP × 3, knee KD × 0.5            |    2.88 s    |    1.68 s    | 2.28 s ↓ |

(6k checkpoint; `--max-seconds 8`; ankle ×1.5 baked default applies to every
row including baseline.)

Key takeaways:

- **Bumping waist KP makes things *worse* on both walks.** Opposite of the
  visual diagnosis. The most likely cause: the policy was trained against
  IsaacLab's *implicit* waist gains and learned a specific waist trajectory.
  Stiffer deployed waist tracks more aggressively → small overshoot → COM
  oscillation the policy never saw in training. The "weak waist" visual is
  the **policy intentionally choosing soft waist coordination**, not a
  missing torque budget.
- **Knee KD reduction is gait-dependent.** It **helps** the more dynamic
  `walk_forward` (E hits 3.98 s, +43% over baseline) but **hurts**
  `relaxed_walk_postfix` (–1.24 s alone) where the standing leg needs the
  damping to lock the knee under load. A single global knee-KD scale can't
  win both.
- **No robust deployment-only winner over baseline.** Best mean across both
  walks is the baseline (tied by config E). Every other config regresses
  on the mean. We're hitting the floor of what deployment-side PD tweaks
  can buy on walking motions specifically.
- **The behavioral asymmetries identified visually are *trained-in*, not
  deployment-tuning gaps.** This is a structural finding: G16b/G17 together
  show that ankle authority transfers cleanly across the
  implicit→explicit boundary, but waist/knee dynamics are entangled with
  the policy's learned strategy and need per-joint domain randomization
  during training to fix, not deployment scaling.

Recommendations:

- **Keep the baked defaults** (ankle KP×1.5, everything else ×1.0). They
  win on standing (G16b) and tie on walking.
- **Do not** apply the G1 waist KP scaling (×6–17) to X2 — it actively
  hurts walking.
- When training cycles open back up, prioritize: (a) joint-level KD DR on
  knee (~±50%) and (b) modest joint-level KP DR on waist (~±20%). These
  are exactly the two places where the policy's learned coordination is
  brittle to small deployed-PD shifts.

Per-cell logs and CSV in `/home/stickbot/sim2sim_armature_eval/walk_pd_sweep/`.

### G15. Solver/contact-regime + action-LPF sweep — also negative

After G11/G13 (foot collider) and a G14 trial of `frictionloss=0` (also
reverted, see `/home/stickbot/sim2sim_armature_eval/icecream_diag/SUMMARY_g14.md`),
this experiment swept the Open Work list's remaining deployment-only knobs
on a wider 50-motion sample. Configs tested against `n=50` `seed=0`
random subsample of `x2_ultra_standing_only.pkl` (550 standing motions),
`max_seconds=6.0`:

- `mj_model.opt.impratio` ∈ {1 (default), 10, 100}
- `mj_model.opt.cone` ∈ {pyramidal (default), elliptic}
- Combinations of the above
- Action low-pass filter (EMA on policy joint target before PD),
  α ∈ {1.0 (none), 0.7, 0.5, 0.3} — implemented as
  `--action-lpf-alpha` in `benchmark_motions_mujoco.py`

Δmean survival vs the n=50 baseline (current MJCF, no overrides):

| Config             | Δmean 2k | Δmean 4k | Δmean 6k |
|--------------------|---------:|---------:|---------:|
| baseline           |    +0.00 |    +0.00 |    +0.00 |
| impratio=10        |    -0.35 |    +0.06 |    +0.01 |
| impratio=100       |    -0.42 |    -0.09 |    -0.21 |
| cone=elliptic      |    -0.08 |    -0.02 |    -0.08 |
| elliptic + imp10   |    -0.14 |    +0.06 |    +0.02 |
| elliptic + imp100  |    -0.04 |    +0.08 |    +0.01 |
| LPF α=0.7          |    -0.08 |    -0.08 |    +0.01 |
| LPF α=0.5          |    -0.21 |    -0.23 |    -0.38 |
| LPF α=0.3          |    -0.67 |    -0.29 |    -0.53 |

Key takeaways:

- **Action LPF strictly hurts.** Smoothing the policy's joint target
  reduces the corrective effort the policy already learned to apply.
  Lower α = larger regression at every checkpoint.
- **`impratio=100` over-corrects** and regresses 2k by −0.42 s; even
  at well-trained 6k it still costs −0.21 s.
- **The "best" combo (`elliptic + impratio=10`) is +0.02 s at 6k,**
  inside the per-seed noise floor (a different `--seed` would change
  the sign).

Conclusion: the deployment-only physics-knob surface is exhausted. No
single MJCF or script-level change moves mean survival outside the
per-motion noise band. The residual gap is structural (training
distribution); see Open Work #1/#2 for the remaining training-side
levers.

The CLI flags added for this sweep (`--impratio`, `--cone`,
`--solver-iters`, `--action-lpf-alpha`, `--seed`) are kept in
`benchmark_motions_mujoco.py` so future experiments can re-run any cell
of the matrix without script edits. Per-cell CSVs and the rerunnable
driver are in `/home/stickbot/sim2sim_armature_eval/post_training_sweep/`
(see `SUMMARY_post_training_knobs.md`).

### G13. Foot contact compliance tuning — also a negative result

Distinct from G11 (which changed *foot geometry* — spheres → box and
back), G13 tested whether keeping the original 12-sphere footprint but
softening the *contact regime* would help. Hypothesis: X2's real foot
appears rubberized/compliant compared to G1's harder/smaller foot, and
hard MuJoCo sphere contacts cause discrete heel-toe rocking on weight
shifts (visible as `take_a_sip` / `eat_hotdog` stumbles). Keeping the
geometry identical means the policy still sees the footprint it was
trained against in PhysX.

Change tested on `<default class="foot">`:

```xml
<default class="foot">
  <geom type="sphere" size="0.005"
        condim="4" friction="1.0 0.02 0.0001"
        solref="0.04 1" solimp="0.85 0.95 0.001"/>
</default>
```

Mean survival vs the G12 (armature-only) "patched" config:

| Step | Δmean   | improve / regress / tied |
|------|--------:|--------------------------|
| 2k   | −0.26 s | 3 / 11 / 1               |
| 4k   | +0.09 s | 10 / 4 / 1               |
| 6k   | −0.13 s | 5 / 7 / 3                |

Crucially, **`take_a_sip` got slightly worse at every checkpoint**
(−0.30 / −0.20 / −0.22 s) — the canonical motion the change was meant
to help — which directly weakens the rubber-pad hypothesis. Some
motions did improve substantially (`rage` +1.1/+1.4, `pick_up` +1.9
at 2k) but comparable-magnitude regressions on others canceled them
out. Reverted.

One side-effect kept: while testing, we discovered the original Agibot
MJCF declared `<default class="foot">` but its 24 foot spheres were
tagged `class="collision"` instead, so the foot default was orphaned.
Renaming the geoms to `class="foot"` is a one-line correctness fix
that's behaviorally a no-op under the (post-revert) sphere defaults
but makes the default block actually do what it says.

Full per-motion table and discussion:
`/home/stickbot/sim2sim_armature_eval/SUMMARY_compliant.md`.

## Debugging Recipe: When IsaacLab Works but MuJoCo Doesn't

1. **Dump the IsaacLab step-0 ground truth** with
   `gear_sonic/scripts/dump_isaaclab_step0.py`. It writes the full
   proprio vector, tokenizer obs, encoder/FSQ/decoder activations,
   and the final action to a `.pt` file.
2. In the MuJoCo loop, after RSI but before the first policy call,
   build the same observations and compare:
   - **Proprioception**: each block (per the G6 layout) should match
     the corresponding `env_state` quantity within IsaacLab's
     observation noise tolerance.
   - **Tokenizer obs**: should match exactly (no noise at step 0).
   - **Encoder output, FSQ output, decoder output, final action**:
     each should match within a few `1e-3` (FSQ rounding + float32
     differences).
3. Whichever block diverges first tells you exactly which side of the
   pipeline has the bug.

## Symptom → Likely Cause

| Symptom | First place to look |
|---|---|
| Robot collapses immediately on reset | Default pose + action-scale offset (G4); RSI not applied (G10) |
| Robot stands but tips over slowly | Implicit-vs-explicit actuator (G5); upper-body sag — sometimes expected |
| Robot walks one stride then falls | qvel angular frame (G1); proprio term order (G6); buffer not primed (G7) |
| Wild whole-body motion, looks "possessed" | DOF reorder direction swapped (G3); FSQ missing (G8) |
| Robot tries to stand on its head | Quaternion sign convention or `quat_rotate_inverse` direction (G2) |
| Arms float, legs over-respond | Single global action-scale instead of per-joint (G4) |
| Walks fine for ~1 s then drifts | Tokenizer obs body-frame transform off (G9); motion phase clock not aligned with RSI frame (G10) |
| Same checkpoint walks in IsaacLab, fails in MuJoCo | Always start with the dump-and-compare recipe above |
| Robot stumbles on heel/toe contact, can't recover | Ankle PD authority — try `--kp-scale-ankle 1.5` (G16b, +0.5 s avg survival on 6k) or `--kp-scale 1.3` (broadband, equivalent gain); foot contact model gap (G11 / G13 — both tried, both negative); confirm armature is set per joint (G12) |
| Joints feel sluggish at ankle/wrist or over-driven at hip | MJCF armature uniform / mismatched vs Isaac per-joint values (G12) |
| Walking falls in 2–3 s while standing/squat motions hold | **Not** a deployment-PD problem — G17 swept waist KP (×3, ×5) and knee KP/KD on walks and every config either tied or regressed against baseline. Walking gait is bottlenecked by training data / joint-dynamics DR coverage, not deployed gains. Visual "waist looks weak / knee looks stiff" reflects the policy's *learned* coordination, not a missing torque budget |

## Open Work — Sim2Sim Gaps Not Yet Closed

Documented for future contributors picking up this thread. None of the
items below are blocking deployment, but they are the most likely
remaining sources of the residual gap visible after applying G1–G12.

1. **Foot contact regime mismatch (see G11, G13).** PhysX rigid-mesh +
   patch-friction during training vs MuJoCo discrete-sphere `condim=3`
   at deployment. Two single-axis interventions tried and reverted:
   geometry rewrite (G11: spheres → box) and compliance-only tuning
   (G13: solref/solimp/condim/friction on the same spheres). Remaining
   paths, neither tried yet:
   - *Deployment side:* mesh-based foot collider in the MJCF matched
     against the IsaacSim URDF foot mesh (i.e. fix geometry *and*
     contact regime jointly, since varying them independently failed).
   - *Training side:* domain randomization over foot friction tuple,
     restitution, and (if feasible) a proxy for contact patch geometry.
2. **Joint-dynamics domain randomization.** The current training run
   randomizes mass, CoM, push, observation noise, but **not** joint
   armature/damping. Per-motion benchmark variance under G12 (some
   motions ±2 s) suggests the policy is brittle to small dynamics
   shifts. **G17 confirmed this empirically on walking motions** — every
   deployed PD scale tried (waist KP ×3/×5, knee KP×1.5, knee KD ×0.5,
   and combinations) either tied or regressed vs the ankle-only baked
   default, even though the X2-vs-G1 deployed PD ratios scream "waist
   under-actuated 17×". The asymmetries are *trained-in*, so the next
   cheap experiment is per-joint DR on knee KD (~±50%) and waist KP
   (~±20%) during training, not more deployment-side scaling.
3. **MuJoCo solver settings.** ✅ **Tested in G15 and ruled out** —
   `impratio` and `cone` and an action-LPF were swept on a 50-motion
   sample and produced regressions or noise-level deltas only. See G15
   above for the table.
4. **Action-rate / control-frequency mismatch.** ✅ **Verified equal:**
   `mj_model.opt.timestep × DECIMATION = 0.005 × 4 = 0.020 s` exactly
   matches IsaacLab's per-action `dt`. No mismatch.
5. **Contact-pair audit.** We have not enumerated which IsaacSim contact
   pairs are *enabled* during training (e.g., self-contact, floor-only,
   per-link `contype`/`conaffinity`-equivalents) and compared them to
   the MJCF `<contact>` block.

## See Also

- `gear_sonic/scripts/eval_x2_mujoco.py` — reference deployment script;
  contains the Python-side PD loop referenced by G12 and the
  `DEPLOYMENT_KP_SCALE` / `DEPLOYMENT_KD_SCALE` tables baked from G16b.
- `gear_sonic/scripts/benchmark_motions_mujoco.py` — headless multi-
  motion stability benchmark used to produce the G12 numbers.
- `gear_sonic/scripts/dump_isaaclab_step0.py` — IsaacLab GT dumper.
- `gear_sonic/envs/manager_env/robots/x2_ultra.py` — `ARMATURE_*`,
  `NATURAL_FREQ`, `DAMPING_RATIO`, and `ImplicitActuatorCfg` used to
  derive the MJCF armature defaults in G12.
- `gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml` — the
  MJCF being deployed; see the `<default class="x2">` block for the
  G12 armature classes.
- {doc}`new_embodiments` — adding a new robot from scratch
- {doc}`../references/conventions` — coordinate / quaternion / DOF conventions
- {doc}`../references/observation_config` — full observation-group reference
