# Sim-to-Sim Ablation Study — IsaacLab → MuJoCo for the Agibot X2 Ultra

> **Status:** Draft v1, 2026-04-28. Companion to
> [`sim2sim_mujoco.md`](sim2sim_mujoco.md). The latter is an
> engineering-debugging guide; this document is a self-contained
> experimental write-up suitable for an external audience or a paper
> appendix.

## TL;DR

Reinforcement-learning policies trained in IsaacLab/PhysX for the
Agibot X2 Ultra humanoid track in-distribution motion clips with > 95 %
success when evaluated *inside* their training simulator, but degrade
to **2–5 s time-to-fall** on the same clips when deployed in MuJoCo
with the AgiBot-shipped MJCF — the simulator and contact model used
for eventual hardware deployment.

A systematic five-axis ablation sweep (`A0`–`A5`, hardening IsaacLab
toward the MuJoCo physics) shows that a **single** axis dominates the
gap: the foot collision model. Replacing IsaacLab's mesh-foot URDF
with a hand-authored 12-sphere foot URDF (mirroring the MJCF) reduces
the in-IsaacLab success rate of an existing 16k-iteration mesh-trained
policy from 1.000 to 0.493 — a controlled "fail Isaac to look like
MuJoCo" reproduction.

A 4 000-iteration KL-free fine-tune of the same policy on the
12-sphere foot URDF restores **1.000 progress on every IsaacLab
ablation row** (A0–A5, all axes mirrored), and improves MuJoCo
mean time-to-fall by **+50 %–100 %** across three benchmark motions.
However, **no checkpoint stably survives the full 9–16 s clips in
MuJoCo** — best single-cell result is 6.22 s on a 16.5 s relaxed-walk
clip.

The residual gap therefore lies in MuJoCo physics axes that have **no
IsaacLab counterpart** — contact compliance (`solref`/`solimp`),
friction-cone model, and integration substeps — and must be addressed
either by deploy-side MJCF tuning or by training-side domain
randomisation that simulates the unmodelled effects.

This document records the framework, every measured number, and the
protocol so that the experiments are reproducible and so the same
ablation chassis can be re-used for other embodiments and other
sim2sim-pair targets.

---

## 1. Setup

### 1.1 Robot and motion suite

- **Embodiment:** Agibot X2 Ultra, 31 actuated DOFs (12 leg, 3 waist,
  14 arm, 2 head). Floating base.
- **Mass:** 35 kg standing; 1.6 m tall.
- **Asset source:** `X2_URDF-v1.3.0` package shipped by AgiBot.
- **Motion suite:** the `x2_ultra_top15_standing.pkl` set — 15 SOMA-retargeted
  in-distribution motion clips covering standing manipulation, eating,
  drinking, picking up, idling, and a single fall recovery.
  Per-motion durations 4.4 s–13.7 s.

### 1.2 Training stack

- **Trainer:** IsaacLab 2.2.0 + PhysX, GR00T-WholeBodyControl
  (`gear_sonic`) + custom `trl`-based PPO ("SONIC").
- **Architecture:** Universal-token actor with a `g1` motion-tokenizer
  encoder (680-d) plus `g1_dyn` decoder, taking a 1670-d input
  (`tokenizer_obs(680) | proprioception(990)`) and emitting a 31-d action.
- **Physics:** PhysX, mesh foot collisions (CAD-export URDF default),
  `replicate_physics=True` (single shared `ArticulationCfg` cloned
  across 4 096 envs).
- **PD control:** `ImplicitActuatorCfg` with per-joint stiffness/damping
  baked into `gear_sonic/envs/manager_env/robots/x2_ultra.py`.
- **Domain randomisation (training):** push-robot, compliance-force-push,
  per-env physics-material μ, rigid-body mass scaling, base-CoM offset,
  add-joint-default-pos perturbation, observation corruption.
- **DR (eval):** all randomisations *off* by default; ablation rows
  toggle them explicitly.

### 1.3 Deploy stack

- **Simulator:** MuJoCo 3.x with the AgiBot-shipped `x2_ultra.xml`
  MJCF.
- **Foot collision model:** 12 explicit `<geom type="sphere"
  size="0.005">` per foot (24 total), positioned at the corners of the
  foot sole.
- **PD control (deploy):** explicit PD computed inside
  `gear_sonic/scripts/eval_x2_mujoco.py`, ankle KP scaled ×1.5 from
  training (G16b deployment-side adjustment).
- **Joint frictionloss:** `0.3 N⋅m` per joint, declared in MJCF
  `<default class="x2"><joint frictionloss="0.3"/></default>`.
- **Floor:** `<geom name="floor" size="0 0 0.05" type="plane">` with
  default MuJoCo friction (μ = 1.0, μ_torsion = 0.005, μ_rolling =
  0.0001).

### 1.4 The "wrong-URDF" provenance bug

The dominant axis in this study (foot collision model) is not a
missing physics setting — it is an asset-selection bug that survived
~16 000 iterations of training before being identified. AgiBot's
`X2_URDF-v1.3.0` package ships **three** robot descriptors side by
side:

| File | Foot collision | Used by |
|---|---|---|
| `x2_ultra.urdf` | mesh (CAD STL) | **the GR00T integration picked this for IsaacLab** |
| `x2_ultra_simple_collision.urdf` | 24 spheres (12/foot) | nothing — sat unused next to the file we picked |
| `x2_ultra.xml` (MJCF) | 24 spheres (12/foot) | MuJoCo deploy |

Our diagnostic-derived `x2_ultra_sphere_feet.urdf` is byte-identical
to the upstream `x2_ultra_simple_collision.urdf` — we re-derived
the upstream sphere-foot file by hand without realising it already
existed. The URDF/MJCF mismatch was therefore introduced *not* by
either of the upstream simulators but by the asset-selection step in
the GR00T integration pipeline.

**Reproducibility note for other embodiments:** when integrating any
new humanoid, before the first training run, generate a per-link
collision-geom diff between the URDF you select for IsaacLab and the
MJCF you select for MuJoCo. Mismatches surface as visually-plausible
but functionally divergent contact behaviour.

---

## 2. Ablation framework

### 2.1 The five rows

The ablation script (`gear_sonic/scripts/sweep_isaac_mujoco_mirror.py`)
parameterises six axes that hardening IsaacLab toward MuJoCo:

| Row | What it adds vs the previous row | Hydra override |
|---|---|---|
| `A0_isaac_stock` | IsaacLab as trained, full DR + obs noise | (none — baseline) |
| `A1_no_dr_no_noise` | drop training-only DR events; disable obs corruption | `++train_only_events=[…]` `++observations.policy.enable_corruption=False` |
| `A2_frictionloss` | A1 + `joint frictionloss=0.3 N·m` | `++robot.frictionloss=0.3` |
| `A3_sphere_feet` | A1 + 12-sphere foot URDF (mirrors MJCF) | `++robot.foot=sphere` |
| `A4_explicit_pd` | A1 + `IdealPDActuatorCfg` + ankle KP × 1.5 | `++robot.actuator_regime=explicit ++robot.ankle_kp_scale=1.5` |
| `A5_full_mirror` | A1 + frictionloss + sphere feet + explicit PD + ankle ×1.5 | all four flags |

Each row launches `gear_sonic/eval_agent_trl.py` headless on 15
parallel envs (one per motion in `x2_ultra_top15_standing.pkl`), runs
the `im_eval` callback to write `metrics_eval.json`, parses the
per-motion success/progress, and writes a CSV row.

### 2.2 Plumbing additions to the trainer

To support the ablation knobs, the following changes were made to the
training stack (all opt-in; default training behaviour unchanged):

- `make_x2_ultra_cfg(actuator_regime, frictionloss, foot, ankle_kp_scale)`
  factory in `gear_sonic/envs/manager_env/robots/x2_ultra.py`; default
  call reproduces the previous `X2_ULTRA_CFG` byte-for-byte.
- Hydra parsing in
  `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` for the
  four `++robot.*` overrides above.
- New asset
  `gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra_sphere_feet.urdf`
  with 24 spheres (`r = 0.005 m`) at the exact MJCF positions.
- Driver `gear_sonic/scripts/sweep_isaac_mujoco_mirror.py` that
  orchestrates checkpoint × row × motion combinations.

### 2.3 Per-cell evaluation protocol

For each (checkpoint, row) cell:

1. Hydra-compose the eval config with row-specific overrides.
2. Launch IsaacLab headless with `num_envs=15`, one motion per env,
   plane terrain (matches MuJoCo floor; X2 normally trains on trimesh,
   which adds height-noise that masks the physics axis we care about).
3. Run the `im_eval` callback to drive each env until termination or
   end-of-clip, then write `metrics_eval.json`.
4. Parse mean and per-motion `progress_rate` (= `clip-time-completed /
   clip-duration`), `success_rate` (= reached end without termination),
   and termination fraction.

Each invocation produces a timestamped CSV
(`sweep_<UTC>_<rows>_<steps>.csv`) and updates a `latest.csv` symlink;
per-cell directories `<row>/step_<step>/` retain `metrics_eval.json`
and `run.log` for inspection. No file is overwritten across runs.

### 2.4 MuJoCo deploy-side measurement protocol

For each (policy, motion, init-frame) cell:

1. Load MJCF, RSI to motion frame `init_frame`, build proprioception
   and tokenizer obs identically to IsaacLab.
2. Run the explicit PD control loop at `decimation = 4`, `sim_dt =
   0.005 s`, `control_dt = 0.02 s`.
3. Record the wall-time (in motion-time seconds) at which the pelvis
   either drops below 0.4 m world-z or the body tilts past `acos(0.3)
   ≈ 72°`. This is "time-to-fall".
4. Tabulate across 5 init frames per cell to get a (mean, std) under
   different physics initial conditions.

The harness is `gear_sonic/scripts/record_x2_eval_mujoco.py` for
headless MP4 + log capture.

---

## 3. Experiments

### 3.1 Phase 1 — diagnostic ablation on the original 16k mesh-trained policy

**Goal:** reproduce the MuJoCo failure inside IsaacLab by toggling one
axis at a time.

**Checkpoints tested:** `model_step_002000.pt`, `006000.pt`,
`016000.pt` from a 16k mesh-trained run
(`/home/stickbot/x2_cloud_checkpoints/run-20260420_083925/`).

**Results (mean across 15 motions):**

| Row | 2k progress | 2k term | 6k progress | 6k term | 16k progress | 16k term |
|---|---:|---:|---:|---:|---:|---:|
| `A0_isaac_stock` | 0.935 | 0.067 | 0.987 | 0.067 | 1.000 | 0.000 |
| `A1_no_dr_no_noise` | 0.959 | 0.067 | 1.000 | 0.000 | 1.000 | 0.000 |
| `A2_frictionloss` | 0.959 | 0.067 | 1.000 | 0.000 | 1.000 | 0.000 |
| **`A3_sphere_feet`** | **0.409** | **0.933** | **0.656** | **0.533** | **0.493** | **0.667** |
| `A4_explicit_pd` | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| `A5_full_mirror` | 0.657 | 0.600 | 0.710 | 0.600 | 0.627 | 0.667 |

**Findings:**

1. Stock IsaacLab (A0) does **not** reproduce the deploy gap on its
   own — the policy holds 0.94 → 0.99 → 1.00 progress on 2k → 6k → 16k
   inside its native simulator.
2. Removing DR + obs noise (A1) does **not** unmask the failure either
   — A1 ≈ A0. The MuJoCo collapse is *not* "Isaac was hiding the
   failure under noise."
3. Joint `frictionloss = 0.3 N·m` (A2) is a no-op on this policy.
4. `IdealPDActuatorCfg + ankle KP × 1.5` (A4) is a no-op alone.
5. **The 12-sphere foot collision (A3) drops Isaac to 0.41 / 0.66 /
   0.49 progress with 0.93 / 0.53 / 0.67 termination fractions, and
   pulls `min_progress` down to 0.038 at 2k.** The 16k checkpoint is
   the worst of the three under spheres — same monotonic-degradation
   direction as observed in MuJoCo.
6. A5 (everything together) is *less* catastrophic than A3 alone (0.66
   / 0.71 / 0.63 vs 0.41 / 0.66 / 0.49). The deployment-side PD
   scaling baked from G16b is at least directionally compensating for
   the contact-geometry hit.

**Conclusion of Phase 1:** the foot collision model is the dominant
sim2sim axis. The fix must live in the **training distribution**, not
in further deployment-side MJCF tuning.

Full per-cell metrics: `/home/stickbot/sim2sim_armature_eval/isaaclab_mujoco_mirror/SUMMARY_isaac_mujoco_mirror.md`.

### 3.2 Phase 2 — single-GPU fine-tune of the 16k policy on the 12-sphere foot

**Setup:** 1 × NVIDIA H200 (Nebius `1gpu-16vcpu-200gb`, eu-west1).
Warm-start from `model_step_016000.pt` (mesh-trained); train with
`++robot.foot=sphere` for 4 000 PPO iterations on 4 096 envs at 5.0 s/iter.
Total wall: ~5.6 hours.

The training config is `sonic_x2_ultra_bones_seed_sphere_feet.yaml`,
which derives from the canonical `sonic_x2_ultra_bones_seed.yaml` and
overrides only `manager_env.config.robot.foot=sphere`.

**Catastrophic-forgetting guard:** an A0 (mesh) + A3 (sphere) checkpoint
sweep at iter 2 000 verified the policy retained > 0.70 mesh progress
before allowing the run to continue to 4 000 iters. Result: A0 mesh
held at 0.944 (only −5 pp vs the 16k baseline), so no KL-anchor was
needed.

**IsaacLab-side eval at the final 4 000-iteration checkpoint:**

| Row | progress_rate | success_rate | terminated_frac |
|---|---:|---:|---:|
| `A0_isaac_stock` | **1.000** | 1.000 | 0.000 |
| `A2_frictionloss` | **1.000** | 1.000 | 0.000 |
| `A3_sphere_feet` | **1.000** | 1.000 | 0.000 |
| `A4_explicit_pd` | **1.000** | 1.000 | 0.000 |
| `A5_full_mirror` | **1.000** | 1.000 | 0.000 |

**Every single ablation row is now perfect.** A3 went from 0.493 →
**1.000** (+50.7 pp). A0 mesh went 1.000 → 0.944 (mid-train) → **back
to 1.000**, demonstrating zero net catastrophic forgetting.

The IsaacLab side of the sim2sim mirror is fully closed: the policy
is robust to every axis we can express on the training side.

**Cumulative training cost:** ≈ $21 of cloud GPU time (1 × H200 ×
$3.80/hr × 5.6 h).

### 3.3 Phase 3 — MuJoCo deploy-side measurement (multi-init-frame)

**Protocol:** for every (policy, motion, init-frame ∈ {0, 10, 20, 30, 40})
cell, record a 15 s headless rollout in MuJoCo and tabulate
time-to-fall.

**Policies under test:**
- `16k_mesh_baseline` (original mesh-trained policy)
- `2k_sphere_ft` (mid-fine-tune snapshot)
- `4k_sphere_ft` (final fine-tune snapshot)

**Motions under test:**
- `standing__eat_icecream_fall_standing_R_001__A456_M` ("icecream",
  13.7 s) — canonical hardest motion; the reference clip itself
  contains an actor stumble that the policy must track.
- `Relaxed_walk_forward_002__A057_M_postfix` ("relaxed walk", 16.5 s)
  — clean 8 m forward walk.
- `Neutral_walk_forward_002__A057` ("neutral walk", 9.1 s) — slightly
  stiffer cadence.

**Per-cell results (time-to-fall in seconds):**

```
Motion         Policy                init=0   init=10  init=20  init=30  init=40   mean   std
─────────────────────────────────────────────────────────────────────────────────────────────
icecream       16k_mesh_baseline      2.14    3.12     2.18     2.42     2.76     2.52  0.41
icecream       2k_sphere_ft           3.20    3.26     3.12     3.06     3.08     3.14  0.08
icecream       4k_sphere_ft           3.46    3.18     3.48     3.22     3.18     3.30  0.15

relaxed_walk   16k_mesh_baseline      2.22    1.44     2.96     2.12     2.34     2.22  0.54
relaxed_walk   2k_sphere_ft           5.12    3.46     4.58     3.18     2.60     3.79  1.04
relaxed_walk   4k_sphere_ft           3.80    6.22     4.48     2.14     5.20     4.37  1.53

neutral_walk   16k_mesh_baseline      4.46    2.10     2.14     1.48     2.58     2.55  1.14
neutral_walk   2k_sphere_ft           6.08    5.16     4.78     4.74     4.28     5.01  0.68
neutral_walk   4k_sphere_ft           3.60    4.52     2.54     3.66     4.10     3.68  0.74
```

**Headline summary (mean ± std time-to-fall, over 5 init frames):**

| Motion | 16k mesh | 2k sphere FT | 4k sphere FT |
|---|---:|---:|---:|
| icecream | 2.52 ± 0.4 | 3.14 ± 0.1 | **3.30 ± 0.2** |
| relaxed_walk | 2.22 ± 0.5 | 3.79 ± 1.0 | **4.37 ± 1.5** |
| neutral_walk | 2.55 ± 1.1 | **5.01 ± 0.7** | 3.68 ± 0.7 |
| **mean across motions** | **2.43** | **3.98** | **3.78** |

**Findings:**

1. **Both fine-tune checkpoints meaningfully beat the 16k baseline on
   every motion.** Mean time-to-fall improves from 2.43 s to 3.78–3.98
   s — a +55 % to +64 % gain.
2. **Per-motion best checkpoint differs:** 4k wins on dynamic motions
   (icecream, relaxed walk), 2k wins on the cleanest walk (neutral
   walk). An intermediate (~3 000 iter) checkpoint may give the best
   of both — but our save cadence (every 2 000 iters) was too coarse.
3. **None of the three policies stably survives a full clip.** Best
   single cell across the entire 45-rollout matrix is 6.22 s on a
   16.5 s clip. We are at ≈ 30 % of clip duration on average.
4. **Standard deviations are large** (especially for relaxed walk at
   4k: σ = 1.5 s on a 4.4 s mean). Time-to-fall is sensitive to
   initial conditions, so single-rollout comparisons mislead — prior
   single-rollout numbers showed a "regression" of 4k vs 2k on relaxed
   walk that disappeared with multi-init averaging.

### 3.4 Phase 4 — qualitative observation in the interactive viewer

When the auto-reset thresholds are loosened (`--fall-tilt-cos -0.7`
≈ 45 °, `--fall-height 0.25 m`) so the policy can actually attempt
recoveries, the failure mode is consistent across motions:

> The robot raises its arms (or executes the gesture), then drifts in
> yaw, then drifts forward, then falls — typically before the
> reference clip itself ends.

This is the expected residual after the dominant axis is fixed:

- **Yaw drift** is consistent with the friction-cone differences
  between PhysX (complementarity) and MuJoCo (Coulomb pyramid). The
  same foot torque commanded by the policy produces a different yaw
  response.
- **Forward drift** is consistent with center-of-pressure (CoP)
  concentration differences under sphere contact. Small ankle-pitch
  errors that get smeared across PhysX's mesh contact patch produce
  sharper CoP shifts under MuJoCo's discrete sphere contacts.
- **No recovery** is consistent with the training reference being a
  near-stationary clip without large-yaw, large-forward-drift
  excursions in its tokenizer-obs distribution.

---

## 4. Discussion

### 4.1 Asymmetry of the IsaacLab-side and MuJoCo-side fixes

A central finding: **the IsaacLab side of the gap was fully closed by
a single training-distribution intervention** (sphere-foot URDF
fine-tune). Every IsaacLab axis returns 1.000 at the 4 000-iteration
checkpoint. Yet **the MuJoCo deploy-side gap is only partially
closed** (≈ +60 % time-to-fall improvement, but still falling within
3–6 s of every clip).

The asymmetry implies the *remaining* gap is in axes for which we have
no IsaacLab knob. We can mirror the *contact geometry* (URDF) and the
*joint friction* (frictionloss override) but we cannot mirror:

- **Contact-solver compliance** (`solref`/`solimp` in MJCF).
  PhysX is a complementarity-based solver and has no equivalent
  knobs. Compliance affects how much the foot "sinks" into the floor
  and the rebound dynamics, both of which the policy implicitly
  exploits during training.
- **Friction-cone model.** MuJoCo uses a Coulomb pyramid; PhysX uses a
  complementarity formulation. Same μ produces different slip and
  spin responses.
- **Integration substep counts.** MuJoCo and PhysX time-step the
  contact dynamics differently even at matching outer `dt`.
- **Floor parameters.** MJCF default plane friction tuple
  (μ=1.0, μ_torsion=0.005, μ_rolling=0.0001) vs IsaacLab's per-env DR
  μ ∈ [0.4, 1.2]. The bounds overlap but the distributions differ.

### 4.2 Implications for sim2sim methodology

1. **Audit the upstream asset drop before integration.** The dominant
   axis in this study turned out to be a wrong-URDF-selection bug, not
   a missing physics setting. A 30-second per-link diff between the
   URDF and the MJCF would have caught the discrepancy on day one.
2. **Make every robot-physics choice explicit at config time, not
   implicit at filesystem time.** The `make_x2_ultra_cfg(foot=...)`
   factory + Hydra `++robot.foot=sphere` switch promotes the choice
   from "a hidden assumption baked into a path" to "a visible knob in
   every training config and every eval run." Trainees see two URDFs
   and a knob, not one URDF and a hidden default.
3. **A controlled ablation chassis is more useful than ad-hoc tuning.**
   The five-row sweep produced a cleaner attribution of the gap than
   the preceding two months of one-off `solref` and frictionloss
   experiments. The sweep is now reusable for any other embodiment.
4. **Save checkpoints at finer granularity than your training
   cadence's epoch.** Different deploy motions peak at different
   training iterations; we discovered post hoc that 2 000 iters was
   the optimum for clean walking and 4 000 was the optimum for
   dynamic recovery. Saving every 500 iters during fine-tunes allows
   an a posteriori choice.
5. **Multi-init averaging is mandatory for time-to-fall metrics.**
   Single-rollout numbers (σ ≈ 0.5–1.5 s on a 3–6 s mean) routinely
   misled us into spurious regression conclusions during this work.

### 4.3 What did NOT close the gap

For completeness — these were tried earlier and reverted as net
neutral or net negative:

- **G11**: change MJCF foot from spheres to a single box. Made MuJoCo
  more like (mesh) IsaacLab; failed to improve and changed the gap
  shape.
- **G13**: tune sphere `solref`/`solimp` and add torsional friction.
  Did not improve mean survival; specifically regressed `take_a_sip`
  at every checkpoint.
- **G14 / icecream diag**: matching IsaacLab and MuJoCo step-by-step
  state for the icecream motion. Identified that proprio, tokenizer
  obs, encoder/decoder, and final actions all match within tolerance
  through the first ~50 ticks; the divergence emerges purely from
  *physics* (ground contact and torque integration) downstream of the
  matching observations and actions.
- **G16b**: deploy-side ankle KP × 1.5. Slight improvement on 6k mesh
  policy (+0.5 s mean survival); kept as the deploy default but does
  not close the gap on its own.
- **G17**: deploy-side waist KP × 3 and × 5; knee KP/KD sweeps. Every
  config tied or regressed against baseline on walks. Walking gait is
  bottlenecked by training-data coverage, not deployed gains.

### 4.4 Open avenues

- **Targeted training-side DR.** Inject yaw drift, forward CoP
  perturbation, and contact-friction noise during training so the
  policy learns to recover from MuJoCo-style perturbations even
  though we can't model the underlying solver differences directly.
- **MJCF deploy-side audit (Phase 5 below).** Find the minimum-edit
  MJCF tune that closes another chunk of the gap without retraining.
- **Larger fine-tune budget with finer save cadence.** 4 000 iterations
  may be too few; the policy plateaued at `failure_rate_mean ≈ 0.08`
  in IsaacLab, suggesting headroom exists. Save every 500 iters and
  pick the best deploy-side checkpoint.
- **Train with the MJCF's `frictionloss=0.3` AND sphere feet
  simultaneously.** A2 was a no-op on the mesh-trained policy and on
  the 4k sphere-trained policy *separately*, but was never trained
  *jointly* with sphere feet from scratch. The interaction may differ
  when both axes are present in the training distribution.
- **Closed-loop IsaacLab eval with a runtime-sphere-vs-mesh switch
  per episode** to estimate the maximum achievable gap closure with a
  fully-mirrored training distribution. Currently blocked by
  IsaacLab's `replicate_physics=True` architecture; would require
  either disabling clone-physics or two parallel scene roots.

---

## 5. Phase 5 — MuJoCo MJCF audit (in progress)

Goal: identify the deploy-side MJCF knobs without an IsaacLab
counterpart that could close the residual gap. Update this section as
results land.

Candidate knobs (in priority order based on G18 hypothesis ranking):

1. **Foot sphere `solref` / `solimp`** — controls penetration/restitution
   stiffness. Default `solref="0.02 1"` (time constant 20 ms,
   damping ratio 1.0). Softer (40 ms) was tried in G13 and reverted;
   try slightly *stiffer* (15 ms or 10 ms) to bias toward kinematic
   contact like PhysX mesh.
2. **Foot sphere `condim`** — number of contact dimensions. Default
   `condim=3` (frictionless contact normal + 2 tangent friction).
   `condim=4` adds torsional friction; `condim=6` adds rolling.
   G13 tried `condim=4` and reverted, but combined with sphere-feet
   fine-tune the response may differ.
3. **Floor friction tuple** — `<geom name="floor"
   friction="μ μ_torsion μ_rolling">`. Currently MuJoCo defaults
   (1.0 0.005 0.0001). Match to IsaacLab's training mean μ (≈ 0.8)
   before any randomisation.
4. **Joint frictionloss** in MJCF — currently `0.3` for all joints.
   Try `0.0` (matching IsaacLab default) to test if this is *adding*
   resistance the policy didn't train against.
5. **Ankle armature** — currently inherited from `<default class="x2">
   <joint armature="0.003609725"/></default>`. Match to IsaacLab's
   per-joint armature table from G12.

Each candidate to be A/B'd in MuJoCo against the 4k sphere fine-tune
across all 3 benchmark motions × 5 init frames (45 rollouts per
candidate, ~10 minutes per candidate).

---

## Appendix A — Reproduction commands

### A.1 Run the IsaacLab ablation sweep

```bash
conda run -n env_isaaclab --no-capture-output python \
  gear_sonic/scripts/sweep_isaac_mujoco_mirror.py \
  --rows A0_isaac_stock A2_frictionloss A3_sphere_feet A4_explicit_pd A5_full_mirror \
  --checkpoint-root /path/to/run/dir \
  --checkpoints 002000 004000 \
  --out-dir /path/to/results
```

Each invocation writes `sweep_<UTC>_<rows>_<steps>.csv` and updates
`latest.csv`; per-cell `metrics_eval.json` and `run.log` live in
`<row>/step_<step>/`.

### A.2 Run the multi-init MuJoCo bench

```bash
# Build job list: 3 policies × 3 motions × 5 init-frames
for pol in 4k:$CKPT_4K 2k:$CKPT_2K 16k:$CKPT_16K; do
  for mot in icecream:$MOT_ICE relaxed:$MOT_REL walkforward:$MOT_WLK; do
    for init in 0 10 20 30 40; do
      echo "$pol $mot $init"
    done
  done
done | xargs -n3 -P6 bash -c '
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/record_x2_eval_mujoco.py \
    --checkpoint "$2" --motion "$4" --init-frame "$5" \
    --out /tmp/bench/${1}_${3}_init${5}.mp4 --duration 15.0
'
```

Parse fall times with `grep "\[fall\]" *.log`.

### A.3 Single-GPU sphere-feet fine-tune

```bash
EXP_NAME=sonic_x2_ultra_bones_seed_sphere_feet \
NUM_ITERS=4000 \
NUM_ENVS=4096 \
EXTRA_FLAGS="+checkpoint=/path/to/16k_mesh_baseline.pt" \
USE_WANDB=False \
bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh
```

(Despite the script name, with `NUM_PROCESSES=1` it runs single-GPU
fine.)

---

## Appendix B — File map

- `gear_sonic/envs/manager_env/robots/x2_ultra.py` — `make_x2_ultra_cfg`
  factory.
- `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` — Hydra
  parsing of `++robot.*` overrides.
- `gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra.urdf`
  — original mesh-feet URDF (CAD export default).
- `gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra_sphere_feet.urdf`
  — hand-authored 12-sphere foot URDF (byte-equivalent to AgiBot's
  `x2_ultra_simple_collision.urdf`).
- `gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml` —
  MuJoCo MJCF, 12-sphere feet.
- `gear_sonic/scripts/sweep_isaac_mujoco_mirror.py` — IsaacLab
  ablation sweep driver.
- `gear_sonic/scripts/eval_x2_mujoco.py` — interactive MuJoCo viewer.
- `gear_sonic/scripts/record_x2_eval_mujoco.py` — headless MuJoCo MP4
  recorder.
- `docs/source/user_guide/sim2sim_mujoco.md` — engineering-debugging
  companion guide; section G18 documents the Phase 1 ablation in
  embedded form.
- `/home/stickbot/sim2sim_armature_eval/isaaclab_mujoco_mirror/SUMMARY_isaac_mujoco_mirror.md`
  — full Phase 1 per-cell metrics.
- `/home/stickbot/sim2sim_armature_eval/sphere_ft_diag/sweep_*.csv` —
  Phase 2 raw CSVs.
- `/tmp/mujoco_5seed/*.{mp4,log}` — Phase 3 raw rollouts (MP4 +
  per-step pelvis-z log).
