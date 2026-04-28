#!/usr/bin/env python3
"""Slot-by-slot diff: IsaacLab step-0 obs vs MuJoCo eval rebuild.

The user's checkpoint trains and evaluates cleanly in IsaacSim but fails on
the deploy stack (which inherits MuJoCo's obs construction). This script is
the only ground-truth check that doesn't require the real robot:

  1. Read /tmp/x2_step0_isaaclab.pt (produced by dump_isaaclab_step0.py).
     That dump contains:
       - actor_obs[0] (1670)            -- the exact tensor IsaacLab fed
       - tokenizer_obs dict              -- per-feature parsed slices
       - proprioception_input            -- 990 from IsaacLab's term order
       - decoder_action_mean             -- the action IsaacLab produced
       - env_state.{joint_pos, joint_vel, root_quat_w_wxyz, root_ang_vel_b,
                    motion_ids, motion_times, default_joint_pos, ...}

  2. Use env_state to drive the MuJoCo `build_tokenizer_obs` /
     `ProprioceptionBuffer` exactly as eval_x2_mujoco.py does, producing
     a rebuilt 1670-D actor_obs.

  3. Print per-block max-abs diff and first few mismatching indices. Run
     the ONNX session on BOTH vectors and print both action norms.

Run:

    python gear_sonic_deploy/scripts/compare_isaaclab_vs_mujoco_obs.py \
        --dump /tmp/x2_step0_isaaclab.pt \
        --model gear_sonic_deploy/models/x2_sonic_16k.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.scripts.eval_x2_mujoco import (  # noqa: E402
    DEFAULT_DOF,
    HISTORY_LEN,
    IL_TO_MJ_DOF,
    MJ_TO_IL_DOF,
    NUM_DOFS,
    NUM_FUTURE_FRAMES,
    DT_FUTURE_REF,
    ProprioceptionBuffer,
    build_tokenizer_obs,
    quat_rotate_inverse,
)
from gear_sonic.scripts.eval_x2_mujoco_onnx import (  # noqa: E402
    OnnxActor,
    PROP_DIM,
    TOK_DIM,
)


def _block_diff(name: str, a: np.ndarray, b: np.ndarray, *, head: int = 8):
    diff = np.abs(a - b)
    inf = float(diff.max()) if diff.size else 0.0
    l2 = float(np.linalg.norm(diff))
    n_bad = int((diff > 1e-4).sum())
    print(f"  {name:<28s}  size={diff.size:5d}  inf={inf:.4e}  L2={l2:.4e}  "
          f"  >1e-4: {n_bad}/{diff.size}")
    if inf > 1e-4 and head > 0:
        bad_idx = np.argsort(diff)[::-1][:head]
        for i in bad_idx:
            print(f"      idx {int(i):4d}  IL={a[i]:+.5f}  MJ={b[i]:+.5f}  "
                  f"diff={diff[i]:+.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, type=Path,
                    help="path to /tmp/x2_step0_isaaclab.pt")
    ap.add_argument("--model", required=True, type=Path,
                    help="path to the same ONNX file IsaacLab eval used")
    ap.add_argument("--motion", type=Path, default=None,
                    help="motion .pkl matching env_state.motion_ids[0]; "
                         "if omitted, we attempt to find one in "
                         "gear_sonic/data/motions/")
    ap.add_argument("--env-idx", type=int, default=0,
                    help="which env in the dump batch to use (default 0)")
    args = ap.parse_args()

    dump = torch.load(args.dump, map_location="cpu", weights_only=False)
    actor_obs_il = dump["actor_obs"][args.env_idx, 0].cpu().numpy().astype(np.float32)
    print(f"IsaacLab actor_obs:  shape={actor_obs_il.shape}  "
          f"min={actor_obs_il.min():+.3f} max={actor_obs_il.max():+.3f}")
    print(f"IsaacLab decoder_action_mean[{args.env_idx}]: "
          f"{dump['decoder_action_mean'][args.env_idx].cpu().numpy()}")

    env_state = dump.get("env_state")
    if env_state is None:
        sys.exit("dump has no env_state; rerun dump_isaaclab_step0.py with "
                 "the env state patch enabled (it should be by default).")

    joint_pos_il = env_state["joint_pos"][args.env_idx].cpu().numpy().astype(np.float64)
    joint_vel_il = env_state["joint_vel"][args.env_idx].cpu().numpy().astype(np.float64)
    root_quat_wxyz = env_state["root_quat_w_wxyz"][args.env_idx].cpu().numpy().astype(np.float64)
    root_ang_vel_b = env_state["root_ang_vel_b"][args.env_idx].cpu().numpy().astype(np.float64)

    motion_id = int(env_state["motion_ids"][args.env_idx].item())
    motion_t = float(env_state["motion_times"][args.env_idx].item())
    print(f"\nIL env_state: motion_id={motion_id}  motion_t={motion_t:.4f}s  "
          f"root_quat(wxyz)={root_quat_wxyz}")

    motion_path = args.motion
    if motion_path is None:
        # eval_agent_trl loads motions from a directory; we don't replicate
        # the resolution, just nudge the user.
        sys.exit("--motion not provided. Re-run with the same motion .pkl "
                 "the IsaacLab eval was using (look for "
                 "`motion_command.motion_path` in dump's hydra config or "
                 "the eval_agent_trl logs).")

    import joblib
    motion_data = joblib.load(motion_path)
    first = motion_data[list(motion_data.keys())[0]]
    motion_fps = float(first.get("fps", 30.0))

    # Rebuild tokenizer obs using MuJoCo eval pipeline.
    tok_mj = build_tokenizer_obs(motion_data, motion_t, root_quat_wxyz, motion_fps)

    # Rebuild proprioception. IsaacLab's CircularBuffer is broadcast-primed
    # on the first sample, so for step 0 the 10-frame history is just the
    # current obs replicated.
    grav_b = quat_rotate_inverse(root_quat_wxyz, np.array([0.0, 0.0, -1.0]))
    jpos_rel_il = joint_pos_il - DEFAULT_DOF[IL_TO_MJ_DOF]
    last_action_il = np.zeros(NUM_DOFS, dtype=np.float32)
    pb = ProprioceptionBuffer()
    pb.append(grav_b, root_ang_vel_b, jpos_rel_il, joint_vel_il, last_action_il)
    prop_mj = pb.get_flat()

    # Convert the MuJoCo tokenizer (PT-interleaved) to ONNX-grouped to
    # match what IsaacLab feeds the fused graph. eval_x2_mujoco_onnx does
    # the same conversion inside OnnxActor.__call__.
    from gear_sonic.scripts.eval_x2_mujoco_onnx import _interleaved_to_grouped
    tok_mj_onnx = _interleaved_to_grouped(tok_mj.astype(np.float32))
    actor_obs_mj = np.concatenate([tok_mj_onnx, prop_mj.astype(np.float32)])

    print("\n=== Slot-by-slot diff (IsaacLab vs MuJoCo-eval rebuild) ===")
    # Tokenizer: 620 cmd + 60 ori
    _block_diff("tok cmd_flat", actor_obs_il[:620], actor_obs_mj[:620])
    _block_diff("tok ori_flat", actor_obs_il[620:680], actor_obs_mj[620:680])
    # Proprioception term-by-term (matches PolicyCfg attribute order):
    #   base_ang_vel(30), joint_pos_rel(310), joint_vel(310),
    #   last_action(310), gravity_dir(30)
    p0, p1 = 680, 680 + 30
    _block_diff("prop base_ang_vel(30)", actor_obs_il[p0:p1], actor_obs_mj[p0:p1])
    p0, p1 = p1, p1 + 310
    _block_diff("prop joint_pos_rel(310)", actor_obs_il[p0:p1], actor_obs_mj[p0:p1])
    p0, p1 = p1, p1 + 310
    _block_diff("prop joint_vel(310)", actor_obs_il[p0:p1], actor_obs_mj[p0:p1])
    p0, p1 = p1, p1 + 310
    _block_diff("prop last_action(310)", actor_obs_il[p0:p1], actor_obs_mj[p0:p1])
    p0, p1 = p1, p1 + 30
    _block_diff("prop gravity_dir(30)", actor_obs_il[p0:p1], actor_obs_mj[p0:p1])

    # Run the ONNX on BOTH vectors and print the action norms.
    actor = OnnxActor(str(args.model))
    out_il = actor.session.run(
        [actor.output_name], {actor.input_name: actor_obs_il.reshape(1, -1)}
    )[0][0]
    out_mj = actor.session.run(
        [actor.output_name], {actor.input_name: actor_obs_mj.reshape(1, -1)}
    )[0][0]
    print("\n=== ONNX action on each vector ===")
    print(f"  on IsaacLab obs:  inf={np.max(np.abs(out_il)):.4f}  L2={np.linalg.norm(out_il):.4f}")
    print(f"  on MuJoCo  obs:  inf={np.max(np.abs(out_mj)):.4f}  L2={np.linalg.norm(out_mj):.4f}")
    print(f"  diff(IL action vs MJ action): inf={np.max(np.abs(out_il-out_mj)):.4e}")
    print(f"  vs dumped IL decoder_action_mean: "
          f"inf={np.max(np.abs(out_il - dump['decoder_action_mean'][args.env_idx].cpu().numpy())):.4e}")


if __name__ == "__main__":
    main()
