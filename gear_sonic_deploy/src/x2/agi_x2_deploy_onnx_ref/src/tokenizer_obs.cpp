#include "tokenizer_obs.hpp"

#include "math_utils.hpp"

namespace agi_x2 {

std::vector<float> BuildTokenizerObs(const ReferenceMotion& ref,
                                     double current_time,
                                     const std::array<double, 4>& base_quat_wxyz)
{
  std::vector<float> out;
  out.reserve(TOK_DIM);

  // Pre-compute current orientation conjugate (xyzw) used for the relative
  // rotation cur^-1 * future across all 10 future frames.
  const auto cur_quat_xyzw      = wxyz_to_xyzw(base_quat_wxyz);
  const auto cur_quat_inv_xyzw  = quat_conj_xyzw(cur_quat_xyzw);

  // -------------------------------------------------------------------------
  // ONNX 680-D tokenizer layout = PER-FRAME INTERLEAVED:
  //
  //   [cmd_f0(62) | ori_f0(6) | cmd_f1(62) | ori_f1(6) | ... | cmd_f9 | ori_f9]
  //
  // where for each frame k:
  //   cmd_fk[0..30]  = jpos_fk  (IL order, 31)
  //   cmd_fk[31..61] = jvel_fk  (IL order, 31)
  //   ori_fk[0..5]   = rot6d(cur^-1 * future_k)  -- IL convention: first 2
  //                                                 COLUMNS of the rotmat
  //                                                 (see math_utils::rot6d_*)
  //
  // and the FUTURE TIME WINDOW for frame k is
  //
  //   t_k = current_time + k * DT_FUTURE_REF                  for k = 0..9
  //
  // i.e. frame 0 IS THE CURRENT MOTION FRAME (t + 0.0), not t + 0.1.
  // This matches IsaacLab `TrackingCommand.future_time_steps_init = arange(N) *
  // frame_skips` (commands.py:354-361), verified against a captured step-0
  // dump from IsaacSim where command_multi_future_nonflat[frame 0] == the
  // robot's current joint_pos to <1e-6.
  //
  // -------------------------------------------------------------------------
  // Bug history #3 (THIS revision): the previous layout was the "grouped"
  // form [all_cmd(620) | all_ori(60)]. That was correct for the *old*
  // (broken) ONNX shipped at gear_sonic_deploy/models/x2_sonic_16k.onnx,
  // because that ONNX was exported via inference_helpers.py::
  // UniversalTokenWrapper which sliced `obs[:, :620]` as cmd and
  // `obs[:, 620:680]` as ori. After the bit-perfect re-export by
  // gear_sonic/scripts/reexport_x2_g1_onnx.py (FusedG1Wrapper.forward), the
  // new ONNX expects the layout actually fed to the trained MLP in
  // IsaacLab, which is `view(-1, 10, 68)` over the per-frame-interleaved
  // catenation `cat([command(62), ori(6)], -1)`. Verified: the IL dump's
  // `encoder_input_for_mlp_view` flattens (10, 68) row-major over per-frame
  // interleaved frames.
  //
  // Symptom of the mismatch on real hardware: every joint saw raw policy
  // outputs of order 1-3 rad on the very first CONTROL tick, even on the
  // gantry with the robot near-default and a stand-still / idle reference
  // motion -- because the new ONNX was reading the 60 ori floats from what
  // used to be the last 60 cmd floats (= jpos/jvel of frame 9), then the 620
  // cmd floats from a 620-byte chunk that overlapped the actual ori block.
  // gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py made this
  // visible: deploy ori[:6] looked suspiciously like another copy of jpos.
  //
  // Bug history #2: an earlier port from `eval_x2_mujoco.build_tokenizer_obs`
  // used `(f + 1) * DT_FUTURE_REF` (== [t+0.1, ..., t+1.0]). That window is
  // shifted by exactly one frame from training, so the policy was being
  // asked to track a 100ms-ahead reference instead of the current one and
  // produced large catch-up actions on every step.
  //
  // Bug history #1: a previous "fix" reshuffled this to
  // [jpos_f0..f9 | jvel_f0..f9 | ori_f0..f9] thinking that was the "grouped"
  // layout. That misread `cmd_part.reshape(-1)` (C-order flatten still
  // produces per-frame interleave for jpos/jvel), and it produced uniform
  // 30/31 joint clamp saturation on the real robot.
  //
  // -------------------------------------------------------------------------
  // The reference returns joint pos/vel in MuJoCo order; we IL-remap here
  // (matches eval_x2_mujoco.py's `m["dof"][fi][IL_TO_MJ_DOF]` -- using the
  // IsaacLab->MuJoCo permutation INDEXED BY IL gives an IL-ordered output).
  // -------------------------------------------------------------------------
  for (std::size_t k = 0; k < NUM_FUTURE_FRAMES; ++k) {
    const double t_k   = current_time + static_cast<double>(k) * DT_FUTURE_REF;
    const auto   frame = ref.Sample(t_k);

    for (std::size_t il = 0; il < NUM_DOFS; ++il) {
      const std::size_t mj = static_cast<std::size_t>(isaaclab_to_mujoco[il]);
      out.push_back(static_cast<float>(frame.joint_pos_mj[mj]));
    }
    for (std::size_t il = 0; il < NUM_DOFS; ++il) {
      const std::size_t mj = static_cast<std::size_t>(isaaclab_to_mujoco[il]);
      out.push_back(static_cast<float>(frame.joint_vel_mj[mj]));
    }

    const auto rel_xyzw = quat_mul_xyzw(cur_quat_inv_xyzw, frame.root_quat_xyzw);
    const auto rot6     = rot6d_from_quat_xyzw(rel_xyzw);
    for (double v : rot6) {
      out.push_back(static_cast<float>(v));
    }
  }

  return out;
}

}  // namespace agi_x2
