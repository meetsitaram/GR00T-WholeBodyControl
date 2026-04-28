/**
 * @file tokenizer_obs.hpp
 * @brief 680-D tokenizer observation builder, per-frame interleaved layout.
 *
 * Mirrors what the trained PyTorch encoder MLP saw in IsaacLab (see
 * gear_sonic/scripts/dump_isaaclab_step0.py::encoder_input_for_mlp_view).
 * The bit-perfect re-export at gear_sonic/scripts/reexport_x2_g1_onnx.py
 * (FusedG1Wrapper.forward) does
 *
 *   obs[:, : enc_dim].view(-1, 10, 68)
 *
 * over the per-frame catenation ``cat([command(62), ori(6)], -1)``, so the
 * deploy must emit the same per-frame interleaved 680-float layout:
 *
 *   for k in 0..9:
 *     [jpos_fk_il(31) | jvel_fk_il(31) | rot6d_fk(6)]      # 68 floats
 *
 * Each future frame is sampled from the current ReferenceMotion at
 *   t_future_k = current_time + k * DT_FUTURE_REF          (k = 0..9)
 *
 * The 6-D rotation diff per frame is the first two COLUMNS of
 *   M_k = (cur_root_quat^-1 * future_root_quat).as_matrix()
 * flattened row-major (matches IsaacLab's
 * ``mat[..., :2].reshape(B, -1)`` -- see math_utils::rot6d_from_quat_xyzw).
 *
 * Bug history: an older revision of the deploy emitted the "grouped" form
 *   [ jpos_il_f0(31) | jvel_il_f0(31) | ... | jpos_il_f9 | jvel_il_f9     <-620
 *   | rot6d_f0(6) | ... | rot6d_f9(6) ]                                   <-60
 * because the *original* (broken) ONNX shipped at
 * ``gear_sonic_deploy/models/x2_sonic_16k.onnx`` was exported via
 * ``inference_helpers.py::UniversalTokenWrapper`` which expected that
 * layout. After the bit-perfect re-export the new ONNX expects the
 * per-frame interleaved layout shown above; the mismatch produced
 * ~1-3 rad raw policy outputs on the first CONTROL tick even on a
 * stand-still / idle motion, and was caught by the slot diff in
 * gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py.
 */

#ifndef AGI_X2_TOKENIZER_OBS_HPP
#define AGI_X2_TOKENIZER_OBS_HPP

#include "policy_parameters.hpp"
#include "reference_motion.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace agi_x2 {

constexpr std::size_t TOK_DIM           = 680;
constexpr std::size_t TOK_CMD_FLAT_DIM  = NUM_FUTURE_FRAMES * (NUM_DOFS + NUM_DOFS);  // 620
constexpr std::size_t TOK_ORI_FLAT_DIM  = NUM_FUTURE_FRAMES * 6;                       // 60
static_assert(TOK_CMD_FLAT_DIM + TOK_ORI_FLAT_DIM == TOK_DIM,
              "tokenizer obs widths do not sum to 680");

/**
 * Build the 680-D tokenizer observation in PER-FRAME INTERLEAVED layout.
 *
 * @param ref          reference-motion source (StandStill or PklMotion)
 * @param current_time seconds since CONTROL state entry (matches Sample(t)
 *                     contract used elsewhere in the deploy)
 * @param base_quat_wxyz current robot torso orientation, MuJoCo / IMU convention
 * @return float vector of length TOK_DIM, ready to feed the ONNX session
 *         alongside the 990-D proprioception.
 *
 * The output layout is:
 *
 *   for k in 0..9: [jpos_fk_il(31) | jvel_fk_il(31) | rot6d_fk(6)]    # 68 floats
 *
 * for a total of 10 * 68 = 680 floats. This matches both the trained
 * PyTorch encoder input (encoder_input_for_mlp_view) and the bit-perfect
 * ONNX re-export (FusedG1Wrapper.forward in
 * gear_sonic/scripts/reexport_x2_g1_onnx.py).
 *
 * Feeding the OLD "grouped" ([all_cmd(620)|all_ori(60)]) layout to the new
 * ONNX produces ~1-3 rad outputs on the very first tick because the model
 * reads the 60 ori floats from what used to be the last 60 cmd floats.
 */
std::vector<float> BuildTokenizerObs(const ReferenceMotion& ref,
                                     double current_time,
                                     const std::array<double, 4>& base_quat_wxyz);

}  // namespace agi_x2

#endif  // AGI_X2_TOKENIZER_OBS_HPP
