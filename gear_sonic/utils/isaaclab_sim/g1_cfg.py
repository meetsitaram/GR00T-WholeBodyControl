"""
G1 29-DOF ArticulationCfg for SONIC WBC in Isaac Lab.

Actuators use stiffness=0, damping=0 (pure torque mode) so that our sim loop
can apply torques from the deploy binary's LowCmd (kp, kd, q, dq, tau)
identically to how the MuJoCo sim does:
    torque = tau_ff + kp * (q_target - q_actual) + kd * (dq_target - dq_actual)

Joint order (matches SONIC's MuJoCo model exactly):
    0-5:   left  leg  (hip_pitch/roll/yaw, knee, ankle_pitch/roll)
    6-11:  right leg
    12-14: waist (yaw, roll, pitch)
    15-21: left  arm  (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
    22-28: right arm
    29-42: hand fingers (14 joints, not used by SONIC)
"""

import math
import os

import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GEAR_SONIC_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
G1_29DOF_USD = os.path.join(_GEAR_SONIC_ROOT, "gear_sonic", "data", "robot_model", "model_data", "g1", "g1_29dof_usd", "g1_29dof.usd")

# SONIC 29-DOF joint names in MuJoCo order
SONIC_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

SONIC_DEFAULT_ANGLES = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# ---------------------------------------------------------------------------
# PD gains from policy_parameters.hpp (armature-based computation)
# torque = tau_ff + kp*(q_target - q_actual) + kd*(dq_target - dq_actual)
# ---------------------------------------------------------------------------
_NATURAL_FREQ = 10 * 2.0 * math.pi
_DAMPING_RATIO = 2

_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425

_S = {
    "5020": _ARMATURE_5020 * _NATURAL_FREQ ** 2,
    "7520_14": _ARMATURE_7520_14 * _NATURAL_FREQ ** 2,
    "7520_22": _ARMATURE_7520_22 * _NATURAL_FREQ ** 2,
    "4010": _ARMATURE_4010 * _NATURAL_FREQ ** 2,
}
_D = {
    "5020": 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ,
    "7520_14": 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ,
    "7520_22": 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ,
    "4010": 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ,
}

SONIC_DEFAULT_KP = np.array([
    _S["7520_22"],       # left_hip_pitch
    _S["7520_22"],       # left_hip_roll
    _S["7520_14"],       # left_hip_yaw
    _S["7520_22"],       # left_knee
    2.0 * _S["5020"],    # left_ankle_pitch
    2.0 * _S["5020"],    # left_ankle_roll
    _S["7520_22"],       # right_hip_pitch
    _S["7520_22"],       # right_hip_roll
    _S["7520_14"],       # right_hip_yaw
    _S["7520_22"],       # right_knee
    2.0 * _S["5020"],    # right_ankle_pitch
    2.0 * _S["5020"],    # right_ankle_roll
    _S["7520_14"],       # waist_yaw
    2.0 * _S["5020"],    # waist_roll
    2.0 * _S["5020"],    # waist_pitch
    _S["5020"],          # left_shoulder_pitch
    _S["5020"],          # left_shoulder_roll
    _S["5020"],          # left_shoulder_yaw
    _S["5020"],          # left_elbow
    _S["5020"],          # left_wrist_roll
    _S["4010"],          # left_wrist_pitch
    _S["4010"],          # left_wrist_yaw
    _S["5020"],          # right_shoulder_pitch
    _S["5020"],          # right_shoulder_roll
    _S["5020"],          # right_shoulder_yaw
    _S["5020"],          # right_elbow
    _S["5020"],          # right_wrist_roll
    _S["4010"],          # right_wrist_pitch
    _S["4010"],          # right_wrist_yaw
], dtype=np.float64)

SONIC_DEFAULT_KD = np.array([
    _D["7520_22"],       # left_hip_pitch
    _D["7520_22"],       # left_hip_roll
    _D["7520_14"],       # left_hip_yaw
    _D["7520_22"],       # left_knee
    2.0 * _D["5020"],    # left_ankle_pitch
    2.0 * _D["5020"],    # left_ankle_roll
    _D["7520_22"],       # right_hip_pitch
    _D["7520_22"],       # right_hip_roll
    _D["7520_14"],       # right_hip_yaw
    _D["7520_22"],       # right_knee
    2.0 * _D["5020"],    # right_ankle_pitch
    2.0 * _D["5020"],    # right_ankle_roll
    _D["7520_14"],       # waist_yaw
    2.0 * _D["5020"],    # waist_roll
    2.0 * _D["5020"],    # waist_pitch
    _D["5020"],          # left_shoulder_pitch
    _D["5020"],          # left_shoulder_roll
    _D["5020"],          # left_shoulder_yaw
    _D["5020"],          # left_elbow
    _D["5020"],          # left_wrist_roll
    _D["4010"],          # left_wrist_pitch
    _D["4010"],          # left_wrist_yaw
    _D["5020"],          # right_shoulder_pitch
    _D["5020"],          # right_shoulder_roll
    _D["5020"],          # right_shoulder_yaw
    _D["5020"],          # right_elbow
    _D["5020"],          # right_wrist_roll
    _D["4010"],          # right_wrist_pitch
    _D["4010"],          # right_wrist_yaw
], dtype=np.float64)

SONIC_DEFAULT_ANGLES_ARRAY = np.array([
    SONIC_DEFAULT_ANGLES[j] for j in SONIC_JOINT_ORDER
], dtype=np.float64)

# Per-joint torque limits from MuJoCo XML actuatorfrcrange (SONIC order).
# MuJoCo clamps actuator forces to these ranges automatically; we must do the
# same explicitly since Isaac Lab's effort_limit_sim may not apply to
# set_joint_effort_target() with stiffness=0/damping=0.
_TORQUE_LIMIT_MAP = {
    "hip_pitch": 88.0,
    "hip_roll": 88.0,
    "hip_yaw": 88.0,
    "knee": 139.0,
    "ankle_pitch": 50.0,
    "ankle_roll": 50.0,
    "waist_yaw": 88.0,
    "waist_roll": 50.0,
    "waist_pitch": 50.0,
    "shoulder_pitch": 25.0,
    "shoulder_roll": 25.0,
    "shoulder_yaw": 25.0,
    "elbow": 25.0,
    "wrist_roll": 25.0,
    "wrist_pitch": 5.0,
    "wrist_yaw": 5.0,
}

def _joint_to_limit(jname: str) -> float:
    for suffix, limit in _TORQUE_LIMIT_MAP.items():
        if jname.endswith(suffix + "_joint"):
            return limit
    return 50.0  # safe fallback

SONIC_TORQUE_LIMITS = np.array([
    _joint_to_limit(j) for j in SONIC_JOINT_ORDER
], dtype=np.float64)

NUM_BODY_MOTORS = 29
SPAWN_HEIGHT = 0.793
FALL_HEIGHT = 0.2
SIM_DT = 0.005

G1_SONIC_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_29DOF_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.793),
        joint_pos=SONIC_DEFAULT_ANGLES,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
            ],
            effort_limit_sim=88.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit_sim=139.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=50.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "waist_rp": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=25.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=5.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_.*_joint"],
            effort_limit_sim=5.0,
            stiffness=5.0,
            damping=0.5,
            armature=0.001,
        ),
    },
)
"""G1 29-DOF ArticulationCfg with torque-mode actuators for SONIC deploy."""
