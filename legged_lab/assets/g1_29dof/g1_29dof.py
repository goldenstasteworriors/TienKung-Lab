# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Configuration for Unitree G1 29DOF humanoid robot.

The following configurations are available:

* :obj:`G1_29DOF_CFG`: Unitree G1 humanoid robot with full 29 degrees of freedom

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/g1_29dof/usd/g1_29dof.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # Left leg
            "left_hip_pitch_joint": -0.35,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.7,
            "left_ankle_pitch_joint": -0.35,
            "left_ankle_roll_joint": 0.0,
            # Right leg
            "right_hip_pitch_joint": -0.35,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.7,
            "right_ankle_pitch_joint": -0.35,
            "right_ankle_roll_joint": 0.0,
            # Waist
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # Left arm
            "left_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.5,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            # Right arm
            "right_shoulder_pitch_joint": 0.3,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.5,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_hip": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_yaw_joint": 150.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 15.0,
                ".*_hip_roll_joint": 15.0,
                ".*_hip_yaw_joint": 15.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_yaw_joint": 100.0,
            },
            damping={
                ".*_hip_pitch_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_yaw_joint": 3.0,
            },
        ),
        "legs_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit_sim=300.0,
            velocity_limit_sim=15.0,
            stiffness=150.0,
            damping=5.0,
        ),
        "legs_ankle": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 12.0,
                ".*_ankle_roll_joint": 12.0,
            },
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 1.0,
                ".*_ankle_roll_joint": 1.0,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 100.0,
                "waist_roll_joint": 100.0,
                "waist_pitch_joint": 100.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 10.0,
                "waist_roll_joint": 10.0,
                "waist_pitch_joint": 10.0,
            },
            stiffness={
                "waist_yaw_joint": 50.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            damping={
                "waist_yaw_joint": 2.0,
                "waist_roll_joint": 2.0,
                "waist_pitch_joint": 2.0,
            },
        ),
        "arms_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 45.0,
                ".*_shoulder_roll_joint": 45.0,
                ".*_shoulder_yaw_joint": 45.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 12.0,
                ".*_shoulder_roll_joint": 12.0,
                ".*_shoulder_yaw_joint": 12.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
            },
        ),
        "arms_elbow": ImplicitActuatorCfg(
            joint_names_expr=[".*_elbow_joint"],
            effort_limit_sim=45.0,
            velocity_limit_sim=12.0,
            stiffness=40.0,
            damping=2.0,
        ),
        "arms_wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_wrist_roll_joint": 20.0,
                ".*_wrist_pitch_joint": 20.0,
                ".*_wrist_yaw_joint": 20.0,
            },
            velocity_limit_sim={
                ".*_wrist_roll_joint": 10.0,
                ".*_wrist_pitch_joint": 10.0,
                ".*_wrist_yaw_joint": 10.0,
            },
            stiffness={
                ".*_wrist_roll_joint": 15.0,
                ".*_wrist_pitch_joint": 15.0,
                ".*_wrist_yaw_joint": 15.0,
            },
            damping={
                ".*_wrist_roll_joint": 1.0,
                ".*_wrist_pitch_joint": 1.0,
                ".*_wrist_yaw_joint": 1.0,
            },
        ),
    },
)
