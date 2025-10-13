#!/usr/bin/env python3
"""Convert G1 URDF to USD format for Isaac Sim"""

from isaaclab.app import AppLauncher

# Create the app launcher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

def convert_urdf_to_usd():
    """Convert G1 URDF to USD"""

    urdf_path = "/home/ykj/project/TienKung-Lab/legged_lab/assets/g1_29dof/urdf/g1_29dof.urdf"
    usd_path = "/home/ykj/project/TienKung-Lab/legged_lab/assets/g1_29dof/usd/g1_29dof.usd"

    # URDF conversion configuration
    urdf_converter_cfg = sim_utils.UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir="/home/ykj/project/TienKung-Lab/legged_lab/assets/g1_29dof/usd",
        usd_file_name="g1_29dof.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    )

    # Convert URDF to USD
    urdf_converter = sim_utils.UrdfConverter(urdf_converter_cfg)

    print(f"✅ Successfully converted URDF to USD: {usd_path}")

if __name__ == "__main__":
    convert_urdf_to_usd()
    simulation_app.close()
