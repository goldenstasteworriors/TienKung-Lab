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

import argparse

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the txt file")
parser.add_argument("--fps", type=float, default=30.0, help="Target fps")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play_amp_animation():
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.commands.debug_vis = False

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    frame_cnt = 0
    all_frames = []
    while simulation_app.is_running():
        time = (frame_cnt % (env.motion_len)) * (1.0/args_cli.fps)
        frame = env.visualize_motion(time)
        if args_cli.save_path:
            frame = frame.cpu().numpy().reshape(-1)
            all_frames.append(frame)
            # 如果保存模式,只播放一次
            if frame_cnt >= (env.motion_len - 1):
                break
        frame_cnt += 1

    if args_cli.save_path:
        all_frames_np = np.stack(all_frames, axis=0)
        np.savetxt(args_cli.save_path, all_frames_np, fmt='%f', delimiter=', ')

        with open(args_cli.save_path, 'r') as f:
            frames_data = f.readlines()

        frames_data_len = len(frames_data)
        with open(args_cli.save_path, 'w') as f:
            f.write('{\n')
            f.write('"LoopMode": "Wrap",\n')
            f.write(f'"FrameDuration": {1.0 / args_cli.fps:.3f},\n')
            f.write('"EnableCycleOffsetPosition": true,\n')
            f.write('"EnableCycleOffsetRotation": true,\n')
            f.write('"MotionWeight": 0.5,\n\n')
            f.write('"Frames":\n[\n')

            for i, line in enumerate(frames_data):
                line_start_str = '  ['
                if i == frames_data_len - 1:
                    f.write(line_start_str + line.rstrip() + ']\n')
                else:
                    f.write(line_start_str + line.rstrip() + '],\n')

            f.write(']\n}')

        print(f"✅ Successfully converted to {args_cli.save_path}")
if __name__ == "__main__":
    play_amp_animation()
    simulation_app.close()
