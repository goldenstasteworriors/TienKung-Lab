# TienKung-Lab: Direct IsaacLab Workflow for TienKung

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RK](https://img.shields.io/badge/RSL_RL-2.3.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

<div align="center">
  <img src="docs/Tienkung_marathon.jpg" width="600">
  <br>
  <b><span style="font-size:1.2em;">TienKung humanoid robot won the championship in the first Humanoid Robot Half Marathon</span></b>
  <br>
</div>



## Overview

| Motion |              AMP Animation               |                     Sensors                      |                   RL + AMP                    |                   Sim2Sim                   |
| :----: | :--------------------------------------: | :----------------------------------------------: | :-------------------------------------------: | :-----------------------------------------: |
|  Walk  | <img src="docs/walk_amp.gif" width="200"> | <img src="docs/walk_with_sensor.gif" width="200"> | <img src="docs/walk_isaaclab.gif" width="200"> | <img src="docs/walk_mujoco.gif" width="200"> |
|  Run   | <img src="docs/run_amp.gif" width="200">  | <img src="docs/run_with_sensor.gif" width="200">  | <img src="docs/run_isaaclab.gif" width="200">  | <img src="docs/run_mujoco.gif" width="200">  |

This framework is an RL-based locomotion control system designed for full-sized humanoid robots, TienKung. It integrates AMP-style rewards with periodic gait rewards, facilitating natural, stable, and efficient walking and running behaviors.

The codebase is built on IsaacLab, supports Sim2Sim transfer to MuJoCo, and features a modular architecture for seamless customization and extension. Additionally, it incorporates ray-casting-based sensors for enhanced perception, enabling precise environmental interaction and obstacle avoidance.

## TODO List
- [ ] Add motion dataset for TienKung
- [ ] Motion retargeting support
- [ ] Add more sensors
- [ ] Add Perceptive Control

## Installation
TienKung-Lab is built with IsaacSim 4.5.0 and IsaacLab 2.1.0.

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory)

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
cd TienKung-Lab
pip install -e .
```
- Install the rsl-rl library

```bash
cd TienKung-Lab/rsl_rl
pip install -e .
```

- Verify that the extension is correctly installed by running the following command:

```bash
python legged_lab/scripts/train.py --task=walk  --logger=tensorboard --headless --num_envs=64
```

## Usage

### Visualize motion

Visualize the motion by updating the simulation with data from tienkung/datasets/motion_visualization.

```bash
python legged_lab/scripts/play_amp_animation.py --task=walk --num_envs=1
python legged_lab/scripts/play_amp_animation.py --task=run --num_envs=1
```

### Visualize motion with sensors

Visualize the motion with sensors by updating the simulation with data from tienkung/datasets/motion_visualization.

```bash
python legged_lab/scripts/play_amp_animation.py --task=walk_with_sensor --num_envs=1
python legged_lab/scripts/play_amp_animation.py --task=run_with_sensor --num_envs=1
```

### Train

Train the policy using AMP expert data from tienkung/datasets/motion_amp_expert.

```bash
python legged_lab/scripts/train.py --task=walk --headless --logger=tensorboard --num_envs=4096
python legged_lab/scripts/train.py --task=run --headless --logger=tensorboard --num_envs=4096
```

### Play

Run the trained policy.

```bash
python legged_lab/scripts/play.py --task=walk --num_envs=1
python legged_lab/scripts/play.py --task=run --num_envs=1
```

### Sim2Sim(MuJoCo)

Evaluate the trained policy in MuJoCo to perform cross-simulation validation.

Exported_policy/ contains pretrained policies provided by the project. When using the play script, trained policy is exported automatically and saved to path like logs/run/[timestamp]/exported/policy.pt.
```bash
python legged_lab/scripts/sim2sim.py --task walk --policy Exported_policy/walk.pt --duration 10
python legged_lab/scripts/sim2sim.py --task run --policy Exported_policy/run.pt --duration 10
```

### Tensorboard
```bash
tensorboard --logdir=logs/walk
tensorboard --logdir=logs/run
```

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/legged_lab",
        "<path-to-IsaacLab>/source/isaaclab_tasks",
        "<path-to-IsaacLab>/source/isaaclab_mimic",
        "<path-to-IsaacLab>/source/extensions",
        "<path-to-IsaacLab>/source/isaaclab_assets",
        "<path-to-IsaacLab>/source/isaaclab_rl",
        "<path-to-IsaacLab>/source/isaaclab",
    ]
}
```

## Acknowledgement

* [Legged Lab](https://github.com/Hellod035/LeggedLab): a direct IsaacLab Workflow for Legged Robots.
* [Humanoid-Gym](https://github.com/roboterax/humanoid-gym):a reinforcement learning (RL) framework based on NVIDIA Isaac Gym, with Sim2Sim support.
* [RSL RL](https://github.com/leggedrobotics/rsl_rl): a fast and simple implementation of RL algorithms.
* [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware?tab=readme-ov-file): codebase for learning skills from short reference motions using Adversarial Motion Priors.
* [Omni-Perception](https://acodedog.github.io/OmniPerceptionPages/): a perception library for legged robots, which provides a set of sensors and perception algorithms.
* [Warp](https://github.com/NVIDIA/warp): a Python framework for writing high-performance simulation and graphics code.

## Discussions
If you're interested in TienKung-Lab, welcome to join our WeChat group for discussions.

<img src="./docs/qrcode.png" border=0 width=40%>

