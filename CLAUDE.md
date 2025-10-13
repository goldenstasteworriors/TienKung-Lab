# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

TienKung-Lab 是一个基于 IsaacLab 的全尺寸人形机器人(TienKung)强化学习运动控制框架。该框架集成了 AMP(Adversarial Motion Priors)奖励和周期性步态奖励,支持自然稳定的行走和跑步行为,并提供 Sim2Sim 到 MuJoCo 的迁移能力。

### 技术栈
- **IsaacSim**: 4.5.0
- **IsaacLab**: 2.1.0
- **RSL_RL**: 2.3.1
- **Python**: 3.10
- **MuJoCo**: 3.3.2 (用于 Sim2Sim 验证)

## 常用命令

### 安装与设置
```bash
# 安装主包
pip install -e .

# 安装 RSL-RL 库
cd rsl_rl && pip install -e .

# 验证安装
python legged_lab/scripts/train.py --task=walk --logger=tensorboard --headless --num_envs=64
```

### 训练
```bash
# 训练行走策略
python legged_lab/scripts/train.py --task=walk --headless --logger=tensorboard --num_envs=4096

# 训练跑步策略
python legged_lab/scripts/train.py --task=run --headless --logger=tensorboard --num_envs=4096
```

### 测试与可视化
```bash
# 运行训练好的策略
python legged_lab/scripts/play.py --task=walk --num_envs=1
python legged_lab/scripts/play.py --task=run --num_envs=1

# 可视化动作(不带传感器)
python legged_lab/scripts/play_amp_animation.py --task=walk --num_envs=1
python legged_lab/scripts/play_amp_animation.py --task=run --num_envs=1

# 可视化动作(带传感器)
python legged_lab/scripts/play_amp_animation.py --task=walk_with_sensor --num_envs=1
python legged_lab/scripts/play_amp_animation.py --task=run_with_sensor --num_envs=1
```

### Sim2Sim 验证
```bash
# 在 MuJoCo 中评估策略
python legged_lab/scripts/sim2sim.py --task walk --policy Exported_policy/walk.pt --duration 10
python legged_lab/scripts/sim2sim.py --task run --policy Exported_policy/run.pt --duration 10
```

### TensorBoard
```bash
tensorboard --logdir=logs/walk
tensorboard --logdir=logs/run
```

### 代码格式化
```bash
pre-commit run --all-files
```

## 架构设计

### 核心组件层次结构

1. **任务注册系统** (`legged_lab/utils/task_registry.py`)
   - 全局任务注册表管理所有可用任务
   - 注册的任务: `walk`, `run`, `walk_with_sensor`, `run_with_sensor`
   - 每个任务包含环境类、环境配置和智能体配置

2. **基础环境架构** (`legged_lab/envs/base/`)
   - `BaseEnv`: 继承自 `VecEnv` 的基础环境类
   - `BaseEnvCfg`: 环境配置基类,包含场景、奖励、域随机化等配置
   - `BaseAgentCfg`: 智能体配置基类,包含 PPO/AMP 算法参数

3. **TienKung 环境实现** (`legged_lab/envs/tienkung/`)
   - `TienKungEnv`: TienKung 机器人的具体环境实现
   - 配置文件:
     - `walk_cfg.py`: 行走任务配置
     - `run_cfg.py`: 跑步任务配置
     - `walk_with_sensor_cfg.py`: 带传感器的行走配置
     - `run_with_sensor_cfg.py`: 带传感器的跑步配置

4. **奖励与感知系统**
   - **MDP 奖励** (`legged_lab/mdp/rewards.py`): 定义各种奖励函数
   - **传感器** (`legged_lab/sensors/`):
     - 相机系统 (包括 D455 深度相机、Luxonis OAK-D)
     - 激光雷达
     - 射线投射高度扫描器

5. **RSL-RL 算法** (`rsl_rl/rsl_rl/`)
   - `algorithms/ppo.py`: PPO 算法实现
   - `algorithms/amp_ppo.py`: AMP-PPO 算法实现
   - `modules/actor_critic.py`: Actor-Critic 网络
   - `runners/on_policy_runner.py`: 训练循环管理
   - `runners/amp_on_policy_runner.py`: AMP 训练循环

### 动作重定向工作流

项目支持从 SMPLX 格式(AMASS、OMOMO)到 TienKung 机器人的动作重定向,使用 GMR 工具:

1. **数据集结构** (`legged_lab/envs/tienkung/datasets/`):
   - `motion_visualization/`: 用于动作回放检查,包含 [root_pos, root_rot, dof_pos, root_lin_vel, root_ang_vel, dof_vel]
   - `motion_amp_expert/`: 用于 AMP 训练的专家数据,包含 [dof_pos, dof_vel, end-effector pos]

2. **数据转换流程**:
   - GMR 重定向: `scripts/smplx_to_robot.py` (外部工具)
   - 可视化数据转换: `legged_lab/scripts/gmr_data_conversion.py`
   - 专家数据生成: `legged_lab/scripts/play_amp_animation.py --save_path`

### 关键配置参数

- **步态参数** (`GaitCfg`):
  - `gait_cycle`: 步态周期 (walk: 0.85s, run: 0.5s)
  - `gait_air_ratio_l/r`: 左右脚离地比例
  - `gait_phase_offset_l/r`: 左右脚相位偏移

- **奖励权重** (`LiteRewardCfg`):
  - 速度跟踪、能量消耗、关节加速度、不期望接触等
  - 使用 IsaacLab 的 `RewardTermCfg` 系统

- **仿真参数**:
  - `physics_dt`: 0.005s (200Hz)
  - `step_dt`: physics_dt * decimation (通常 decimation=4, 即 50Hz 控制频率)

### 导出与部署

- 训练脚本自动导出策略为 JIT 和 ONNX 格式
- 导出路径: `logs/{task}/{timestamp}/exported/`
- 策略包含观察归一化器,可直接用于 Sim2Sim 或实际部署

### 与 IsaacLab 的集成

- 使用 `InteractiveScene` 管理仿真场景
- 通过 `SceneEntityCfg` 配置机器人、传感器、地形
- 事件管理器 (`EventManager`) 处理域随机化
- 命令生成器 (`UniformVelocityCommand`) 生成速度指令

## 开发注意事项

1. **带传感器的任务**: 在 `train.py` 中自动检测任务名包含 "sensor" 时启用相机渲染
2. **策略导出**: `play.py` 运行时自动导出策略到 `exported/` 目录
3. **日志路径**: 训练日志保存在 `logs/{experiment_name}/{timestamp}/`
4. **Conda 环境**: 项目通常使用与项目名相关的 conda 环境
