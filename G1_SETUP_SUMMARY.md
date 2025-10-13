# Unitree G1 29DOF 支持 - 配置总结

## 完成的工作

已成功为 TienKung-Lab 项目添加 **Unitree G1 29DOF** 完整支持。

### 1. 机器人资源文件

**位置**: `legged_lab/assets/g1_29dof/`

- ✅ URDF 文件: `urdf/g1_29dof.urdf`
- ✅ USD 文件: `usd/g1_29dof.usd`
- ✅ 网格文件: `meshes/` (从 unitree_rl_gym 复制)
- ✅ 机器人配置: `g1_29dof.py`

**G1 29DOF 关节列表**:
```
腿部 (12 DOF):
  - left/right_hip_pitch_joint
  - left/right_hip_roll_joint
  - left/right_hip_yaw_joint
  - left/right_knee_joint
  - left/right_ankle_pitch_joint
  - left/right_ankle_roll_joint

腰部 (3 DOF):
  - waist_yaw_joint
  - waist_roll_joint
  - waist_pitch_joint

手臂 (14 DOF):
  - left/right_shoulder_pitch_joint
  - left/right_shoulder_roll_joint
  - left/right_shoulder_yaw_joint
  - left/right_elbow_joint
  - left/right_wrist_roll_joint
  - left/right_wrist_pitch_joint
  - left/right_wrist_yaw_joint
```

### 2. 环境配置

**位置**: `legged_lab/envs/g1/`

- ✅ 环境类: `g1_env.py` (G1Env, 继承自 BaseEnv)
- ✅ Walk 任务配置: `g1_walk_cfg.py` (G1WalkFlatEnvCfg)
- ✅ Run 任务配置: `g1_run_cfg.py` (G1RunFlatEnvCfg)
- ✅ 初始化文件: `__init__.py`

### 3. 任务注册

在 `legged_lab/envs/__init__.py` 中注册了以下任务:

- ✅ `g1_walk` - G1 行走任务
- ✅ `g1_run` - G1 跑步任务

### 4. 数据集目录

创建了动作数据集目录结构:

```
legged_lab/envs/g1/datasets/
├── motion_visualization/  # 用于 play_amp_animation.py 可视化
└── motion_amp_expert/     # 用于 AMP 训练的专家数据
```

## 使用方法

### 训练

```bash
# 训练 G1 行走策略
python legged_lab/scripts/train.py --task=g1_walk --headless --logger=tensorboard --num_envs=4096

# 训练 G1 跑步策略
python legged_lab/scripts/train.py --task=g1_run --headless --logger=tensorboard --num_envs=4096
```

### 测试与可视化

```bash
# 运行训练好的 G1 策略
python legged_lab/scripts/play.py --task=g1_walk --num_envs=1
python legged_lab/scripts/play.py --task=g1_run --num_envs=1

# 可视化 GMR 生成的动作数据
python legged_lab/scripts/play_amp_animation.py --task=g1_walk --num_envs=1
```

### GMR 数据转换

将 GMR 生成的 G1 动作数据转换为项目格式:

```bash
# 转换为可视化格式
conda run -n TieKung python legged_lab/scripts/gmr_data_conversion.py \
  --input_pkl /home/ykj/project/GMR/save/g1/dance1_2.pkl \
  --output_txt legged_lab/envs/g1/datasets/motion_visualization/dance1_2.txt \
  --fps 30.0
```

然后在配置文件中设置:
```python
# legged_lab/envs/g1/g1_walk_cfg.py 或 g1_run_cfg.py
amp_motion_files_display = ["legged_lab/envs/g1/datasets/motion_visualization/dance1_2.txt"]
```

## 关键差异: G1 vs TienKung2-Lite

| 特性 | G1 29DOF | TienKung2-Lite |
|------|----------|----------------|
| **总自由度** | 29 | 18 |
| **腰部关节** | ✅ 3 DOF (yaw, roll, pitch) | ❌ 无 |
| **手腕关节** | ✅ 6 DOF (3 per wrist) | ❌ 无 |
| **关节命名** | `left_hip_pitch_joint` | `hip_pitch_l_joint` |
| **Link命名** | `left_hip_pitch_link` | `hip_pitch_l_link` |
| **初始高度** | 0.75m | 1.0m |

## 配置文件修改

所有 G1 配置文件已经自动适配:

1. **关节名称模式**:
   - TienKung: `hip_pitch_.*_joint` → G1: `.*_hip_pitch_joint`
   - TienKung: `knee_pitch_.*_joint` → G1: `.*_knee_joint`
   - TienKung: `elbow_pitch_.*_joint` → G1: `.*_elbow_joint`

2. **Body 名称模式**:
   - TienKung: `ankle_roll.*` → G1: `.*_ankle_roll_link`
   - TienKung: `knee_pitch.*` → G1: `.*_knee_link`

3. **新增执行器组**:
   - `waist`: 腰部 3 个关节
   - `arms_wrist`: 手腕 6 个关节

## 文件结构

```
TienKung-Lab/
├── legged_lab/
│   ├── assets/
│   │   └── g1_29dof/
│   │       ├── g1_29dof.py          # 机器人配置
│   │       ├── urdf/
│   │       │   ├── g1_29dof.urdf
│   │       │   └── g1_29dof.xml
│   │       ├── meshes/              # STL 网格文件
│   │       └── usd/
│   │           └── g1_29dof.usd     # Isaac Sim USD 文件
│   │
│   └── envs/
│       └── g1/
│           ├── __init__.py          # 任务注册
│           ├── g1_env.py            # 环境类
│           ├── g1_walk_cfg.py       # Walk 配置
│           ├── g1_run_cfg.py        # Run 配置
│           └── datasets/
│               ├── motion_visualization/
│               └── motion_amp_expert/
```

## 验证状态

✅ 所有 Python 文件语法正确
✅ USD 文件成功生成
✅ 任务已注册到 task_registry
✅ 配置文件关节名称已适配
✅ 目录结构完整

## 下一步

1. **准备训练数据**: 使用 GMR 将 AMASS/OMOMO 动作数据重定向到 G1
2. **生成专家数据**: 运行 `play_amp_animation.py --save_path` 生成 AMP 训练数据
3. **开始训练**: 使用上述训练命令训练 G1 策略
4. **Sim2Sim 验证**: 在 MuJoCo 中测试导出的策略

## 注意事项

- G1 的关节命名规则为 `{side}_{joint_name}_joint` (如 `left_hip_pitch_joint`)
- TienKung 的关节命名规则为 `{joint_name}_{side}_joint` (如 `hip_pitch_l_joint`)
- 所有配置文件中的正则表达式已自动适配
- GMR 生成的 G1 数据可以直接使用，因为 G1 是 GMR 支持的标准机器人
