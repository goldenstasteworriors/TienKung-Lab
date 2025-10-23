# Pelvis 高度终止条件使用指南

## 概述

已为 TienKung-Lab 环境添加了 pelvis 高度终止条件。当机器人的 pelvis（骨盆/躯干）高度低于设定的阈值时，环境会自动终止该 episode，这有助于训练更稳定的行走策略。

## 实现位置

### 1. 配置参数
- **文件**: `legged_lab/envs/base/base_config.py:69`
- **参数**: `RobotCfg.terminate_pelvis_height`
- **默认值**: `0.0` (禁用该终止条件)
- **说明**: 设置为大于 0 的值时启用该终止条件

### 2. 终止逻辑
- **文件**:
  - `legged_lab/envs/base/base_env.py:270-274`
  - `legged_lab/envs/tienkung/tienkung_env.py:531-534`
- **逻辑**: 当 `pelvis_height < terminate_pelvis_height` 时触发终止

### 3. 高度监测
- **文件**: `legged_lab/envs/tienkung/tienkung_env.py:520-529`
- **功能**: 每 100 步打印一次 pelvis 高度统计信息（平均、最小、最大值）

## 使用步骤

### 步骤 1: 确定合适的阈值

运行以下命令观察正常行走时的 pelvis 高度：

```bash
# 方法 1: 使用播放模式（推荐）
python legged_lab/scripts/play.py --task=walk --num_envs=4

# 方法 2: 使用测试脚本
python test_pelvis_height.py
```

观察终端输出：
```
[Pelvis Height] Step 100: Avg=0.850m, Min=0.820m, Max=0.880m
[Pelvis Height] Step 200: Avg=0.845m, Min=0.810m, Max=0.875m
```

**建议阈值设置**：
- 观察多个 episode 的最小高度
- 选择一个比正常最小高度低 0.1-0.2m 的值作为阈值
- 例如：如果正常最小高度为 0.80m，可设置阈值为 0.60-0.70m

### 步骤 2: 在配置文件中设置阈值

编辑任务配置文件（如 `legged_lab/envs/tienkung/walk_cfg.py`）：

```python
robot: RobotCfg = RobotCfg(
    actor_obs_history_length=10,
    critic_obs_history_length=10,
    action_scale=0.25,
    terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
    feet_body_names=["ankle_roll.*"],
    terminate_pelvis_height=0.65,  # 添加这一行，设置为你确定的阈值
)
```

### 步骤 3: 运行训练

```bash
# 训练行走策略
python legged_lab/scripts/train.py --task=walk --headless --logger=tensorboard --num_envs=4096

# 训练跑步策略
python legged_lab/scripts/train.py --task=run --headless --logger=tensorboard --num_envs=4096
```

## 配置示例

### Walk 任务（推荐值）
```python
# 在 legged_lab/envs/tienkung/walk_cfg.py 中
robot: RobotCfg = RobotCfg(
    actor_obs_history_length=10,
    critic_obs_history_length=10,
    action_scale=0.25,
    terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
    feet_body_names=["ankle_roll.*"],
    terminate_pelvis_height=0.65,  # 行走任务的建议值
)
```

### Run 任务（推荐值）
```python
# 在 legged_lab/envs/tienkung/run_cfg.py 中
robot: RobotCfg = RobotCfg(
    actor_obs_history_length=10,
    critic_obs_history_length=10,
    action_scale=0.25,
    terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
    feet_body_names=["ankle_roll.*"],
    terminate_pelvis_height=0.60,  # 跑步任务可能需要更低的阈值
)
```

## 禁用该终止条件

如果不想使用该终止条件，保持默认值即可：

```python
terminate_pelvis_height=0.0  # 0 或负值表示禁用
```

## 技术细节

### Pelvis 高度的获取
- 使用 `self.robot.data.root_pos_w[:, 2]` 获取世界坐标系中的 Z 轴高度
- `root_pos_w` 是机器人根部（pelvis）在世界坐标系中的位置

### 终止条件触发
在 `check_reset()` 函数中：
```python
if self.cfg.robot.terminate_pelvis_height > 0.0:
    pelvis_height = self.robot.data.root_pos_w[:, 2]
    pelvis_too_low = pelvis_height < self.cfg.robot.terminate_pelvis_height
    reset_buf |= pelvis_too_low
```

### 监测输出
每 100 步输出统计信息：
```python
if self.episode_length_buf[0] % 100 == 0:
    avg_height = pelvis_height.mean().item()
    min_height = pelvis_height.min().item()
    max_height = pelvis_height.max().item()
    print(f"[Pelvis Height] Step {self.episode_length_buf[0].item()}: "
          f"Avg={avg_height:.3f}m, Min={min_height:.3f}m, Max={max_height:.3f}m")
```

## 清理

完成阈值确定后，可以删除临时测试脚本：
```bash
rm test_pelvis_height.py
```

如果不想在运行时看到 pelvis 高度的打印信息，可以注释掉 `tienkung_env.py` 中的打印代码（第 524-529 行）。
