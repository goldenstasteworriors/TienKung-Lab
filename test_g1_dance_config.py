#!/usr/bin/env python3
"""测试 G1 dance1_2.txt 配置是否正确"""

import os
import json

# 检查文件是否存在
dance_file = "legged_lab/envs/g1/datasets/motion_visualization/dance1_2.txt"
if os.path.exists(dance_file):
    print(f"✅ 找到文件: {dance_file}")

    # 读取文件头部信息
    with open(dance_file, 'r') as f:
        content = f.read(500)
        print(f"\n文件前500字符:")
        print(content)

    # 尝试解析 JSON
    try:
        with open(dance_file, 'r') as f:
            data = json.load(f)
        print(f"\n✅ JSON 格式正确")
        print(f"  - LoopMode: {data.get('LoopMode')}")
        print(f"  - FrameDuration: {data.get('FrameDuration')}")
        print(f"  - 帧数: {len(data.get('Frames', []))}")
        if len(data.get('Frames', [])) > 0:
            print(f"  - 每帧数据维度: {len(data['Frames'][0])}")
    except Exception as e:
        print(f"\n❌ JSON 解析错误: {e}")
else:
    print(f"❌ 文件不存在: {dance_file}")

# 检查配置文件
print("\n" + "="*60)
print("检查 G1 配置文件:")
print("="*60)

walk_cfg = "legged_lab/envs/g1/g1_walk_cfg.py"
if os.path.exists(walk_cfg):
    with open(walk_cfg, 'r') as f:
        for i, line in enumerate(f, 1):
            if 'amp_motion_files_display' in line:
                print(f"✅ {walk_cfg}:{i}")
                print(f"   {line.strip()}")

run_cfg = "legged_lab/envs/g1/g1_run_cfg.py"
if os.path.exists(run_cfg):
    with open(run_cfg, 'r') as f:
        for i, line in enumerate(f, 1):
            if 'amp_motion_files_display' in line:
                print(f"✅ {run_cfg}:{i}")
                print(f"   {line.strip()}")

print("\n配置检查完成!")
