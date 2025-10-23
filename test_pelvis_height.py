#!/usr/bin/env python3
"""
临时测试脚本：用于观察 pelvis 高度并确定终止阈值
完成后将被删除
"""

import subprocess
import sys

def main():
    print("=" * 80)
    print("Pelvis 高度测试脚本")
    print("=" * 80)
    print("\n请选择测试模式:")
    print("1. 播放已训练的策略 (推荐 - 可以看到正常行走的 pelvis 高度)")
    print("2. 运行训练 (可以看到训练过程中的 pelvis 高度)")

    choice = input("\n请输入选择 (1 或 2): ").strip()

    if choice == "1":
        print("\n正在启动播放模式...")
        print("请观察终端输出的 [Pelvis Height] 信息")
        print("这将显示 pelvis 的平均、最小和最大高度")
        print("-" * 80)

        # 使用 play.py 播放已训练的策略
        cmd = [
            "python", "legged_lab/scripts/play.py",
            "--task=walk",
            "--num_envs=4"
        ]
        subprocess.run(cmd)

    elif choice == "2":
        print("\n正在启动训练模式...")
        print("请观察终端输出的 [Pelvis Height] 信息")
        print("-" * 80)

        # 使用短时间训练来观察
        cmd = [
            "python", "legged_lab/scripts/train.py",
            "--task=walk",
            "--headless",
            "--num_envs=64"
        ]
        subprocess.run(cmd)

    else:
        print("无效选择！")
        sys.exit(1)

if __name__ == "__main__":
    main()
