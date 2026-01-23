#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# G1 AMP expert frame layout (70 dims):
# [0:29]   dof_pos (29)
# [29:58]  dof_vel (29)
# [58:70]  end_effector_pos (12)
#          = left_hand(3) + right_hand(3) + left_foot(3) + right_foot(3)


def load_amp_frames(json_path: Path) -> tuple[np.ndarray, float]:
    with json_path.open("r", encoding="utf-8") as f:
        motion = json.load(f)

    frames = np.asarray(motion["Frames"], dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"Frames must be 2D array, got shape={frames.shape}")
    if frames.shape[1] != 70:
        raise ValueError(f"Expected 70 dims per frame, got {frames.shape[1]} in {json_path}")

    frame_duration = float(motion.get("FrameDuration", 0.0))
    if frame_duration <= 0:
        raise ValueError("Missing/invalid FrameDuration in motion file")

    return frames, frame_duration


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot left/right foot height (z) over time from a G1 AMP expert file.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to AMP expert motion file (JSON .txt) with Frames and FrameDuration.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save the plot image (e.g. out.png). If omitted, only shows window.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window (useful when --out is also provided).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    frames, dt = load_amp_frames(input_path)
    t = np.arange(frames.shape[0], dtype=np.float32) * dt

    # End effector block [58:70]
    # left_foot is [64:67], right_foot is [67:70]
    left_foot = frames[:, 64:67]
    right_foot = frames[:, 67:70]
    left_z = left_foot[:, 2]
    right_z = right_foot[:, 2]

    plt.figure(figsize=(10, 4))
    plt.plot(t, left_z, label="left_foot z")
    plt.plot(t, right_z, label="right_foot z")
    plt.xlabel("time (s)")
    plt.ylabel("foot height z (m, root frame)")
    plt.title(input_path.name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"saved: {out_path}")

    if args.show or not args.out:
        plt.show()


if __name__ == "__main__":
    main()
