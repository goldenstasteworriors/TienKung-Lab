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

"""
AMPLoader_G1: Motion loader for G1 robot with 29 DOF for AMP training.

This class is specifically designed for G1's 29 DOF configuration:
- 12 leg joints (6 per leg)
- 3 waist joints
- 8 arm joints (4 per arm)
- 6 wrist joints (3 per wrist)

The AMP expert data format is:
[dof_pos(29) + dof_vel(29) + end_effector_pos(12)] = 70 dims total
- [0:29]   dof_pos (29 DOF)
- [29:58]  dof_vel (29 DOF)
- [58:70]  end_effector_pos (left_hand(3) + right_hand(3) + left_foot(3) + right_foot(3))
"""

import glob
import json

import numpy as np
import torch


class AMPLoader_G1:
    """AMP motion loader specifically designed for G1 robot with 29 DOF.

    This class handles the expert motion data for AMP (Adversarial Motion Priors) training.
    Unlike TienKung (20 DOF), G1 has 29 DOF including waist and wrist joints.

    Data structure (70 dims):
        [0:29]   dof_pos (29 DOF: legs(12) + waist(3) + arms(8) + wrists(6))
        [29:58]  dof_vel (29 DOF)
        [58:70]  end_effector_pos (4 end effectors × 3 coords = 12)
    """

    # G1 has 29 DOF
    JOINT_POS_SIZE = 29
    JOINT_VEL_SIZE = 29
    END_EFFECTOR_POS_SIZE = 12  # 4 end effectors (2 hands + 2 feet) × 3 coords

    # Data structure indices
    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE  # 29

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE  # 58

    END_POS_START_IDX = JOINT_VEL_END_IDX
    END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE  # 70

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
    ):
        """Initialize G1 AMP motion loader.

        Args:
            device: Torch device for tensor storage.
            time_between_frames: Time interval between frames in seconds.
            data_dir: Directory containing motion files (unused, kept for compatibility).
            preload_transitions: Whether to preload transitions for faster sampling.
            num_preload_transitions: Number of transitions to preload.
            motion_files: List of motion file paths to load.
        """
        self.device = device
        self.time_between_frames = time_between_frames

        # Storage for trajectories
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Trajectory length in seconds
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        # Load all motion files
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                # Validate data dimensions
                if motion_data.shape[1] != self.END_POS_END_IDX:
                    print(f"Warning: Motion file {motion_file} has {motion_data.shape[1]} dims, "
                          f"expected {self.END_POS_END_IDX} dims (29 dof_pos + 29 dof_vel + 12 end_effector).")

                # Store full 70-dim data
                self.trajectories.append(
                    torch.tensor(motion_data[:, : AMPLoader_G1.END_POS_END_IDX], dtype=torch.float32, device=device)
                )
                self.trajectories_full.append(
                    torch.tensor(motion_data[:, : AMPLoader_G1.END_POS_END_IDX], dtype=torch.float32, device=device)
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s motion ({motion_data.shape[0]} frames, {motion_data.shape[1]} dims) "
                  f"from {motion_file}.")

        # Normalize trajectory weights
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions if requested
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions for G1 (29 DOF)")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)

            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading G1 AMP transitions")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Sample trajectory index using weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample trajectory indices."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for a trajectory."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random times for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1, frame2, blend):
        """Linear interpolation between two frames."""
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """Returns full trajectory data."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low

        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time (batch version)."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """Batch version of get_full_frame_at_time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_amp_starts = torch.zeros(
            len(traj_idxs), AMPLoader_G1.END_POS_END_IDX - AMPLoader_G1.JOINT_POSE_START_IDX, device=self.device
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs), AMPLoader_G1.END_POS_END_IDX - AMPLoader_G1.JOINT_POSE_START_IDX, device=self.device
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoader_G1.JOINT_POSE_START_IDX : AMPLoader_G1.END_POS_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, AMPLoader_G1.JOINT_POSE_START_IDX : AMPLoader_G1.END_POS_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        """Returns batch of random full frames."""
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between the two frames.

        Returns:
            An interpolation of the two frames (70 dims).
        """
        joints0, joints1 = AMPLoader_G1.get_joint_pose(frame0), AMPLoader_G1.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader_G1.get_joint_vel(frame0), AMPLoader_G1.get_joint_vel(frame1)
        end_pos_0, end_pos_1 = AMPLoader_G1.get_end_pos(frame0), AMPLoader_G1.get_end_pos(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)
        blend_end_pos = self.slerp(end_pos_0, end_pos_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel, blend_end_pos])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, AMPLoader_G1.JOINT_POSE_START_IDX : AMPLoader_G1.END_POS_END_IDX]
                s_next = self.preloaded_s_next[idxs, AMPLoader_G1.JOINT_POSE_START_IDX : AMPLoader_G1.END_POS_END_IDX]
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations (70 dims for G1)."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        """Number of loaded motion files."""
        return len(self.trajectory_names)

    @staticmethod
    def get_joint_pose(pose):
        """Extract joint positions [0:29]."""
        return pose[AMPLoader_G1.JOINT_POSE_START_IDX : AMPLoader_G1.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_pose_batch(poses):
        """Extract joint positions from batch."""
        return poses[:, AMPLoader_G1.JOINT_POSE_START_IDX : AMPLoader_G1.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_vel(pose):
        """Extract joint velocities [29:58]."""
        return pose[AMPLoader_G1.JOINT_VEL_START_IDX : AMPLoader_G1.JOINT_VEL_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses):
        """Extract joint velocities from batch."""
        return poses[:, AMPLoader_G1.JOINT_VEL_START_IDX : AMPLoader_G1.JOINT_VEL_END_IDX]

    @staticmethod
    def get_end_pos(pose):
        """Extract end effector positions [58:70]."""
        return pose[AMPLoader_G1.END_POS_START_IDX : AMPLoader_G1.END_POS_END_IDX]

    @staticmethod
    def get_end_pos_batch(poses):
        """Extract end effector positions from batch."""
        return poses[:, AMPLoader_G1.END_POS_START_IDX : AMPLoader_G1.END_POS_END_IDX]
