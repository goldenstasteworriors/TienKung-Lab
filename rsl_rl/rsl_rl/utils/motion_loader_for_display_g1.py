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
AMPLoaderDisplay_G1: Motion loader for G1 robot with 29 DOF.

This class extends the original AMPLoaderDisplay to support G1's 29 DOF
without data truncation. It reads the full 70-dim motion data:
[root_pos(3) + root_rot(3) + dof_pos(29) + root_lin_vel(3) + root_ang_vel(3) + dof_vel(29)]
"""

import glob
import json

import numpy as np
import torch


class AMPLoaderDisplay_G1:
    """Motion loader specifically designed for G1 robot with 29 DOF.

    Data format (70 dims):
        [0:3]    root_pos
        [3:6]    root_rot (euler angles)
        [6:35]   dof_pos (29 DOF)
        [35:38]  root_lin_vel
        [38:41]  root_ang_vel
        [41:70]  dof_vel (29 DOF)

    Unlike the original AMPLoaderDisplay which truncates at 52 dims,
    this class reads all 70 dims to preserve complete velocity information.
    """

    # G1 has 29 DOF (including waist and wrist joints)
    DOF_SIZE = 29

    # Data structure indices
    ROOT_POS_SIZE = 3
    ROOT_ROT_SIZE = 3
    ROOT_VEL_SIZE = 3
    ROOT_ANG_VEL_SIZE = 3

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + ROOT_POS_SIZE  # 3

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROOT_ROT_SIZE  # 6

    DOF_POS_START_IDX = ROOT_ROT_END_IDX
    DOF_POS_END_IDX = DOF_POS_START_IDX + DOF_SIZE  # 35

    ROOT_LIN_VEL_START_IDX = DOF_POS_END_IDX
    ROOT_LIN_VEL_END_IDX = ROOT_LIN_VEL_START_IDX + ROOT_VEL_SIZE  # 38

    ROOT_ANG_VEL_START_IDX = ROOT_LIN_VEL_END_IDX
    ROOT_ANG_VEL_END_IDX = ROOT_ANG_VEL_START_IDX + ROOT_ANG_VEL_SIZE  # 41

    DOF_VEL_START_IDX = ROOT_ANG_VEL_END_IDX
    DOF_VEL_END_IDX = DOF_VEL_START_IDX + DOF_SIZE  # 70

    TOTAL_DATA_SIZE = DOF_VEL_END_IDX  # 70

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
    ):
        """Initialize G1 motion loader.

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
                if motion_data.shape[1] != self.TOTAL_DATA_SIZE:
                    print(f"Warning: Motion file {motion_file} has {motion_data.shape[1]} dims, "
                          f"expected {self.TOTAL_DATA_SIZE} dims.")

                # Store full 70-dim data without truncation
                self.trajectories_full.append(
                    torch.tensor(motion_data, dtype=torch.float32, device=device)
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
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading")

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

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full 70-dim frame for the given trajectory at the specified time.

        Args:
            traj_idx: Index of the trajectory.
            time: Time in seconds within the trajectory.

        Returns:
            Tensor of shape (70,) containing the interpolated frame.
        """
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        idx_low = min(idx_low, n - 1)
        idx_high = min(idx_high, n - 1)

        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low

        return self.blend_frame(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """Batch version of get_full_frame_at_time.

        Args:
            traj_idxs: Array of trajectory indices.
            times: Array of times corresponding to each trajectory.

        Returns:
            Tensor of shape (batch_size, 70) containing interpolated frames.
        """
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)

        all_frame_starts = torch.zeros(len(traj_idxs), self.TOTAL_DATA_SIZE, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.TOTAL_DATA_SIZE, device=self.device)

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            # Clamp indices to valid range
            idx_low_clamped = np.minimum(idx_low[traj_mask], trajectory.shape[0] - 1)
            idx_high_clamped = np.minimum(idx_high[traj_mask], trajectory.shape[0] - 1)
            all_frame_starts[traj_mask] = trajectory[idx_low_clamped]
            all_frame_ends[traj_mask] = trajectory[idx_high_clamped]

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame(self):
        """Returns random full 70-dim frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        """Returns batch of random full frames.

        Args:
            num_frames: Number of frames to sample.

        Returns:
            Tensor of shape (num_frames, 70).
        """
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame(self, frame0, frame1, blend):
        """Linearly interpolate between two complete frames.

        For G1, all 70 dims are interpolated linearly:
        - Root position, rotation, velocities: linear interpolation
        - DOF positions and velocities: linear interpolation

        Args:
            frame0: First frame (blend = 0).
            frame1: Second frame (blend = 1).
            blend: Float between [0, 1] for interpolation.

        Returns:
            Interpolated frame of shape (70,).
        """
        return self.slerp(frame0, frame1, blend)

    @property
    def observation_dim(self):
        """Size of full observations (70 dims)."""
        return self.TOTAL_DATA_SIZE

    @property
    def num_motions(self):
        """Number of loaded motion files."""
        return len(self.trajectory_names)

    # Static methods for extracting specific components
    @staticmethod
    def get_root_pos(frame):
        """Extract root position [0:3]."""
        return frame[AMPLoaderDisplay_G1.ROOT_POS_START_IDX : AMPLoaderDisplay_G1.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_rot(frame):
        """Extract root rotation (euler) [3:6]."""
        return frame[AMPLoaderDisplay_G1.ROOT_ROT_START_IDX : AMPLoaderDisplay_G1.ROOT_ROT_END_IDX]

    @staticmethod
    def get_dof_pos(frame):
        """Extract DOF positions [6:35] (29 DOF)."""
        return frame[AMPLoaderDisplay_G1.DOF_POS_START_IDX : AMPLoaderDisplay_G1.DOF_POS_END_IDX]

    @staticmethod
    def get_root_lin_vel(frame):
        """Extract root linear velocity [35:38]."""
        return frame[AMPLoaderDisplay_G1.ROOT_LIN_VEL_START_IDX : AMPLoaderDisplay_G1.ROOT_LIN_VEL_END_IDX]

    @staticmethod
    def get_root_ang_vel(frame):
        """Extract root angular velocity [38:41]."""
        return frame[AMPLoaderDisplay_G1.ROOT_ANG_VEL_START_IDX : AMPLoaderDisplay_G1.ROOT_ANG_VEL_END_IDX]

    @staticmethod
    def get_dof_vel(frame):
        """Extract DOF velocities [41:70] (29 DOF)."""
        return frame[AMPLoaderDisplay_G1.DOF_VEL_START_IDX : AMPLoaderDisplay_G1.DOF_VEL_END_IDX]

    # Batch versions
    @staticmethod
    def get_dof_pos_batch(frames):
        """Extract DOF positions from batch."""
        return frames[:, AMPLoaderDisplay_G1.DOF_POS_START_IDX : AMPLoaderDisplay_G1.DOF_POS_END_IDX]

    @staticmethod
    def get_dof_vel_batch(frames):
        """Extract DOF velocities from batch."""
        return frames[:, AMPLoaderDisplay_G1.DOF_VEL_START_IDX : AMPLoaderDisplay_G1.DOF_VEL_END_IDX]
