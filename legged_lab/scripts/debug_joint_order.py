"""
Debug script to check joint order mismatch between GMR output and G1 environment
"""
import pickle
import sys

# GMR G1 29DOF joint order (from MuJoCo XML)
GMR_JOINT_ORDER = [
    "left_hip_pitch_joint",      # 0
    "left_hip_roll_joint",       # 1
    "left_hip_yaw_joint",        # 2
    "left_knee_joint",           # 3
    "left_ankle_pitch_joint",    # 4
    "left_ankle_roll_joint",     # 5
    "right_hip_pitch_joint",     # 6
    "right_hip_roll_joint",      # 7
    "right_hip_yaw_joint",       # 8
    "right_knee_joint",          # 9
    "right_ankle_pitch_joint",   # 10
    "right_ankle_roll_joint",    # 11
    "waist_yaw_joint",           # 12
    "waist_roll_joint",          # 13
    "waist_pitch_joint",         # 14
    "left_shoulder_pitch_joint", # 15
    "left_shoulder_roll_joint",  # 16
    "left_shoulder_yaw_joint",   # 17
    "left_elbow_joint",          # 18
    "left_wrist_roll_joint",     # 19
    "left_wrist_pitch_joint",    # 20
    "left_wrist_yaw_joint",      # 21
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26
    "right_wrist_pitch_joint",    # 27
    "right_wrist_yaw_joint",      # 28
]

# G1 Env expected order (from g1_env.py find_joints with preserve_order=True)
G1_ENV_LEFT_LEG = [
    "left_hip_roll_joint",    # Index in robot
    "left_hip_pitch_joint",   # Index in robot
    "left_hip_yaw_joint",     # Index in robot
    "left_knee_joint",        # Index in robot
    "left_ankle_pitch_joint", # Index in robot
    "left_ankle_roll_joint",  # Index in robot
]

G1_ENV_RIGHT_LEG = [
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

G1_ENV_LEFT_ARM = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]

G1_ENV_RIGHT_ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

print("=" * 80)
print("GMR Joint Order vs G1 Environment Expected Order")
print("=" * 80)

print("\n### LEFT LEG MAPPING ###")
print("GMR dof_pos[0:6]:")
for i in range(6):
    print(f"  [{i}] {GMR_JOINT_ORDER[i]}")

print("\nG1 Env expects (self.left_leg_ids):")
for i, joint in enumerate(G1_ENV_LEFT_LEG):
    print(f"  [{i}] {joint}")

print("\n⚠️  MISMATCH DETECTED:")
print("  GMR [0] = left_hip_pitch_joint")
print("  G1 Env expects [0] = left_hip_roll_joint")
print("  These don't match!")

print("\n### RIGHT LEG MAPPING ###")
print("GMR dof_pos[6:12]:")
for i in range(6, 12):
    print(f"  [{i-6}] {GMR_JOINT_ORDER[i]}")

print("\nG1 Env expects (self.right_leg_ids):")
for i, joint in enumerate(G1_ENV_RIGHT_LEG):
    print(f"  [{i}] {joint}")

print("\n⚠️  MISMATCH DETECTED:")
print("  GMR [6] = right_hip_pitch_joint")
print("  G1 Env expects [0] = right_hip_roll_joint")
print("  These don't match!")

print("\n### LEFT ARM MAPPING ###")
print("GMR dof_pos[15:19] (excluding wrist):")
for i in range(15, 19):
    print(f"  [{i-15}] {GMR_JOINT_ORDER[i]}")

print("\nG1 Env expects (self.left_arm_ids):")
for i, joint in enumerate(G1_ENV_LEFT_ARM):
    print(f"  [{i}] {joint}")

print("\n✓ ARM MAPPING LOOKS CORRECT")

print("\n" + "=" * 80)
print("SOLUTION:")
print("=" * 80)
print("""
The issue is that GMR outputs joints in the order they appear in the MuJoCo XML,
which follows the kinematic tree structure:
  left_hip_pitch → left_hip_roll → left_hip_yaw

But find_joints() in IsaacSim returns joints in alphabetical order when using
regex patterns like "left_hip_*_joint":
  left_hip_pitch → left_hip_roll → left_hip_yaw (alphabetically)

Wait, that's the same order! Let me check the actual URDF/USD joint order...

Actually, the issue might be that IsaacSim's find_joints() returns joints in
the order defined in the URDF, not alphabetically. Need to verify the actual
order returned by find_joints().
""")
