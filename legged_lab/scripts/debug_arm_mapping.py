"""
Debug script to check arm joint order mapping
"""

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

print("=" * 80)
print("GMR Output Joint Order (29 DOF)")
print("=" * 80)
for i, joint in enumerate(GMR_JOINT_ORDER):
    print(f"[{i:2d}] {joint}")

print("\n" + "=" * 80)
print("G1 Env Mapping in visualize_motion")
print("=" * 80)

print("\n### LEFT LEG (dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12])")
print("visual_motion_frame[6:12] maps to:")
for i in range(6, 12):
    print(f"  GMR[{i}] = {GMR_JOINT_ORDER[i]}")

print("\n### RIGHT LEG (dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18])")
print("visual_motion_frame[12:18] maps to:")
for i in range(12, 18):
    print(f"  GMR[{i}] = {GMR_JOINT_ORDER[i]}")

print("\n### LEFT ARM (dof_pos[:, self.left_arm_ids] = visual_motion_frame[18:22])")
print("visual_motion_frame[18:22] maps to:")
for i in range(18, 22):
    print(f"  GMR[{i}] = {GMR_JOINT_ORDER[i]}")

print("\n### RIGHT ARM (dof_pos[:, self.right_arm_ids] = visual_motion_frame[22:26])")
print("visual_motion_frame[22:26] maps to:")
for i in range(22, 26):
    print(f"  GMR[{i}] = {GMR_JOINT_ORDER[i]}")

print("\n" + "=" * 80)
print("⚠️  PROBLEM DETECTED!")
print("=" * 80)
print("""
The mapping assumes visual_motion_frame contains only the controllable joints (26 DOF),
but GMR outputs ALL 29 DOF including 3 waist joints!

Current mapping:
- visual_motion_frame[6:12]  → expects left leg joints
- visual_motion_frame[12:18] → expects right leg joints
- visual_motion_frame[18:22] → expects left arm joints (4 joints, no wrist)
- visual_motion_frame[22:26] → expects right arm joints (4 joints, no wrist)

But GMR data structure is:
[0:6]   left leg
[6:12]  right leg
[12:15] waist (3 joints) ← MISSING in current mapping!
[15:19] left arm (4 joints: shoulder_pitch/roll/yaw + elbow)
[19:22] left wrist (3 joints) ← Not used in environment
[22:26] right arm (4 joints: shoulder_pitch/roll/yaw + elbow)
[26:29] right wrist (3 joints) ← Not used in environment

SOLUTION: The mapping indices need to account for waist joints!
Should be:
- visual_motion_frame[0:6]   → left leg
- visual_motion_frame[6:12]  → right leg
- visual_motion_frame[15:19] → left arm (skip waist [12:15])
- visual_motion_frame[22:26] → right arm (skip left wrist [19:22])
""")
