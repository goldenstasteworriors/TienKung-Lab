#!/usr/bin/env python3
"""Test script to verify G1 29DOF setup"""

import sys
print("Testing G1 29DOF setup...")

# Test 1: Import G1 robot configuration
print("\n[1/4] Testing G1 robot configuration import...")
try:
    from legged_lab.assets.g1_29dof.g1_29dof import G1_29DOF_CFG
    print("✅ G1_29DOF_CFG imported successfully")
    print(f"   - Number of actuator groups: {len(G1_29DOF_CFG.actuators)}")
    print(f"   - Initial position: {G1_29DOF_CFG.init_state.pos}")
except Exception as e:
    print(f"❌ Failed to import G1_29DOF_CFG: {e}")
    sys.exit(1)

# Test 2: Import G1 environment
print("\n[2/4] Testing G1 environment import...")
try:
    from legged_lab.envs.g1.g1_env import G1Env
    print("✅ G1Env imported successfully")
except Exception as e:
    print(f"❌ Failed to import G1Env: {e}")
    sys.exit(1)

# Test 3: Import G1 task configs
print("\n[3/4] Testing G1 task configurations import...")
try:
    from legged_lab.envs.g1.g1_walk_cfg import G1WalkFlatEnvCfg, G1WalkAgentCfg
    from legged_lab.envs.g1.g1_run_cfg import G1RunFlatEnvCfg, G1RunAgentCfg
    print("✅ G1 walk and run configs imported successfully")
except Exception as e:
    print(f"❌ Failed to import G1 configs: {e}")
    sys.exit(1)

# Test 4: Check task registry
print("\n[4/4] Testing task registry...")
try:
    from legged_lab.utils import task_registry

    registered_tasks = task_registry.task_registry._task_env_cfgs.keys()
    print(f"✅ Task registry loaded, total tasks: {len(registered_tasks)}")

    g1_tasks = [t for t in registered_tasks if 'g1' in t.lower()]
    print(f"\n📋 G1 tasks registered:")
    for task in sorted(g1_tasks):
        print(f"   - {task}")

    if not g1_tasks:
        print("❌ No G1 tasks found in registry!")
        sys.exit(1)

except Exception as e:
    print(f"❌ Failed to check task registry: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ All tests passed! G1 29DOF is properly configured.")
print("="*60)
print("\n📝 You can now use the following tasks:")
print("   python legged_lab/scripts/train.py --task=g1_walk")
print("   python legged_lab/scripts/train.py --task=g1_run")
print("   python legged_lab/scripts/play.py --task=g1_walk --num_envs=1")
print("   python legged_lab/scripts/play.py --task=g1_run --num_envs=1")
