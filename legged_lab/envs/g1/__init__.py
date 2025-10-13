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

"""Package containing G1 robot environments."""

from legged_lab.utils.task_registry import task_registry

from .g1_env import G1Env
from .g1_run_cfg import G1RunFlatEnvCfg, G1RunAgentCfg
from .g1_walk_cfg import G1WalkFlatEnvCfg, G1WalkAgentCfg

##
# Register G1 tasks
##

task_registry.register("g1_walk", G1Env, G1WalkFlatEnvCfg(), G1WalkAgentCfg())
task_registry.register("g1_run", G1Env, G1RunFlatEnvCfg(), G1RunAgentCfg())
