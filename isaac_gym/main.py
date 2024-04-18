import random
from isaacgym import gymapi, gymtorch
import torch

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1/60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

import sys
if len(sys.argv) > 1 and sys.argv[1] == "cpu":
    use_gpu = False
else:
    use_gpu = True

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

sim_params.physx.max_gpu_contact_pairs = sim_params.physx.max_gpu_contact_pairs*20

sim = gym.create_sim(0,
                     0,
                     gymapi.SIM_PHYSX,
                     sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

asset_root = "../models"
asset_file = "oleg.xml"
asset = gym.load_asset(sim, asset_root, asset_file)

import math
# set up the env grid
n = 2
num_envs = n * n
envs_per_row = int(math.sqrt(num_envs))
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.01, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

camera_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    height = 20 + random.uniform(1.0, 2.5)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, height)

    actor_handle = gym.create_actor(env, asset, pose, f"MyActor_{i}", i, 1)
    actor_handles.append(actor_handle)

import tqdm
with tqdm.tqdm() as bar:
    while True:
        # num_dofs = gym.get_sim_dof_count(sim)
        # actions = 1.0 - 2.0 * torch.rand(num_dofs, dtype=torch.float32)
        # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(actions))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        bar.update(1)
        import time
        # time.sleep(1.0)
        # import pdb; pdb.set_trace()