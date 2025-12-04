import numpy as np
import torch
import genesis as gs

from genesis.utils.geom import xyz_to_quat, quat_to_xyz, \
    transform_by_quat, quat_to_R, inv_quat, transform_quat_by_quat,\
    R_to_quat
import copy
import math
import open3d as o3d
from scipy.spatial.transform import Rotation as R
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	def clear(self):
		max_size = self.max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, self.state_dim))
		self.action = np.zeros((max_size, self.action_dim))
		self.next_state = np.zeros((max_size, self.state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
'''
MoveCube environment configuration
1. The parallelism is disabled. To simulate the behavior from real world,
where parallelism is not possible, num_envs is forced to be 1;
2. The episode length is also disabled, since it's not used in non-parallelized environment,
and it is compatible with TD3 implementation;
'''
def get_cfgs():
    env_cfg = {
        "num_actions": 3,
        # termination
        "termination_if_x_greater_than": 1,
        "termination_if_y_greater_than": 1,
        "termination_if_angle_greater_than": 4*np.pi,
        # base pose
        "episode_length_s": 0.5,
        "at_target_threshold": 0.03,
        "simulate_action_latency": True,
        "clip_actions": 0.3,
        "horizon_size": 0.3,
        # visualization
        "visualize_target": True,
        "visualize_camera": True,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "num_obs": 3, # NOTE: 6-> gripper & object, 3 -> object
        "obs_length": 3,
        "obs_scales": {
            "rel_pose": 1,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        # "yaw_lambda": -10.0,
        "reward_scales": {
            "target": 100.0,
            # "smooth": -1e-1,
            # "yaw": 0.01,
            # "stuck": 0.1, 
            # "angular": -2e-2,
            "crash": -10000.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg

class MoveCubeEnv:
    def __init__(self, obj_filename, grasp_pose, \
                 num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        assert num_envs == 1, "MoveCubeEnv is not parallelized, num_envs must be 1"
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_obs = obs_cfg["num_obs"]
        self.obs_length = obs_cfg["obs_length"] if "obs_length" in obs_cfg else 1
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(0.0, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num)), shadow=False),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            # renderer=gs.options.renderers.RayTracer(),
            show_viewer=show_viewer,
        )
        # self.scene.add_light(morph=gs.morphs.Sphere(pos=(0, 0, 1), radius = 0.2,), color=(0, 0, 0),intensity=200000)
        # add plane
        plane = self.scene.add_entity(gs.morphs.Plane())
        plane.set_friction(0.01)
        
        # Read the height of the lowest point
        mesh = o3d.io.read_triangle_mesh(obj_filename)
        height = -np.min(np.asarray(mesh.vertices)[:, 2])
        self.height = height

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                gs.morphs.Mesh(
                    file = obj_filename,
                    pos = [0, 0, height],
                    quat = [1, 0, 0, 0],
                    fixed = True,
                    collision = False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
        )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(0.0, 0.0, 1.0),
                lookat=(0, 0, 0.0),
                fov=40,
                GUI=False,
            )

        # add the entities:
        # 1. A cube to move
        # 2. A floating gripper to move the cube
        if self.env_cfg["horizon_size"]:
            self.horizon_size = self.env_cfg["horizon_size"]
        else:
            self.horizon_size = 0.3
        

        self.cube = self.scene.add_entity(
            gs.morphs.Mesh(
                file = obj_filename,
                pos = [0, 0, height],
                quat = [1, 0, 0, 0],
                convexify=False,
                decompose_nonconvex=True,
            ),
            surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(
                            color=(65/225,152/255,10/255),
                        ),
                    ),
            # Reduce the weight
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        self.franka_gripper = self.scene.add_entity(
            gs.morphs.URDF(
                file = "./panda_v2_gripper.urdf",
            ),
            # Ignore gravity for the robot
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        # build scene
        self.scene.build(n_envs=num_envs)
        
        # initialize other necessary attributes of the environment
        # indices of the dofs
        self.base_dofs = np.arange(6)
        self.fingers_dof = np.arange(6, 8)

        # set control gains
        self.franka_gripper.set_dofs_kp(
            np.array([40, 40, 40, 40, 40, 40, 40, 40]),
        )
        self.franka_gripper.set_dofs_kv(
            np.array([20, 20, 20, 20, 20, 20, 10, 10]),
        )
        self.franka_gripper.set_dofs_force_range(
            np.array([-87, -87, -87,  -80, -80, -80, -100, -100]),
            np.array([ 87,  87,  87,  80,  80,  80,  100,  100]),
        )


        # get the end-effector link
        self.end_effector = self.franka_gripper.get_link('panda_hand')

        # The initial grasp pose
        self.grasp_pose = grasp_pose.copy()

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.obs_length *  self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.obj_pos = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.obj_ang= torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float)
        self.obj_lin_vel = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.obj_ang_vel = torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float)
        self.obj_pose = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_obj_pose = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.gripper_pose = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
    def close(self):
        gs.destroy()

    def sample_action(self):
        # Sample a random action from the action space
        return (2 * self.env_cfg["clip_actions"] * torch.rand(self.num_envs, self.num_actions, device=gs.device, dtype=gs.tc_float) - self.env_cfg["clip_actions"])

    def _move(self, vector, body = False, timesteps = 10):
        '''
        The function to move the gripper along the target displacement vector
        NOTE: to use displacement seems to be better than velocity
        NOTE: only to support one environment currently
        '''
        current_pos = self.franka_gripper.get_pos()
        current_quat = quat_to_xyz(self.franka_gripper.get_quat())

        qpos = torch.cat((current_pos, current_quat), dim=1)
        displacement = torch.zeros((self.num_envs, 6), device=gs.device, dtype=gs.tc_float)
        displacement[:, :2] = vector[:, :2]
        displacement[:, -1] = vector[:, 2]
        qpos += displacement
        if body is True: # Rotation represented w.r.t the body z-axis (body-1-2-3)
            current_pos = self.franka_gripper.get_pos().cpu().numpy()
            current_angle = quat_to_xyz(self.franka_gripper.get_quat()).cpu().numpy()
            qpos = np.zeros(6)
            qpos[0:3] = [current_pos[0] + vector[0], current_pos[1] + vector[1], current_pos[2]]
            target_angle_body_z = current_angle[2] + vector[2]
            qpos[3:6] = [current_angle[0], current_angle[1], target_angle_body_z]
            
        else: # Rotation represented w.r.t the world z-axis (world-1-2-3)
            # TODO: Modify it to support parallelism
            vector = vector[0].cpu().detach().numpy()
            current_pos = self.franka_gripper.get_pos()[0].cpu().numpy()
            current_quat = R.from_quat(self.franka_gripper.get_quat().cpu().numpy(), scalar_first=True).as_matrix()
            new_quat = R.from_quat([np.cos(vector[2]/2), 0, 0, np.sin(vector[2]/2)], scalar_first=True).as_matrix() @ current_quat
            new_quat = R.from_matrix(new_quat).as_quat(scalar_first=True)
            new_quat= quat_to_xyz(new_quat)
            qpos = np.zeros(6)
            qpos[0:3] = [current_pos[0] + vector[0], current_pos[1] + vector[1], current_pos[2]]
            qpos[3:6] = new_quat
        

        self.franka_gripper.control_dofs_position(qpos, self.base_dofs)
        for i in range(timesteps):
            self.scene.step()
            self.cam.render()

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions
        # Execute the actions
        self._move(exec_actions, timesteps=150)

        # update buffers
        self.episode_length_buf += 1

        # gripper poses
        gripper_pose_pos = self.franka_gripper.get_pos()[:, :2]
        gripper_pose_ang = quat_to_xyz(self.franka_gripper.get_quat(), rpy=True)[:, 2]
        gripper_pose = torch.cat((gripper_pose_pos, gripper_pose_ang.view(-1, 1)), dim=1)
        
        self.gripper_pose = gripper_pose.clone()
        
        # object poses
        self.obj_pos[:] = self.cube.get_pos()[:, :2]
        self.obj_ang[:] = (quat_to_xyz(self.cube.get_quat(), rpy=True)[:, 2]).view(-1,1)

        # obj_pose: used for policy prediction & reward calculation
        self.last_obj_pose = self.obj_pose.clone()
        obj_pose = torch.cat((self.obj_pos, self.obj_ang), dim=1)
        self.obj_pose = obj_pose.clone()

        # the relative pose w.r.t. the origin: used for rewarding & resetting
        rel_pose = torch.cat((self.obj_pos, 0.6 * self.obj_ang), dim=1)
        self.last_rel_pose = self.rel_pose.clone()
        self.rel_pose = rel_pose.clone()

        inv_base_quat = inv_quat(self.cube.get_quat())
        self.obj_lin_vel[:] = transform_by_quat(self.cube.get_vel(), inv_base_quat)[:, :2]
        self.obj_ang_vel[:] = (transform_by_quat(self.cube.get_ang(), inv_base_quat)[:, 2]).view(-1,1)

        # resample commands
        # envs_idx = self._at_target()

        # check termination and reset
        dist = torch.sqrt(torch.sum(torch.square(self.rel_pose[:, 0:2]), dim=1)) + torch.norm(self.rel_pose[:, 2])
        self.done_signal = dist < self.env_cfg["at_target_threshold"]
        self.crash_condition = (
            (torch.abs(self.obj_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.obj_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.obj_ang[:, 0]) > self.env_cfg["termination_if_angle_greater_than"])
           
        )

        # if the gripper is closed, it means that it has lost contact with the object
        self.lose_contact = self.franka_gripper.get_qpos()[:, 7] < 0.002
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition | self.lose_contact
        # The "accidents" that require reset

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        '''
        NOTE: the following block establishs the state space as (gripper pose, object pose)
        '''
        # gripper_pose_hist = self.obs_buf[:, :self.obs_length * self.num_actions]
        # gripper_pose_hist = torch.concatenate([gripper_pose_hist, gripper_pose], dim=1)
        # # keep the last obs_length observations
        # gripper_pose_hist = gripper_pose_hist[:, -self.obs_length * self.num_actions:]
        # object_pose_hist = self.obs_buf[:, -self.obs_length * (self.num_obs - self.num_actions):]
        # object_pose_hist = torch.concatenate([object_pose_hist, self.obj_pose], dim=1)
        # # keep the last obs_length observations
        # object_pose_hist = object_pose_hist[:, -self.obs_length * (self.num_obs - self.num_actions):]
        # self.obs_buf = torch.cat([gripper_pose_hist, object_pose_hist], dim=1)
        
        '''
        NOTE: the following block establishs the state space as (object pose)
        '''
        object_pose_hist = self.obs_buf[:, -self.obs_length * self.num_obs:]
        object_pose_hist = torch.concatenate([object_pose_hist, self.obj_pose], dim=1)
        # keep the last obs_length observations
        object_pose_hist = object_pose_hist[:, -self.obs_length * self.num_obs:]
        self.obs_buf = copy.deepcopy(object_pose_hist)
        
        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.done_signal, self.reset_buf, self.extras
    
    def state_transform(self, state, grasp_pose_new):
        # The function to transform the gripper poses to "virtual" ones
        # so that the network can predict better results with the "familiar" states
        T = np.linalg.inv(self.grasp_pose) @ grasp_pose_new
        pass
    def action_transform(self, action, grasp_pose_new):
        # The function to transform the gripper actions predicted from "virtual"
        # gripper poses back to the ones to execute by real gripper
        pass
    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx, cube_pos = None, cube_quat= None):

        '''
        The function to reset the environment
        '''
        if len(envs_idx) == 0:
            return
        # Place the cube at any random pose
        if cube_pos is None and cube_quat is None:
            angle = torch.rand(size=(len(envs_idx), ), device=gs.device) * 2 * torch.pi
            cube_pos = torch.hstack((\
                self.horizon_size * torch.cos(angle).view(-1,1), 
                self.horizon_size * torch.sin(angle).view(-1,1), 
                0.025 * torch.ones((len(envs_idx), 1), device=gs.device))
            )
            cube_angle = torch.zeros((len(envs_idx), 3), device=gs.device)
            cube_angle[:, 2] = torch.rand(size=(len(envs_idx), ), device=gs.device) * np.pi * 2 - np.pi
            cube_quat = xyz_to_quat(cube_angle)

        self.cube.set_pos(cube_pos, envs_idx=envs_idx)
        self.cube.set_quat(cube_quat, envs_idx=envs_idx)

        # Transform the grasp pose to the world frame
        object_pos = self.cube.get_pos(envs_idx=envs_idx)
        object_quat = quat_to_R(self.cube.get_quat(envs_idx=envs_idx))
        object_transform = torch.zeros((len(envs_idx), 4, 4), device=gs.device, dtype=gs.tc_float)
        object_transform[:, :3, :3] = object_quat
        object_transform[:, :3, 3] = object_pos
        grasp_pose_world = object_transform[:] @ \
            torch.tensor(self.grasp_pose, device=gs.device, dtype=gs.tc_float)
        # Move to grasp pose
        grasp_pos = grasp_pose_world[:, :3, 3]
        grasp_quat = R_to_quat(grasp_pose_world[:, :3, :3])        # Move the gripper there
        qpos = torch.zeros((len(envs_idx), 9), device=gs.device, dtype=gs.tc_float)
        qpos[:, 0:3] = grasp_pos
        qpos[:, 3:7] = grasp_quat
        qpos[:, 7:] = 0.04  # gripper open pos

        # At the start, manually set the gripper to the pre-grasp pose
        self.franka_gripper.set_qpos(qpos, envs_idx=envs_idx)

        # Close the gripper
        vpos = torch.zeros((len(envs_idx), 6), device=gs.device)
        self.franka_gripper.control_dofs_velocity(vpos, self.base_dofs, envs_idx=envs_idx)
        gripper_force = torch.zeros((len(envs_idx), 2), device=gs.device, dtype=gs.tc_float)
        gripper_force[:, 0] = -0.9
        gripper_force[:, 1] = -0.9
        self.franka_gripper.control_dofs_force(gripper_force, self.fingers_dof, envs_idx=envs_idx)

        # Allow the process to complete
        for i in range(100):
            self.scene.step()

        # reset base
        self.obj_pos[envs_idx] = self.cube.get_pos(envs_idx=envs_idx)[:, :2]
        self.obj_ang[envs_idx] = (quat_to_xyz(self.cube.get_quat(envs_idx=envs_idx), rpy=True)[:, 2]).view(-1,1)
        
        # obj_pose: used for policy prediction
        obj_pose = torch.cat((self.obj_pos, self.obj_ang), dim=1)
        self.obj_pose[envs_idx] = obj_pose[envs_idx]
        self.last_obj_pose[envs_idx] = obj_pose[envs_idx]

        # gripper_pose: used for policy prediction, also
        gripper_pose_pos = self.franka_gripper.get_pos(envs_idx=envs_idx)[:, :2]
        gripper_pose_ang = (quat_to_xyz(self.franka_gripper.get_quat(envs_idx=envs_idx), rpy=True)[:, 2]).view(-1,1)
        gripper_pose = torch.cat((gripper_pose_pos, gripper_pose_ang), dim=1)
        self.gripper_pose[envs_idx] = gripper_pose.clone()
        
        # the relative pose w.r.t. the origin: used for rewarding & resetting
        rel_pose = torch.cat((self.obj_pos, 0.6 * self.obj_ang), dim=1)
        self.rel_pose =  rel_pose.clone()
        self.last_rel_pose = rel_pose.clone()

        
        
        self.obj_lin_vel[envs_idx] = 0
        self.obj_ang_vel[envs_idx] = 0


        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        '''
        NOTE: The following code block assumes observations are based on 
        both gripper pose & object pose
        '''
        # obj_pose_stack = self.obj_pose[envs_idx].repeat(1, self.obs_length)
        # gripper_pose_stack = self.gripper_pose[envs_idx].repeat(1, self.obs_length)

        # self.obs_buf[envs_idx] = torch.concatenate((gripper_pose_stack, obj_pose_stack), dim = 1)

        '''
        NOTE: The following code block assumes observations on only object pose
        '''
        obj_pose_stack = self.obj_pose[envs_idx].repeat(1, self.obs_length)
        gripper_pose_stack = self.gripper_pose[envs_idx].repeat(1, self.obs_length)

        self.obs_buf[envs_idx] = copy.deepcopy(obj_pose_stack)
        
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0



    def reset(self, cube_pos_lst = [], cube_quat_lst = []):
        self.reset_buf[:] = True
        if len(cube_pos_lst) == 0 and len(cube_quat_lst) == 0:
            self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        else:
            for idx, cube_pos, cube_quat in zip(torch.arange(self.num_envs, device=gs.device), cube_pos_lst, cube_quat_lst):
                self.reset_idx(torch.tensor([idx]), cube_pos=cube_pos, cube_quat=cube_quat)
        return self.obs_buf, None

        
    # ------------ reward functions----------------
    def _reward_target(self):
        # target_rew = torch.sum(torch.square(self.last_rel_pose), dim=1) - torch.sum(torch.square(self.rel_pose), dim=1)
        target_rew = - torch.sqrt(torch.sum(torch.square(self.rel_pose[:, 0:2]), dim=1)) - torch.norm(self.rel_pose[:, 2])
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_stuck(self):
        # The reward to encourage the agent to move out of the current stuck state
        obs_diff = self.obs_buf.view(-1, self.obs_length, self.num_obs)
        obs_diff = torch.diff(obs_diff, dim=1)
        stuck_rew = torch.sum(torch.square(obs_diff), dim=(1, 2))
        return stuck_rew


    def _reward_angular(self):
        angular_rew = torch.norm(self.obj_ang_vel, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew