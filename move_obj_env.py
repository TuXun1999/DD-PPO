import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, xyz_to_quat
from scipy.spatial.transform import Rotation as R
import tkinter as tk
import threading
import time
import pickle
import torch

from denoising_diffusion_pytorch import ConditionalUnet1D, \
    GaussianDiffusion1DConditional, Trainer1DCond, Dataset1DCond

import open3d as o3d
def se2norm(array):
    # Find the norm of a se2 vector; use l2 norm temporarily
    return torch.norm(array[0:2]) + 0.1 * torch.norm(array[2])
class Joystick:
    def __init__(self, master, env):
        self.master = master
        self.canvas = tk.Canvas(master, width=200, height=200, bg="lightgray")
        self.canvas.place(x=175, y=50)

        buttons = {}
        buttons["left"] = [10, 120]
        buttons["right"] = [400, 120]
        buttons["top"] = [250, 10]

        self.frames = {}
        self.labels = {}
        for key, button in buttons.items():
            frame = tk.Frame(
                master=master,
                relief=tk.RAISED,
                borderwidth=1
            )
            x, y = button
            self.frames[key] = frame
            frame.place(x=x, y=y)
            frame.configure(bg="Grey")
            
            if key == "left":
                txt = "Rotate left\n(clockwise)"
            elif key == "right":
                txt = "Rotate right\n(counter-clockwise)"
            elif key == "top":
                txt = "Reset"
            label = tk.Label(master=frame, text=txt, font=("Arial", 16, "bold"))
            label.pack(padx=2, pady=2)
            label.configure(bg="Grey")
            self.labels[key] = label


        self.center_x, self.center_y = 100, 100
        self.base_radius = 80
        self.knob_radius = 20

        self.canvas.create_oval(self.center_x - self.base_radius, self.center_y - self.base_radius,
                                self.center_x + self.base_radius, self.center_y + self.base_radius,
                                fill="gray", outline="darkgray")

        self.knob = self.canvas.create_oval(self.center_x - self.knob_radius, self.center_y - self.knob_radius,
                                            self.center_x + self.knob_radius, self.center_y + self.knob_radius,
                                            fill="blue", outline="darkblue")

        self.knob_x, self.knob_y = self.center_x, self.center_y

        # Optional: Set focus when mouse enters the canvas
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.tag_bind(self.knob, "<Button-1>", self.on_button_press)
        self.canvas.tag_bind(self.knob, "<B1-Motion>", self.on_mouse_drag)
        self.canvas.tag_bind(self.knob, "<ButtonRelease-1>", self.on_button_release)

        self.canvas.bind("<KeyRelease>", self.on_key_release)

        # Bind the button with events
        self.master.bind("<KeyPress-a>", lambda event: self.button_color_change(event, "left"))
        self.master.bind("<KeyRelease-a>", lambda event: self.button_color_restore(event, "left"))
        self.master.bind("<KeyPress-d>", lambda event: self.button_color_change(event, "right"))
        self.master.bind("<KeyRelease-d>", lambda event: self.button_color_restore(event, "right"))
        self.master.bind("<KeyPress-r>", lambda event: self.button_color_change(event, "top"))
        self.master.bind("<KeyRelease-r>", lambda event: self.button_color_restore(event, "top"))
        # The simulation environment linked to this Joystick
        self.env = env
    
    def button_color_change(self, event, option):
        # Change the color of the button 
        self.frames[option].configure(bg="Red")
        self.labels[option].configure(bg="Red")
    def button_color_restore(self, event, option):
        # Restore the color of the button 
        self.frames[option].configure(bg="Grey")
        self.labels[option].configure(bg="Grey")
    def on_button_press(self, event):
        self.start_x = self.knob_x
        self.start_y = self.knob_y

    def on_mouse_drag(self, event):
        new_knob_x = event.x
        new_knob_y = event.y


        # Keep knob within base boundaries
        distance = ((new_knob_x - self.center_x)**2 + (new_knob_y - self.center_y)**2)**0.5
        if distance > self.base_radius - self.knob_radius:
            ratio = (self.base_radius - self.knob_radius) / distance
            new_knob_x = self.center_x + (new_knob_x - self.center_x) * ratio
            new_knob_y = self.center_y + (new_knob_y - self.center_y) * ratio

        self.canvas.coords(self.knob, new_knob_x - self.knob_radius, new_knob_y - self.knob_radius,
                           new_knob_x + self.knob_radius, new_knob_y + self.knob_radius)
        
        self.knob_x = new_knob_x
        self.knob_y = new_knob_y

        # Calculate joystick values (e.g., -1 to 1)
        joystick_x = (self.knob_x - self.center_x) / (self.base_radius - self.knob_radius)
        joystick_y = (self.knob_y - self.center_y) / (self.base_radius - self.knob_radius)
        # print(f"Joystick X: {joystick_x:.2f}, Joystick Y: {joystick_y:.2f}")

        # Export the value to the external env linked to this joystick
        self.env.execute([joystick_x, joystick_y, 0])

    def on_button_release(self, event):
        # Reset knob to center
        self.canvas.coords(self.knob, self.center_x - self.knob_radius, self.center_y - self.knob_radius,
                           self.center_x + self.knob_radius, self.center_y + self.knob_radius)
        self.knob_x, self.knob_y = self.center_x, self.center_y
        # print("Joystick released, reset to center.")
    def on_key_release(self, event):
        if event.keysym =='a':
            self.env.execute([0, 0, 1])
        elif event.keysym == 'd':
            self.env.execute([0, 0, -1])
        elif event.keysym == 'r':
            self.env.reset()
        elif event.keysym == 'q':
            print("Command received. Ready to quit")
            self.env._exit()

class MoveCubeEnv():
    def __init__(self, obj_filename, obj_type, grasp_pose, \
                 horizon_size = 0.3, load_prev = False):
        ''''
        The function to initialize the simulation environment containing the block 
        and the floating gripper
        '''
        ########################## init ##########################
        gs.init(backend=gs.gpu)

        ########################## create a scene ##########################
        self.scene = gs.Scene(
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = (0, 0, 1.5),
                camera_lookat = (0.0, 0.0, 0.0),
                camera_fov    = 45,
                max_FPS       = 60,
            ),
            sim_options = gs.options.SimOptions(
                dt = 0.01,
            ),
            show_viewer = True,
            show_FPS=False,
        )
        # TODO: cast a light to remove the shadow?
        # self.scene.add_light(gs.morphs.Sphere(pos=(0, 0, 1), radius = 0.2,), color=(0, 0, 0),intensity=200000)
        ########################## entities ##########################
        # Entity 1: the ground plane
        plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        plane.set_friction(0.01)
        # cube = scene.add_entity(
        #     gs.morphs.Box(
        #         size = (0.04, 0.04, 0.04),
        #         pos  = (0.65, 0.0, 0.02),
        #     )
        # )

        # Entity 2: the cube
        self.horizon_size = horizon_size

        # Read the height of the lowest point
        mesh = o3d.io.read_triangle_mesh(obj_filename)
        height = -np.min(np.asarray(mesh.vertices)[:, 2])
        self.height = height

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
                            color=(1.0, 1.0, 1.0),
                        ),
                    ),
            # Reduce the weight
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )


        # Entity 3: the target pose of the object
        self.cube_virtual = self.scene.add_entity(
            gs.morphs.Mesh(
                file = obj_filename,
                pos = [0, 0, height],
                quat = [1, 0, 0, 0],
                fixed = True,
                collision = False
            ),
            surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(
                            color=(1.0, 0.5, 0.5),
                        ),
                    )
        )


        # Entity 4: The floating parallel gripper
        self.franka_gripper = self.scene.add_entity(
            gs.morphs.URDF(
                file = "./panda_v2_gripper.urdf",
            ),
            # Ignore gravity for the robot
            material=gs.materials.Rigid(gravity_compensation=0.9),
        )
        ########################## build ##########################
        self.scene.build()

        ##################### Configure #####################
        self.base_dofs = np.arange(6)
        self.fingers_dof = np.arange(6, 8)

        # set control gains
        # Note: the following values are tuned for achieving best behavior with Franka
        # Typically, each new robot would have a different set of parameters.
        # Sometimes high-quality URDF or XML file would also provide this and will be parsed.
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


        ############################ Teleoperation GUI ###################
        self.joystick_root = tk.Tk()
        self.joystick_root.geometry("600x300") 
        self.joystick = Joystick(self.joystick_root, self)
        
        # Maximum speed for the cube
        self.max_v = 0.04
        self.max_angular_v = 10*np.pi/180


        ########################### Poses to Collect #####################
        self.object_poses = []
        self.gripper_poses = []
        self.data_collection_thread = threading.Thread(target = self.collect_poses)
        self.kill_thread = threading.Event()
        self.data_filename = obj_type + ".pkl"
        
        obj_mesh = o3d.io.read_triangle_mesh(obj_filename)
        obj_vertices = np.asarray(obj_mesh.vertices)
        std_x = 0.001 #0.01 * (np.max(obj_vertices[:, 0]) - np.min(obj_vertices[:, 0]))
        std_y = 0.001 #0.01 * (np.max(obj_vertices[:, 1]) - np.min(obj_vertices[:, 1]))
        std_theta = 0.001 #np.pi/180 * 0.1 # 0.1 degree
        self.pose_noise_std = {"x": std_x, "y": std_y, "theta": std_theta}

    def reset(self, collect_data = True):
        '''
        The function to reset the environment
        '''
        # At first, stop the old data collection thread
        if collect_data is True:
            self.kill_thread.set()

        # Place the cube at any random pose
        angle = np.random.rand() * 2 * np.pi
        cube_pos = [\
            self.horizon_size * np.cos(angle), 
            self.horizon_size * np.sin(angle), 
            self.height]
        cube_angle = np.random.rand() * np.pi * 2 - np.pi
        cube_quat = xyz_to_quat(torch.tensor([0, 0, cube_angle]), rpy=True)
        self.cube.set_pos(cube_pos)
        self.cube.set_quat(cube_quat)

        # Transform the grasp pose to the world frame
        object_pos = self.cube.get_pos().cpu().numpy()
        object_quat = self.cube.get_quat().cpu().numpy()
        object_transform = np.eye(4)
        object_transform[:3, :3] = R.from_quat(
            object_quat, scalar_first=True).as_matrix()
        object_transform[:3, 3] = object_pos
        grasp_pose_world = object_transform @ self.grasp_pose
        # Move to grasp pose
        grasp_pos = grasp_pose_world[:3, 3]
        grasp_quat = R.from_matrix(grasp_pose_world[:3, :3]).as_quat(scalar_first=True)
        # Move the gripper there
        qpos = np.zeros(9)
        qpos[0:3] = grasp_pos
        qpos[3:7] = grasp_quat
        qpos[7:] = 0.04  # gripper open pos

        # At the start, manually set the gripper to the pre-grasp pose
        self.franka_gripper.set_qpos(qpos)

        # Close the gripper
        vpos = np.zeros(6)
        self.franka_gripper.control_dofs_velocity(vpos, self.base_dofs) 
        self.franka_gripper.control_dofs_force(np.array([-0.6, -0.6]), self.fingers_dof)

        # Allow the process to complete
        for i in range(200):
            self.scene.step()

        self.in_progress = False

        # Start the new data collection thread
        if collect_data is True:
            self.kill_thread.clear()
            self.data_collection_thread = threading.Thread(target = self.collect_poses)
            self.data_collection_thread.start()

    def teleoperation(self):
        '''
        The function to control the gripper using teleoperation from user
        '''
        # Print helper messages
        print("**** Avoid Long Pressing *****")
        print("[ad]: Rotational movement")

        # Loop the joystick GUI
        self.joystick_root.mainloop()
        

    def execute(self, joystick_vector):
        '''
        The control functions execute the value read from the joystick
        '''
        # Control arm by sending commands
        displacement_vector = [0, 0, 0]
       
        displacement_vector[0] = joystick_vector[0] * self.max_v 
        displacement_vector[1] = -joystick_vector[1] * self.max_v
        displacement_vector[2] = joystick_vector[2] * self.max_angular_v
        if not self.in_progress:
            self._move(np.array(displacement_vector))
    def _move(self, vector, body = False, timesteps = 10):
        '''
        The function to move the gripper along the target displacement vector
        NOTE: to use displacement seems to be better than velocity
        '''
        if np.linalg.norm(vector) != 0:
            self.in_progress = True # Deny the new commands until the current one is completed
            
            if body is True: # Rotation along the body z-axis (body-1-2-3)
                current_pos = self.franka_gripper.get_pos().cpu().numpy()
                current_angle = quat_to_xyz(self.franka_gripper.get_quat()).cpu().numpy()
                qpos = np.zeros(6)
                qpos[0:3] = [current_pos[0] + vector[0], current_pos[1] + vector[1], current_pos[2]]
                target_angle_body_z = current_angle[2] + vector[2]
                qpos[3:6] = [current_angle[0], current_angle[1], target_angle_body_z]
                
            else: # Rotation along the world z-axis (world-1-2-3)
                # NOTE: I didn't add the angles to rpy directly,
                # because the quat_to_xyz formulas are different 
                # in quat_to_xyz and numpy
                current_pos = self.franka_gripper.get_pos().cpu().numpy()
                current_quat = R.from_quat(self.franka_gripper.get_quat().cpu().numpy(), scalar_first=True).as_matrix()
                new_quat = R.from_quat([np.cos(vector[2]/2), 0, 0, np.sin(vector[2]/2)], scalar_first=True).as_matrix() @ current_quat
                new_quat = R.from_matrix(new_quat).as_quat(scalar_first=True)
                new_quat= quat_to_xyz(new_quat)
                qpos = np.zeros(6)
                qpos[0:3] = [current_pos[0] + vector[0], current_pos[1] + vector[1], current_pos[2]]
                qpos[3:6] = new_quat
                
            # Always execute body-1-2-3/ world-3-2-1
            self.franka_gripper.control_dofs_position(qpos, self.base_dofs)
            for i in range(timesteps):
                self.scene.step()
            self.in_progress = False
    def _exit(self):
        '''
        The function to exit the current simulation environment
        '''
        self.kill_thread.set()
        # Store the collected data into a separate file
        data_dict = {}
        data_dict["gripper_poses"] = self.gripper_poses
        data_dict["object_poses"] = self.object_poses
        data_dict["grasp_pose"] = self.grasp_pose
        with open(self.data_filename, "wb") as f:
            pickle.dump(data_dict, f)

        exit(0)
    
    def collect_poses(self):

        gripper_poses_one_demo = []
        object_poses_one_demo = []
        while True and not self.kill_thread.is_set():
            # Obtain the gripper pose
            current_pos = self.franka_gripper.get_pos().cpu().numpy()

            # Euler convention: body-3-2-1 / world-1-2-3
            current_angle = quat_to_xyz(self.franka_gripper.get_quat(), rpy=True).cpu().numpy()
            gripper_pose = [\
                current_pos[0] + np.random.normal(0, self.pose_noise_std["x"]), \
                current_pos[1] + np.random.normal(0, self.pose_noise_std["y"]), \
                current_angle[2] + np.random.normal(0, self.pose_noise_std["theta"])]

            # Obtain the object pose
            current_pos = self.cube.get_pos().cpu().numpy()
            current_angle = quat_to_xyz(self.cube.get_quat(), rpy=True).cpu().numpy()
            object_pose = [\
                current_pos[0] + np.random.normal(0, self.pose_noise_std["x"]), \
                current_pos[1] + np.random.normal(0, self.pose_noise_std["y"]), \
                current_angle[2] + np.random.normal(0, self.pose_noise_std["theta"])]
            
            gripper_poses_one_demo.append(gripper_pose)
            object_poses_one_demo.append(object_pose)

            # Wait for some time so we can drive the robot to a new position.
            time.sleep(0.2)

        # Extend the existing list or Create a new list
        gripper_poses_one_demo = np.array(gripper_poses_one_demo)
        object_poses_one_demo = np.array(object_poses_one_demo)

        self.gripper_poses.append(gripper_poses_one_demo)
        self.object_poses.append(object_poses_one_demo)
    def collect_demo(self):
        

        ########################## Reset ##########################
        self.reset(collect_data=True)

        ########################## Teleoperation ##########################

        self.teleoperation()

    def policy(self, **kwargs):
        self.reset(collect_data=False)
        if kwargs["policy"] == "diffusion":
            self.diffusion_policy(**kwargs)
        else:
            print("Unknown policy type. Exiting")
    
    def diffusion_policy(self, **kwargs):
        '''
        The function to execute the diffusion policy
        '''
        # Load the normalization statistics
        normalization_stats = torch.load(kwargs["normalization_stats"])

        # Hyperparameters
        obs_length = 3
        obs_dim = 3
        seq_length = 4
        action_dim = 3
        
        v_min = normalization_stats["v_min"].cpu()
        v_max = normalization_stats["v_max"].cpu()
        angular_v_max = normalization_stats["angular_v_max"].cpu()
        angular_v_min = normalization_stats["angular_v_min"].cpu()
        training_sq = normalization_stats["training_sq"]
        local_label = normalization_stats["local_label"]
        global_label = normalization_stats["global_label"]
        # Construct the models
        option = 'unet1d'  # 
        if option == "unet1d":
            model = ConditionalUnet1D(
                input_dim = action_dim,
                local_cond_dim = 1,
                global_cond_dim = 2 * obs_length * obs_dim,
            )
        else:
            # model = ConditionalUnet1D(
            #     input_dim = 6,
            #     local_cond_dim = 1,
            #     global_cond_dim = 6,
            # )
            pass # TODO: the transformer

        diffusion = GaussianDiffusion1DConditional(
            model,
            seq_length = seq_length,
            timesteps = 1000,
            objective = 'pred_noise'
        )

        dataset = Dataset1DCond(training_sq, local_label, global_label)
        trainer = Trainer1DCond(
            diffusion,
            dataset = dataset,
            train_batch_size = 32,
            train_lr = 8e-5,
            train_num_steps = 5000,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                       # turn on mixed precision
            save_and_sample_every=100000      # Force not to save the sample result
        )

        # Load the previously trained model
        trainer.load(1)

        # Execute the diffusion policy on the novel task
        batch_size_sample = 1

        # Obtain the gripper pose
        current_pos = self.franka_gripper.get_pos().cpu().numpy()
        current_angle = quat_to_xyz(self.franka_gripper.get_quat(), rpy=True).cpu().numpy()
        gripper_pose = torch.tensor([current_pos[0], current_pos[1], current_angle[2]])

        # Obtain the object pose
        current_pos = self.cube.get_pos().cpu().numpy()
        current_angle = quat_to_xyz(self.cube.get_quat(), rpy=True).cpu().numpy()
        object_pose = torch.tensor([current_pos[0], current_pos[1], current_angle[2]])
        gripper_pose_init = gripper_pose
        gripper_pose_init = torch.concatenate([gripper_pose, gripper_pose, gripper_pose]) # Stack three gripper poses together
        object_pose_init = object_pose
        object_pose_init = object_pose.repeat(obs_length) #torch.concatenate([object_pose, object_pose, object_pose]) # Stack three object poses together
        
        # NOTE: use the object pose & gripper pose
        global_label_sample = torch.tile(torch.concatenate([gripper_pose_init, object_pose_init]), (batch_size_sample, 1)).float() # (2 x obs_length x obs_dim)
        # global_label_sample = torch.tile(object_pose_init, (batch_size_sample, 1)).float()
        local_label_sample = torch.zeros(batch_size_sample, 1, seq_length).float() # This is constant
        steps = 0

        while True:
            steps += 1

            # Sample the sequence
            sampled_seq = diffusion.sample(batch_size = batch_size_sample, \
                    local_cond = local_label_sample, global_cond = global_label_sample)

            traj_recon = torch.mean(sampled_seq, dim = 0)
            traj_recon = traj_recon.to(device='cpu') # DxT
            torch.cuda.synchronize()

            # Extract the normalization statistics
            traj_noisy_max_sel = torch.tensor([v_max, v_max, angular_v_max]).unsqueeze(-1).to(device=traj_recon.device)
            traj_noisy_min_sel = torch.tensor([v_min, v_min, angular_v_min]).unsqueeze(-1).to(device=traj_recon.device)
            traj_recon = traj_recon * (traj_noisy_max_sel - traj_noisy_min_sel) + traj_noisy_min_sel



            # Execute the policy
            print(traj_recon.shape)
            action = torch.mean(traj_recon, dim=1)
            print(action.shape)
            # action = (x, y, theta)
            # x -> translation along world x-axis
            # y -> translation along world y-axis
            # theta -> rotation around world-z axis
            
            # By default, Genesis needs
            # x -> translation algon world x-axis
            # y -> translation along world y-axis
            # theta -> rotation around body-z axis
            self._move(action, timesteps=150)
            # Obtain the gripper pose
            current_pos = self.franka_gripper.get_pos().cpu().numpy()
            current_angle = quat_to_xyz(self.franka_gripper.get_quat().cpu().numpy(), rpy=True)
            current_gripper_pose = torch.tensor([current_pos[0], current_pos[1], current_angle[2]])

            # Obtain the object pose
            current_pos = self.cube.get_pos().cpu().numpy()
            current_angle = quat_to_xyz(self.cube.get_quat().cpu().numpy(), rpy=True)
            current_object_pose = torch.tensor([current_pos[0], current_pos[1], current_angle[2]])
            
            # NOTE: care about the gripper & object pose
            obs_gripper_pose = global_label_sample[0][action_dim : obs_length * action_dim]
            obs_gripper_pose = torch.concatenate((obs_gripper_pose, current_gripper_pose))
            obs_object_pose = global_label_sample[0][(-obs_length + 1) * obs_dim :]
            obs_object_pose = torch.concatenate((obs_object_pose, current_object_pose))
            global_label_sample = torch.concatenate((obs_gripper_pose, obs_object_pose)) #obs_object_pose.clone()

            # Determine whether to exit
            if se2norm(object_pose) < 0.08 or steps > 150:
                print(object_pose)
                break

            # Stack global_label_sample
            global_label_sample = torch.tile(global_label_sample, (batch_size_sample, 1)).float()


# The mesh model of the object
obj_type = "banana"
obj_filename = "./object_models/banana/textured.obj"

# Obtain the grasp pose in the frame of the object
grasp_pose = np.eye(4)
# grasp_pose[:3, :3] = R.from_euler('zyx', [0, -90, 90], degrees=True).as_matrix()

# grasp_pose[:3, 3] = np.array([0.17, 0, 0.09])
grasp_pose[:3, :3] = R.from_euler('zyx', [120, 0, 180], degrees=True).as_matrix()

grasp_pose[:3, 3] = np.array([0.009, 0.051, 0.09])

# Another pose
# grasp_pose = np.array([
#         [  0.0000000,  1.0000000,  0.0000000, 0],
#     [1.0000000, -0.0000000,  0.0000000, 0],
#     [0.0000000, -0.0000000, -1.0000000, 0 ],
#     [0, 0, 0, 1]
#     ])
# grasp_pose[0:3, 3] = np.array([0, 0, 0.1])
env = MoveCubeEnv(obj_filename = obj_filename, obj_type = obj_type, grasp_pose=grasp_pose)
# env.collect_demo()
env.policy(policy="diffusion", normalization_stats="normalization_stats.pth")


