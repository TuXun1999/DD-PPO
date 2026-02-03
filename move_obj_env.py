"""
Data collection environment for object manipulation demos.

This module provides a simulation environment with a floating gripper
for collecting teleoperated demonstrations of object manipulation tasks.
"""
import argparse
import json
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import open3d as o3d
import genesis as gs
from genesis.utils.geom import quat_to_xyz, xyz_to_quat
from scipy.spatial.transform import Rotation as R
import tkinter as tk


@dataclass
class EnvConfig:
    """Configuration for the data collection environment."""
    obj_filename: str
    obj_type: str
    grasp_pose: np.ndarray
    horizon_size: float = 0.3
    max_v: float = 0.04
    max_angular_v: float = 10 * np.pi / 180
    pose_noise_std: Dict[str, float] = field(
        default_factory=lambda: {"x": 0.001, "y": 0.001, "theta": 0.001}
    )


class Joystick:
    """Tkinter-based joystick GUI for teleoperation control."""

    def __init__(self, master: tk.Tk, env: "MoveCubeEnv"):
        self.master = master
        self.env = env
        self._setup_canvas()
        self._setup_buttons()
        self._setup_bindings()

    def _setup_canvas(self):
        """Initialize the joystick canvas."""
        self.canvas = tk.Canvas(self.master, width=200, height=200, bg="lightgray")
        self.canvas.place(x=450, y=150)

        self.center_x, self.center_y = 100, 100
        self.base_radius = 80
        self.knob_radius = 20

        self.canvas.create_oval(
            self.center_x - self.base_radius,
            self.center_y - self.base_radius,
            self.center_x + self.base_radius,
            self.center_y + self.base_radius,
            fill="gray",
            outline="darkgray",
        )

        self.knob = self.canvas.create_oval(
            self.center_x - self.knob_radius,
            self.center_y - self.knob_radius,
            self.center_x + self.knob_radius,
            self.center_y + self.knob_radius,
            fill="blue",
            outline="darkblue",
        )

        self.knob_x, self.knob_y = self.center_x, self.center_y

    def _setup_buttons(self):
        """Setup control button labels."""
        buttons = {
            "left": (100, 200, "Rotate left\n(clockwise)"),
            "right": (700, 200, "Rotate right\n(counter-clockwise)"),
            "top": (450, 10, "Reset"),
        }

        self.frames = {}
        self.labels = {}

        for key, (x, y, txt) in buttons.items():
            frame = tk.Frame(master=self.master, relief=tk.RAISED, borderwidth=1)
            frame.place(x=x, y=y)
            frame.configure(bg="Grey")
            self.frames[key] = frame

            label = tk.Label(master=frame, text=txt, font=("Arial", 16, "bold"))
            label.pack(padx=2, pady=2)
            label.configure(bg="Grey")
            self.labels[key] = label

    def _setup_bindings(self):
        """Setup event bindings for mouse and keyboard."""
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.tag_bind(self.knob, "<Button-1>", self._on_button_press)
        self.canvas.tag_bind(self.knob, "<B1-Motion>", self._on_mouse_drag)
        self.canvas.tag_bind(self.knob, "<ButtonRelease-1>", self._on_button_release)
        self.canvas.bind("<KeyRelease>", self._on_key_release)

        # Button color change bindings
        for key, char in [("left", "a"), ("right", "d"), ("top", "r")]:
            self.master.bind(
                f"<KeyPress-{char}>",
                lambda e, k=key: self._button_color_change(k),
            )
            self.master.bind(
                f"<KeyRelease-{char}>",
                lambda e, k=key: self._button_color_restore(k),
            )

    def _button_color_change(self, option: str):
        self.frames[option].configure(bg="Red")
        self.labels[option].configure(bg="Red")

    def _button_color_restore(self, option: str):
        self.frames[option].configure(bg="Grey")
        self.labels[option].configure(bg="Grey")

    def _on_button_press(self, event):
        self.start_x = self.knob_x
        self.start_y = self.knob_y

    def _on_mouse_drag(self, event):
        new_knob_x, new_knob_y = event.x, event.y

        # Constrain knob within base
        distance = ((new_knob_x - self.center_x) ** 2 + (new_knob_y - self.center_y) ** 2) ** 0.5
        if distance > self.base_radius - self.knob_radius:
            ratio = (self.base_radius - self.knob_radius) / distance
            new_knob_x = self.center_x + (new_knob_x - self.center_x) * ratio
            new_knob_y = self.center_y + (new_knob_y - self.center_y) * ratio

        self.canvas.coords(
            self.knob,
            new_knob_x - self.knob_radius,
            new_knob_y - self.knob_radius,
            new_knob_x + self.knob_radius,
            new_knob_y + self.knob_radius,
        )

        self.knob_x, self.knob_y = new_knob_x, new_knob_y

        # Calculate normalized joystick values (-1 to 1)
        joystick_x = (self.knob_x - self.center_x) / (self.base_radius - self.knob_radius)
        joystick_y = (self.knob_y - self.center_y) / (self.base_radius - self.knob_radius)

        self.env.execute([joystick_x, joystick_y, 0])

    def _on_button_release(self, event):
        # Reset knob to center
        self.canvas.coords(
            self.knob,
            self.center_x - self.knob_radius,
            self.center_y - self.knob_radius,
            self.center_x + self.knob_radius,
            self.center_y + self.knob_radius,
        )
        self.knob_x, self.knob_y = self.center_x, self.center_y

    def _on_key_release(self, event):
        if event.keysym == "a":
            self.env.execute([0, 0, 1])
        elif event.keysym == "d":
            self.env.execute([0, 0, -1])
        elif event.keysym == "r":
            self.env.reset()
        elif event.keysym == "q":
            print("Command received. Exiting...")
            self.env.exit()


class MoveCubeEnv:
    """
    Simulation environment for object manipulation data collection.

    Provides a floating parallel gripper that can be controlled via
    teleoperation to collect demonstration trajectories.
    """

    def __init__(self, config: EnvConfig):
        """
        Initialize the simulation environment.

        Args:
            config: Environment configuration
        """
        self.config = config
        self._init_genesis()
        self._init_scene()
        self._init_entities()
        self._init_gripper_control()
        self._init_gui()
        self._init_data_collection()

    def _init_genesis(self):
        """Initialize Genesis backend."""
        gs.init(backend=gs.gpu)

    def _init_scene(self):
        """Create and configure the simulation scene."""
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, 0, 1.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=45,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            show_viewer=True,
            show_FPS=False,
        )

    def _init_entities(self):
        """Add all entities to the scene."""
        # Ground plane
        plane = self.scene.add_entity(gs.morphs.Plane())
        plane.set_friction(0.01)

        # Load object mesh to get height
        mesh = o3d.io.read_triangle_mesh(self.config.obj_filename)
        self.height = -np.min(np.asarray(mesh.vertices)[:, 2])

        # Target object
        self.cube = self.scene.add_entity(
            gs.morphs.Mesh(
                file=self.config.obj_filename,
                pos=[0, 0, self.height],
                quat=[1, 0, 0, 0],
                convexify=False,
                decompose_nonconvex=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(1.0, 1.0, 1.0))
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        # Virtual target pose indicator
        self.cube_virtual = self.scene.add_entity(
            gs.morphs.Mesh(
                file=self.config.obj_filename,
                pos=[0, 0, self.height],
                quat=[1, 0, 0, 0],
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5))
            ),
        )

        # Floating parallel gripper
        self.franka_gripper = self.scene.add_entity(
            gs.morphs.URDF(file="./panda_v2_gripper.urdf"),
            material=gs.materials.Rigid(gravity_compensation=0.9),
        )

        # Cameras
        self.cam1 = self.scene.add_camera(
            res=(640, 480),
            pos=(3.5, 0.0, 2.5),
            lookat=(0, 0, 0.5),
            fov=30,
            GUI=True,
        )

        self.cam2 = self.scene.add_camera(
            res=(640, 480),
            pos=(0.0, 0.0, 1.5),
            lookat=(0, 0, 0),
            fov=30,
            GUI=True,
        )

        self.scene.build()

    def _init_gripper_control(self):
        """Configure gripper control parameters."""
        self.base_dofs = np.arange(6)
        self.fingers_dof = np.arange(6, 8)

        self.franka_gripper.set_dofs_kp(np.array([40, 40, 40, 40, 40, 40, 40, 40]))
        self.franka_gripper.set_dofs_kv(np.array([20, 20, 20, 20, 20, 20, 10, 10]))
        self.franka_gripper.set_dofs_force_range(
            np.array([-87, -87, -87, -80, -80, -80, -100, -100]),
            np.array([87, 87, 87, 80, 80, 80, 100, 100]),
        )

        self.end_effector = self.franka_gripper.get_link("panda_hand")
        self.in_progress = False

    def _init_gui(self):
        """Initialize the teleoperation GUI."""
        self.joystick_root = tk.Tk()
        self.joystick_root.geometry("1000x500")
        self.joystick = Joystick(self.joystick_root, self)

    def _init_data_collection(self):
        """Initialize data collection structures."""
        self.object_poses: List[np.ndarray] = []
        self.gripper_poses: List[np.ndarray] = []
        self.img_data: Dict[str, np.ndarray] = {}
        self.kill_thread = threading.Event()
        self.data_collection_thread: Optional[threading.Thread] = None

    def reset(self, collect_data: bool = True):
        """
        Reset the environment to a new random configuration.

        Args:
            collect_data: Whether to start data collection after reset
        """
        if collect_data:
            self.kill_thread.set()

        # Random object placement
        angle = np.random.rand() * 2 * np.pi
        cube_pos = [
            self.config.horizon_size * np.cos(angle),
            self.config.horizon_size * np.sin(angle),
            self.height,
        ]
        cube_angle = np.random.rand() * np.pi * 2 - np.pi
        cube_quat = xyz_to_quat(np.array([0, 0, cube_angle]), rpy=True)

        self.cube.set_pos(cube_pos)
        self.cube.set_quat(cube_quat)

        # Transform grasp pose to world frame
        object_pos = self.cube.get_pos().cpu().numpy()
        object_quat = self.cube.get_quat().cpu().numpy()
        object_transform = np.eye(4)
        object_transform[:3, :3] = R.from_quat(object_quat, scalar_first=True).as_matrix()
        object_transform[:3, 3] = object_pos

        grasp_pose_world = object_transform @ self.config.grasp_pose
        grasp_pos = grasp_pose_world[:3, 3]
        grasp_quat = R.from_matrix(grasp_pose_world[:3, :3]).as_quat(scalar_first=True)

        # Set gripper to grasp pose
        qpos = np.zeros(9)
        qpos[0:3] = grasp_pos
        qpos[3:7] = grasp_quat
        qpos[7:] = 0.04  # Gripper open
        self.franka_gripper.set_qpos(qpos)

        # Close gripper
        self.franka_gripper.control_dofs_velocity(np.zeros(6), self.base_dofs)
        self.franka_gripper.control_dofs_force(np.array([-0.6, -0.6]), self.fingers_dof)

        for _ in range(200):
            self.scene.step()

        self.in_progress = False

        if collect_data:
            self.kill_thread.clear()
            self.data_collection_thread = threading.Thread(target=self._collect_poses)
            self.data_collection_thread.start()

    def execute(self, joystick_vector: List[float]):
        """
        Execute a control command from the joystick.

        Args:
            joystick_vector: [x, y, rotation] normalized values
        """
        displacement = [
            joystick_vector[0] * self.config.max_v,
            -joystick_vector[1] * self.config.max_v,
            joystick_vector[2] * self.config.max_angular_v,
        ]
        if not self.in_progress:
            self._move(np.array(displacement))

    def _move(self, vector: np.ndarray, timesteps: int = 10):
        """
        Move the gripper by the specified displacement.

        Args:
            vector: [dx, dy, dtheta] displacement vector
            timesteps: Number of simulation steps
        """
        if np.linalg.norm(vector) == 0:
            return

        self.in_progress = True

        current_pos = self.franka_gripper.get_pos().cpu().numpy()
        current_quat = R.from_quat(
            self.franka_gripper.get_quat().cpu().numpy(), scalar_first=True
        ).as_matrix()

        # Apply rotation around world z-axis
        rot_z = R.from_quat(
            [np.cos(vector[2] / 2), 0, 0, np.sin(vector[2] / 2)], scalar_first=True
        ).as_matrix()
        new_quat = rot_z @ current_quat
        new_euler = quat_to_xyz(R.from_matrix(new_quat).as_quat(scalar_first=True))

        qpos = np.zeros(6)
        qpos[0:3] = [current_pos[0] + vector[0], current_pos[1] + vector[1], current_pos[2]]
        qpos[3:6] = new_euler

        self.franka_gripper.control_dofs_position(qpos, self.base_dofs)
        for _ in range(timesteps):
            self.scene.step()

        self.in_progress = False

    def _collect_poses(self):
        """Background thread for collecting pose and image data."""
        gripper_poses_demo = []
        object_poses_demo = []
        images_demo = {"forward_view": [], "overhead_view": []}
        noise_std = self.config.pose_noise_std

        while not self.kill_thread.is_set():
            # Capture images
            rgb_1, _, _, _ = self.cam1.render(rgb=False, depth=False, segmentation=False, normal=False)
            rgb_2, _, _, _ = self.cam2.render(rgb=True, depth=False, segmentation=False, normal=False)
            images_demo["forward_view"].append(rgb_1)
            images_demo["overhead_view"].append(rgb_2)

            # Record gripper pose with noise
            gripper_pos = self.franka_gripper.get_pos().cpu().numpy()
            gripper_angle = quat_to_xyz(self.franka_gripper.get_quat(), rpy=True).cpu().numpy()
            gripper_pose = [
                gripper_pos[0] + np.random.normal(0, noise_std["x"]),
                gripper_pos[1] + np.random.normal(0, noise_std["y"]),
                gripper_angle[2] + np.random.normal(0, noise_std["theta"]),
            ]

            # Record object pose with noise
            object_pos = self.cube.get_pos().cpu().numpy()
            object_angle = quat_to_xyz(self.cube.get_quat(), rpy=True).cpu().numpy()
            object_pose = [
                object_pos[0] + np.random.normal(0, noise_std["x"]),
                object_pos[1] + np.random.normal(0, noise_std["y"]),
                object_angle[2] + np.random.normal(0, noise_std["theta"]),
            ]

            gripper_poses_demo.append(gripper_pose)
            object_poses_demo.append(object_pose)

            time.sleep(0.2)

        # Convert to arrays and store
        self.gripper_poses.append(np.array(gripper_poses_demo))
        self.object_poses.append(np.array(object_poses_demo))
        self.img_data = {
            "forward_view": np.array(images_demo["forward_view"]),
            "overhead_view": np.array(images_demo["overhead_view"]),
        }

        print(f"Collected demo: gripper_poses={self.gripper_poses[-1].shape}, "
              f"object_poses={self.object_poses[-1].shape}")

    def save_data(self, filename: str):
        """
        Save collected demonstration data to file.

        Args:
            filename: Output pickle file path
        """
        data = {
            "gripper_poses": self.gripper_poses,
            "object_poses": self.object_poses,
            "grasp_pose": self.config.grasp_pose,
            "img_data": self.img_data,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")

    def exit(self):
        """Exit the environment and save data."""
        self.kill_thread.set()
        time.sleep(1)
        self.save_data(f"{self.config.obj_type}.pkl")
        self.joystick_root.quit()

    def run_teleoperation(self):
        """Start the teleoperation interface for data collection."""
        print("**** Teleoperation Controls ****")
        print("  Mouse drag: XY movement")
        print("  [a/d]: Rotate left/right")
        print("  [r]: Reset environment")
        print("  [q]: Quit and save data")
        print("********************************")

        self.reset(collect_data=True)
        self.joystick_root.mainloop()


def load_grasp_pose(grasp_pose_path: str) -> np.ndarray:
    """Load grasp pose from JSON file."""
    with open(grasp_pose_path, "r") as f:
        data = json.load(f)
    grasp_pose = np.eye(4)
    grasp_pose[:3, :3] = np.array(data["rotation"])
    grasp_pose[:3, 3] = np.array(data["translation"])
    return grasp_pose


def main():
    parser = argparse.ArgumentParser(description="Collect manipulation demonstrations")
    parser.add_argument("-o", "--object", default="banana", help="Object type name")
    parser.add_argument(
        "-m", "--mesh",
        default="./object_models/banana/textured.obj",
        help="Path to object mesh file"
    )
    parser.add_argument(
        "-g", "--grasp-pose",
        default=None,
        help="Path to grasp pose JSON file (optional)"
    )
    parser.add_argument(
        "--horizon-size",
        type=float,
        default=0.3,
        help="Radius of random object placement"
    )
    args = parser.parse_args()

    # Load or create default grasp pose
    if args.grasp_pose and Path(args.grasp_pose).exists():
        grasp_pose = load_grasp_pose(args.grasp_pose)
    else:
        # Default grasp pose
        grasp_pose = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ])

    config = EnvConfig(
        obj_filename=args.mesh,
        obj_type=args.object,
        grasp_pose=grasp_pose,
        horizon_size=args.horizon_size,
    )

    env = MoveCubeEnv(config)
    env.run_teleoperation()


if __name__ == "__main__":
    main()
