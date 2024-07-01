# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import numpy as np
from numpy.linalg import eig
import torch
from gym import spaces

from scipy.spatial import ConvexHull
import gymnasium as gym # noqa: F401 E261


from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage  # noqa: E501
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path

from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse
from omni.isaac.core.utils.torch.rotations import tensor_clamp

from omni.isaac.core.objects import DynamicCylinder
from omni.isaac.core.prims import ClothPrimView, RigidPrimView, ClothPrim, ParticleSystem  # noqa: E501
from omni.isaac.core.materials import ParticleMaterial


from omniisaacgymenvs.tasks.base.rl_task import RLTask
from robots.articulations.franka import Franka
from robots.articulations.views.franka_view import FrankaView
from robots.articulations.views.second_franka_view import SecondFrankaView

from pxr import Usd, UsdGeom  # noqa: F401


class BagFrankaVision(RLTask):
    """
    Environment Class for Cloth bag opening and lifting environment.
    """
    def __init__(self, name, env, sim_config, offset=None) -> None:
        self.update_config(sim_config)
        self.dt = 1/60
        self._num_observations = 3*64*64
        self.observation_space = spaces.Box(
            np.ones((3, self.camera_width, self.camera_height),
                    dtype=np.float32) * -np.Inf,
            np.ones((3, self.camera_width, self.camera_height),
                    dtype=np.float32) * np.Inf)
        # self.camera_height * 3
        self._num_actions = 9 * 2
        self._ball_radius = 0.25
        
        super().__init__(name, env)
        
        # self.observation_space = spaces.Box(
        #     np.ones((3, self.camera_width, self.camera_height), dtype=np.float32) * -np.Inf,
        #     np.ones((3, self.camera_width, self.camera_height), dtype=np.float32) * np.Inf)
        self.act1 = torch.zeros((self.num_envs, 6), device=self.device)
        self.act2 = torch.zeros((self.num_envs, 6), device=self.device)

        self.rim_index = [0,    3,    5,    7,    9,   11,   13,   14,   16,
                          18,   20,   22, 24,   26,   28,   29,   31,   33,
                          34,   37,   39,   40,   41,   42, 47,   48,   49,
                          50,   55,   56,   57,   58,   63,   64,   65,   66,
                          70,   71,   72,   73]

        self.mid_rim = [224,  233,  240,  247,  254,  261,  268,  273,  280,
                        287,  294,  301, 309,  318,  325,  332,  340,  347,
                        352,  361,  368,  375,  383,  391, 459,  460,  461,
                        462,  487,  488,  489,  490,  563,  564,  565,  566,
                        591,  592,  593,  594]
        self.max_area = -1
        self.pos_reach1 = torch.tensor([[2.8707, 5.0185, 0.8201],
                                        [2.5769, 4.6079, 1.2965]],
                                       device=self.device)
        self.pos_reach2 = torch.tensor([[2.6438, 5.5950, 1.524],
                                        [2.4078, 5.9541, 1.6887]],
                                       device=self.device)
        self.franka_def_pos = torch.tensor([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]], device=self.device)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.camera_type = self._task_cfg["env"].get("cameraType", 'rgb')
        self.camera_width = self._task_cfg["env"]["cameraWidth"]
        self.camera_height = self._task_cfg["env"]["cameraHeight"]

        self.camera_channels = 3
        self._export_images = self._task_cfg["env"]["exportImages"]

        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = (self._task_cfg["env"]
                                          ["fingerCloseRewardScale"])
        self.action_scale = self._task_cfg["env"]["actionScale"]

    def cleanup(self) -> None:
        # initialize remaining buffers
        RLTask.cleanup(self)

        # override observation buffer for camera data
        self.obs_buf = torch.zeros(
            (self.num_envs, 80, 3), device=self.device, dtype=torch.float)
        
    def add_camera(self) -> None:
        stage = get_current_stage()
        camera_path = f"/World/envs/env_0/Camera"
        camera_xform = stage.DefinePrim(f'{camera_path}_Xform', 'Xform')
        # set up transforms for parent and camera prims
        position = (-1.5, 0.0, 2.5)
        rotation = (-90.0, -60, -180)
        UsdGeom.Xformable(camera_xform).AddTranslateOp()
        UsdGeom.Xformable(camera_xform).AddRotateXYZOp()
        camera_xform.GetAttribute('xformOp:translate').Set(position)
        camera_xform.GetAttribute('xformOp:rotateXYZ').Set(rotation)
        camera = stage.DefinePrim(f'{camera_path}_Xform/Camera', 'Camera')
        UsdGeom.Xformable(camera).AddRotateXYZOp()
        camera.GetAttribute("xformOp:rotateXYZ").Set((90, 0, 90))
        # set camera properties
        camera.GetAttribute('focalLength').Set(24)
        camera.GetAttribute('focusDistance').Set(400)
        # hide other environments in the background
        camera.GetAttribute("clippingRange").Set((0.01, 20.0))
        
    def set_up_scene(self, scene) -> None:
        self.stage = get_current_stage()
        self.assets_root_path = get_assets_root_path()

        self.get_cloth_bag()
        self.get_franka()
        self.add_camera()
        super().set_up_scene(scene=scene, replicate_physics=False)
        
        # start replicator to capture image data
        self.rep.orchestrator._orchestrator._is_started = True

        # set up cameras
        # self.render_products = []
        # env_pos = self._env_pos.cpu()
        # print(self._num_envs)
        # for i in range(self._num_envs):
        #     camera = self.rep.create.camera(
        #         position=(-4.2 + env_pos[i][0], env_pos[i][1], 10.0), look_at=(env_pos[i][0], env_pos[i][1], 2.55))
        #     render_product = self.rep.create.render_product(camera, resolution=(self.camera_width, self.camera_height))
        #     self.render_products.append(render_product)
        # print(self.render_products)
        
        # set up cameras
        self.render_products = []
        env_pos = self._env_pos.cpu()
        camera_paths = [f"/World/envs/env_{i}/Camera_Xform/Camera" for i in range(self._num_envs)]
        for i in range(self._num_envs):
            render_product = self.rep.create.render_product(camera_paths[i], resolution=(self.camera_width, self.camera_height))
            self.render_products.append(render_product)
        # initialize pytorch writer for vectorized collection
        self.pytorch_listener = self.PytorchListener()
        # print(self.pytorch_listener)
        self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda")
        self.pytorch_writer.attach(self.render_products)
        
        # TODO: Fix prim path.
        self.bag = ClothPrimView(
            # prim_paths_expr="/World/envs/.*/baghandle/Cube_0_1_021_Cube_136",
            prim_paths_expr="/World/envs/.*/bag_rim_color/sth3/cube_bag/cube_bag",  # noqa: E501
            name="cloth_bag",
            )

        self.cylinder = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Cylinder",
            name="cylinder", reset_xform_properties=False
            )
        self.franka = FrankaView(
            prim_paths_expr="/World/envs/.*/franka",
            name="franka1"
            )

        self.franka2 = SecondFrankaView(
            prim_paths_expr="/World/envs/.*/franka2",
            name="franka2"
            )
        scene.add(self.bag)
        scene.add(self.cylinder)
        scene.add(self.franka)
        scene.add(self.franka2)

        self.init_data()
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("cloth"):
            scene.remove_object("cloth", registry_only=True)
        if scene.object_exists("fancy_sphere1"):
            scene.remove_object("fancy_sphere1", registry_only=True)
        if scene.object_exists("fancy_sphere2"):
            scene.remove_object("fancy_sphere2", registry_only=True)
        if scene.object_exists("franka1"):
            scene.remove_object("franka1", registry_only=True)
        if scene.object_exists("franka2"):
            scene.remove_object("franka2", registry_only=True)

        self.bag = ClothPrimView(
            # prim_paths_expr="/World/envs/.*/baghandle/Cube_0_1_021_Cube_136",
            prim_paths_expr="/World/envs/.*/bag_rim_color/sth3/cube_bag/cube_bag",  # noqa: E501
            name="cloth_bag",
            reset_xform_properties=False
            )
        self.cylinder = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Cylinder",
            name="cylinder",
            reset_xform_properties=False
            )
        self.franka = FrankaView(
            prim_paths_expr="/World/envs/.*/franka",
            name="franka1"
            )

        self.franka2 = SecondFrankaView(
            prim_paths_expr="/World/envs/.*/franka2",
            name="franka2"
            )

        scene.add(self.bag)
        scene.add(self.cylinder)
        scene.add(self.franka)
        scene.add(self.franka2)

        self.init_data()

    def get_franka(self):
        print(self.default_zero_env_path + "/FrankaGripper")
        franka1 = Franka(
            prim_path=self.default_zero_env_path + "/franka",
            translation=(0.0, -0.5, 0.0),
            # orientation=(0.0, 1.0, 0.0, 0.0)
            )
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(franka1.prim_path),
            self._sim_config.parse_actor_config("franka")
            )

        franka2 = Franka(
            prim_path=self.default_zero_env_path + "/franka2",
            translation=(0.0, 0.5, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0)
            )
        self._sim_config.apply_articulation_settings(
            "franka2", get_prim_at_path(franka2.prim_path),
            self._sim_config.parse_actor_config("franka")
            )
        # franka.disable_gravity()

    def get_cloth_bag(self):
        """
        Gets the cloth bag from USD to be added to the stage.
        """
        # TODO: Make this a class to get cloth bag and pass the bag parameters
        # through cfg file.

        # _usd_path = '/home/irobotics/thesis/summer-project/sth3.usd'
        # _usd_path = "usd_files/bag_w_attach_on_floor_w_cylinder.usd"
        # _usd_path = "usd_files/withoutcubesandsmall.usd"
        _usd_path = "usd_files/bag_color_franka.usd"

        mesh_path = self.default_zero_env_path
        add_reference_to_stage(_usd_path, mesh_path)

        particle_system_path = "/World/Materials/particleSystem"
        particle_material_path = "/World/Materials/particleMaterial"
        # # TODO
        _particle_material = ParticleMaterial(
            prim_path=particle_material_path,
            drag=0.1,
            lift=0.1,
            friction=0.6,
        )
        radius = 0.0015  # * 20  # noqa: F841
        restOffset = 0.01  # radius
        contactOffset = 0.0045  # restOffset * 1.5 #0.45 #restOffset * 1.5

        # cloth prim properties
        stretch_stiffness = 100
        bend_stiffness = 10
        shear_stiffness = 10
        spring_damping = 0.2
        particle_mass = 1/1140
        # TODO: Change to dynamic allocation by using mass/number of vertices.

        _particle_system = ParticleSystem(
            prim_path=particle_system_path,
            simulation_owner="/physicsScene",
            rest_offset=restOffset,
            contact_offset=contactOffset,
            solid_rest_offset=restOffset,
            fluid_rest_offset=0.0,
            particle_contact_offset=contactOffset,
            solver_position_iteration_count=56,
            max_neighborhood=96,
            # radius=radius
        )

        ClothPrim(
                prim_path=mesh_path + '/bag_rim_color/sth3/cube_bag/cube_bag',
                particle_system=_particle_system,
                particle_material=_particle_material,
                stretch_stiffness=stretch_stiffness,
                bend_stiffness=bend_stiffness,
                shear_stiffness=shear_stiffness,
                spring_damping=spring_damping,
                particle_mass=particle_mass,
                visible=True,
            )

    def add_cylinder(self):
        """
        Gets the Cylinder Floating in the air to be added to the stage.
        """
        DynamicCylinder(
            prim_path=self.default_zero_env_path + "/Cylinder",
            radius=0.5,
            height=1.0,
            color=np.array([1.0, 0.0, 0.0]),
            mass=1.0
            )

    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device,
                                dtype=torch.float)

        stage = get_current_stage()
        hand_pose1 = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(
                stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")),
            self._device,
        )
        lfinger_pose1 = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(
                stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger"
                                    )),
            self._device,
        )
        rfinger_pose1 = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(
                stage.GetPrimAtPath(
                    "/World/envs/env_0/franka/panda_rightfinger"
                                    )),
            self._device,
        )

        hand_pose2 = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath(
                "/World/envs/env_0/franka2/panda_link7")),
            self._device,
        )
        lfinger_pose2 = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath(
                "/World/envs/env_0/franka2/panda_leftfinger")),
            self._device,
        )
        rfinger_pose2 = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath(
                "/World/envs/env_0/franka2/panda_rightfinger")),
            self._device,
        )

        finger_pose1 = torch.zeros(7, device=self._device)
        finger_pose1[0:3] = (lfinger_pose1[0:3] + rfinger_pose1[0:3]) / 2.0
        finger_pose1[3:7] = lfinger_pose1[3:7]
        hand_pose_inv_rot1, hand_pose_inv_pos1 = tf_inverse(hand_pose1[3:7],
                                                            hand_pose1[0:3])

        finger_pose2 = torch.zeros(7, device=self._device)
        finger_pose2[0:3] = (lfinger_pose2[0:3] + rfinger_pose2[0:3]) / 2.0
        finger_pose2[3:7] = lfinger_pose2[3:7]
        hand_pose_inv_rot2, hand_pose_inv_pos2 = tf_inverse(hand_pose2[3:7],
                                                            hand_pose2[0:3])

        hand_pose_inv_pos = torch.vstack((hand_pose_inv_pos1,
                                          hand_pose_inv_pos2))
        hand_pose_inv_rot = torch.vstack((hand_pose_inv_rot1,
                                          hand_pose_inv_rot2))
        finger_pose = torch.vstack((finger_pose1, finger_pose2))
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[:, 3:7],
            finger_pose[:, 0:3]
        )
        franka_local_pose_pos += torch.tensor([[0, 0.04, 0],
                                               [0, 0.04, 0]],
                                              device=self._device)
        self.franka_local_grasp_pos = (franka_local_pose_pos.unsqueeze(0)
                                       .repeat((self._num_envs, 1, 1)))
        self.franka_local_grasp_rot = (franka_local_grasp_pose_rot.unsqueeze(0)
                                       .repeat((self._num_envs, 1, 1)))

        self.gripper_forward_axis = torch.tensor([0, 0, 1],
                                                 device=self._device,
                                                 dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0],
                                            device=self._device,
                                            dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.franka_default_dof_pos = torch.tensor(
            [[1.157, -1.066, -0.155, -2.239, -1.841, 1.003,
              0.469, 0.035, 0.035],
             [1.157, -1.066, -0.155, -2.239, -1.841, 1.003,
              0.469, 0.035, 0.035]], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, 2, 9), device=self._device)

    def compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
    ):

        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos

    def compute_franka_reward(
        self,
        reset_buf,
        progress_buf,
        actions,
        franka_grasp_pos,
        rim_pos,
        dist_reward_scale,
        action_penalty_scale,
        joint_positions,
        finger_close_reward_scale,
    ):
        # distance from hand to the closest point on rim.
        d1 = torch.norm(
            rim_pos - franka_grasp_pos[:, 0, :].view(self.num_envs, 1, -1),
            p=2, dim=2).min(dim=1).values
        d2 = torch.norm(
            rim_pos - franka_grasp_pos[:, 1, :].view(self.num_envs, 1, -1),
            p=2, dim=2).min(dim=1).values
        rot_reward = torch.zeros(self.num_envs)
        finger_close_reward1 = torch.zeros_like(rot_reward).to(self._device)
        finger_close_reward1 = torch.where(
            d1 <= 0.03,
            (0.04 - joint_positions[:, 0, 7]) + (
                0.04 - joint_positions[:, 0, 8]),
            finger_close_reward1
        )

        finger_close_reward2 = torch.zeros_like(rot_reward).to(self._device)
        finger_close_reward2 = torch.where(
            d2 <= 0.03,
            (0.04 - joint_positions[:, 1, 7]) + (
                0.04 - joint_positions[:, 1, 8]), finger_close_reward2
        )
        
        mid_point = torch.mean(rim_pos, dim=1)
        cylinder_pos, _ = self.cylinder.get_world_poses()
        
        d_cyl_rim = torch.norm(cylinder_pos - mid_point, dim=1)
        reward_lift_bag = 1 / (
            1 + 9*(d_cyl_rim ** 2))
        dist_reward1 = 1.0 / (1.0 + 9*d1**2) - 0.1333
        # dist_reward1 *= dist_reward1
        dist_reward1 = torch.where(d1 <= 0.02, dist_reward1 * 2, dist_reward1)

        dist_reward2 = 1.0 / (1.0 + 9*d2**2) - 0.1333
        # dist_reward2 *= dist_reward2
        dist_reward2 = torch.where(d2 <= 0.02, dist_reward2 * 2, dist_reward2)
        reward_lift_bag = torch.where(d_cyl_rim <= 0.02, dist_reward2 * 2, dist_reward2)
        # print(dist_reward1, dist_reward2)
        
        # print(dist_reward1)
        

        # regularization on the actions (summed for each environment)
        # TODO: fix the dimension issue
        action_penalty = torch.sum(actions.view(self.num_envs, -1)**2, dim=-1)
        # print(dist_reward1, dist_reward2)
        # print(franka_grasp_pos[:, :, :].view(self.num_envs, 2, -1))
        # print(franka_grasp_pos[:, 1, :].view(self.num_envs, 2, -1))

        rewards = (
            dist_reward_scale * dist_reward1
            + dist_reward_scale * dist_reward2
            - action_penalty_scale * action_penalty
            + finger_close_reward1 * finger_close_reward_scale
            + finger_close_reward2 * finger_close_reward_scale
            + reward_lift_bag
        )
        # print(+ finger_close_reward1 * finger_close_reward_scale
        #     + finger_close_reward2 * finger_close_reward_scale)
        # print(rewards, reward_lift_bag)
        return (rewards)

    def get_observations(self) -> dict:
        # self.obs_buf = self.bag.get_world_positions()[:, self.rim_index + self.mid_rim, :].clone().view(self.num_envs, -1)
        hand_pos1, hand_rot1 = self.franka._hands.get_world_poses(clone=False)
        franka_dof_pos1 = self.franka.get_joint_positions(clone=False)
        hand_pos2, hand_rot2 = self.franka2._hands.get_world_poses(clone=False)
        franka_dof_pos2 = self.franka2.get_joint_positions(clone=False)

        franka_dof_pos = torch.hstack((franka_dof_pos1.view(-1, 1, 9),
                                       franka_dof_pos2.view(-1, 1, 9)))
        hand_pos = torch.hstack((hand_pos1.view(-1, 1, 3),
                                 hand_pos2.view(-1, 1, 3)))
        hand_rot = torch.hstack((hand_rot1.view(-1, 1, 4),
                                 hand_rot2.view(-1, 1, 4)))

        self.franka_dof_pos = franka_dof_pos  # - self._env_pos
        self.bag_area = None
        observations = {

        }
        (
            self.franka_grasp_rot,
            self.franka_grasp_pos
        ) = self.compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos
        )
        # print(self.franka_grasp_pos.shape, self._env_pos.unsqueeze(dim=1).shape)
        # print(self._env_pos)
        # print(hand_pos1[0], hand_pos2[0])
        # self.franka_grasp_pos -= self.franka_def_pos.unsqueeze(dim=0)
        (
            self.franka_lfinger_pos1,
            self.franka_lfinger_rot1
        ) = self.franka._lfingers.get_world_poses(clone=False)
        (
            self.franka_rfinger_pos1,
            self.franka_rfinger_rot1
        ) = self.franka._lfingers.get_world_poses(clone=False)

        (self.franka_lfinger_pos2, self.franka_lfinger_rot2
         ) = self.franka2._lfingers.get_world_poses(clone=False)
        (self.franka_rfinger_pos2, self.franka_rfinger_rot2
         ) = self.franka2._lfingers.get_world_poses(clone=False)

        self.franka_rfinger_pos = torch.hstack(
            (self.franka_rfinger_pos1.view(-1, 1, 3),
             self.franka_rfinger_pos2.view(-1, 1, 3)))
        self.franka_lfinger_pos = torch.hstack(
            (self.franka_lfinger_pos1.view(-1, 1, 3),
             self.franka_lfinger_pos2.view(-1, 1, 3)))

        self.rim_pos = (self.bag.get_world_positions()[:, self.rim_index, :]
                        .clone())
        bag_obs = (self.bag.get_world_positions()
                   [:, self.rim_index + self.mid_rim, :]
                   .clone().view(self.num_envs, -1))
        # self.obs_buf = torch.concat(
        #     (bag_obs, self.franka_dof_pos.view(self.num_envs, -1)), dim=-1)
        images = self.pytorch_listener.get_rgb_data()
        if images is not None:
            # if self._export_images:
            #     from torchvision.utils import save_image, make_grid
            #     img = images/255
                # save_image(make_grid(img, nrows = 2), 'cartpole_export.png')
            # from torchvision.utils import save_image, make_grid
            # img = images/255
            # save_image(make_grid(img, nrows = 2), 'cartpole_export.png')
            torch.save(images, "images.pt")
            resize_img = images.clone().float() / 255.0
            self.obs_buf = resize_img.reshape(self.num_envs, -1)
            # self.obs_buf = images.clone().float()/255.0
        else:
            print("Image tensor is NONE!")
            self.obs_buf = torch.zeros(self.num_envs, 3 * 64 * 64)
        # torch.save(self._bag.get_world_positions(), "bag.pt")
        observations = {self.bag.name: {"obs_buf": self.obs_buf}}
        
        return observations

    def pre_physics_step(self, actions):
        if not self._env.world.is_playing():
            return
        # print((3, self.camera_width, self.camera_height))
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        self.actions[:, 0, :] = actions[:, :9].clone().to(self._device)
        self.actions[:, 1, :] = actions[:, 9:].clone().to(self._device)

        # self.actions = actions.clone().to(self._device)
        targets = (self.franka_dof_targets
                   + self.franka_dof_speed_scales
                   * self.dt
                   * self.actions
                   * self.action_scale)
        self.franka_dof_targets[:] = tensor_clamp(targets,
                                                  self.franka_dof_lower_limits,
                                                  self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self.franka.count, dtype=torch.int32,
                                     device=self._device)
        # print(self.franka_dof_targets)
        self.franka.set_joint_position_targets(
            self.franka_dof_targets[:, 0, :],
            indices=env_ids_int32)
        self.franka2.set_joint_position_targets(
            self.franka_dof_targets[:, 1, :], indices=env_ids_int32)

    def post_reset(self):

        self.num_franka_dofs = self.franka.num_dof
        self.franka_dof_pos = torch.zeros(
            (self.num_envs, 2, self.num_franka_dofs), device=self._device)
        dof_limits1 = self.franka.get_dof_limits()
        dof_limits2 = self.franka2.get_dof_limits()
        self.franka_dof_lower_limits = torch.vstack(
            (dof_limits1[0, :, 0],
             dof_limits2[0, :, 0])).to(device=self._device)
        self.franka_dof_upper_limits = torch.vstack(
            (dof_limits1[0, :, 1],
             dof_limits2[0, :, 1])).to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(
            self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[0, self.franka.gripper_indices] = 0.1
        self.franka_dof_speed_scales[1, self.franka2.gripper_indices] = 0.1

        self.franka_dof_targets = torch.zeros(
            (self._num_envs, 2, self.num_franka_dofs), dtype=torch.float,
            device=self._device
        )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64,
                               device=self._device)
        self.reset_idx(indices)

    def reset_idx(self, env_ids):
        """Resets the environment"""
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset robot
        # self.bag.set_world_positions(self.bag_init_pos)
        state = self.bag.get_default_state()
        # self.bag.set_world_poses(
        #     positions=state.positions.clone(),
        #     orientations=state.orientations.clone(),
        #     indices=indices)18
        # self.bag.set_world_poses(
        #     positions=state.positions.clone() + 0.05*torch.randn(
        #         (self.num_envs, 3), device=self.device),
        #     orientations=state.orientations.clone(), indices=indices)

        cylinder_state = self.cylinder.get_default_state()
        self.cylinder.set_local_poses(
            translations=cylinder_state.positions.clone().to(self.device),
            orientations=cylinder_state.orientations.clone().to(self.device))
        self.cylinder.set_velocities(
            torch.zeros(self.num_envs, 6).to(self.device))

        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), 2, self.num_franka_dofs),
                                 device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )

        dof_pos = torch.zeros((num_indices, 2, self.franka.num_dof),
                              device=self._device)
        dof_vel = torch.zeros((num_indices, 2, self.franka.num_dof),
                              device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        self.franka.set_joint_position_targets(
            self.franka_dof_targets[env_ids, 0, :], indices=indices)
        self.franka.set_joint_positions(dof_pos[:, 0, :], indices=indices)
        self.franka.set_joint_velocities(dof_vel[:, 0, :], indices=indices)

        self.franka2.set_joint_position_targets(
            self.franka_dof_targets[env_ids, 1, :], indices=indices)
        self.franka2.set_joint_positions(dof_pos[:, 1, :], indices=indices)
        self.franka2.set_joint_velocities(dof_vel[:, 1, :], indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        # self._env.step(torch.zeros(self.num_envs, self.num_actions), True)
    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_franka_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.franka_grasp_pos,
            self.rim_pos,
            self.dist_reward_scale,
            self.action_penalty_scale,
            self.franka_dof_pos,
            self.finger_close_reward_scale
        )

    def is_done(self) -> None:
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1,
            torch.ones_like(self.reset_buf), self.reset_buf)
        # self.reset_buf = torch.where(
        #     self.bag_area[:, 0] > 1, torch.ones_like(self.reset_buf),
        #     self.reset_buf)

    def major_minor_axes(self, convex_hull_points: np.ndarray) -> tuple[float,
                                                                        float,
                                                                        float,
                                                                        float]:
        """ Compute convex hull using the vertex positions projected on z plane

        Args:
            convex_hull_points (np.ndarray): 2D points for which convex hull
            need to be calculated

        Returns:
            tuple[float, float, float, float]: major_axis_length,
            minor_axis_length, convex hull area, centroid
        """
        # Compute convex hull
        hull = ConvexHull(convex_hull_points)

        # Get the vertices of the convex hull
        hull_vertices = convex_hull_points[hull.vertices]

        # Compute the centroid of the convex hull
        centroid = np.mean(hull_vertices, axis=0)

        # Compute the covariance matrix
        cov_matrix = np.cov(hull_vertices.T)

        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = eig(cov_matrix)

        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Major axis length is twice the square root of the largest eigenvalue
        major_axis_length = 2 * np.sqrt(eigenvalues[0])

        # Minor axis length is twice the square root of the smallest eigenvalue
        minor_axis_length = 2 * np.sqrt(eigenvalues[1])
        if minor_axis_length > major_axis_length:
            minor_axis_length, major_axis_length = (major_axis_length,
                                                    minor_axis_length)
        return major_axis_length, minor_axis_length, hull.volume, centroid
