# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import math
import numpy as np
from numpy.linalg import eig
import torch
import re

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import gymnasium as gym # noqa: F401 E261

from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage  # noqa: E501
from omni.isaac.core.utils.nucleus import get_assets_root_path
# from omni.isaac.core.utils.torch.transformations import *
# from omni.isaac.core.utils.torch.rotations import *

from omni.isaac.core.objects import DynamicCylinder
from omni.isaac.core.prims import ClothPrimView, RigidPrimView, ClothPrim, ParticleSystem  # noqa: E501
from omni.isaac.core.materials import ParticleMaterial


from omniisaacgymenvs.tasks.base.rl_task import RLTask

from utils.prim_utils import find_first_matching_prim, get_all_matching_child_prims
from pxr import PhysxSchema, Sdf, Semantics, Usd, UsdGeom, UsdPhysics, UsdShade


class ClothBagAttachTask(RLTask):
    """
    Environment Class for Cloth bag opening and lifting environment.
    """
    def __init__(self, name, env, sim_config, offset=None) -> None:
        self.update_config(sim_config)
        self.dt = 1/60
        self._num_observations = 40*3 + 3*3
        self._num_actions = 6
        self._ball_radius = 0.25

        RLTask.__init__(self, name, env)
        self.act1 = torch.zeros((self.num_envs, 6), device=self.device)
        self.act2 = torch.zeros((self.num_envs, 6), device=self.device)

        # self.rim_index = [108, 125, 134, 143, 152, 166, 181, 193, 202, 211,
        #                   220, 229, 244, 261, 275, 284, 293, 302, 309, 326,
        #                   340, 349, 358, 367, 448, 449, 450, 451, 502, 503,
        #                   504, 505, 604, 605, 606, 607, 640, 641, 642, 643]
        self.rim_index =[108, 125, 134, 143, 152, 166, 451, 450, 449, 448, 309, 326, 340, 349, 358, 367, 643, 642, 641, 640, 229,220, 211, 202, 193, 181, 505, 504, 503, 502, 244, 261, 275, 284, 293, 302, 607, 606, 605, 604]
        # self.mid_rim = [224,  233,  240,  247,  254,  261,  268,  273,  280,
        #                 287,  294,  301, 309,  318,  325,  332,  340,  347,
        #                 352,  361,  368,  375,  383,  391, 459,  460,  461,
        #                 462,  487,  488,  489,  490,  563,  564,  565,  566,
        #                 591,  592,  593,  594]
        self.max_area = -1
        self.pos_reach1 = torch.tensor([[2.8707, 5.0185, 0.8201],
                                        [2.5769, 4.6079, 1.2965]],
                                       device=self.device)
        self.pos_reach2 = torch.tensor([[2.6438, 5.5950, 1.524],
                                        [2.4078, 5.9541, 1.6887]],
                                       device=self.device)
        self.upper_velocity = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                           device=self.device)
        self.lower_velocity = self.upper_velocity * -1
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
        
        self._prim_path = self.default_zero_env_path + "/baghandle"
        self._usd_path = self._task_cfg["sim"]["bag"]["usd_path"]
        
        self.stretch_stiffness = self._task_cfg["sim"]["bag"]["stretch_stiffness"]
        self.bend_stiffness = self._task_cfg["sim"]["bag"]["bend_stiffness"]
        self.shear_stiffness = self._task_cfg["sim"]["bag"]["shear_stiffness"]
        self.spring_damping = self._task_cfg["sim"]["bag"]["spring_damping"]
        self.particle_mass = self._task_cfg["sim"]["bag"]["particle_mass"]

        self.restOffset = self._task_cfg["sim"]["bag"]["restOffset"]
        self.contactOffset = self._task_cfg["sim"]["bag"]["contactOffset"]

        self.drag = self._task_cfg["sim"]["bag"]["drag"]
        self.lift = self._task_cfg["sim"]["bag"]["lift"]
        self.friction = self._task_cfg["sim"]["bag"]["friction"]

    def cleanup(self) -> None:
        # initialize remaining buffers
        RLTask.cleanup(self)

        # override observation buffer for camera data
        self.obs_buf = torch.zeros(
            (self.num_envs, 80, 3), device=self.device, dtype=torch.float)

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Gets intrinsic matrix for the camera.

        Returns:
            np.ndarray: Intrinsic Matrix (3x3)
        """
        # https://github.com/NVIDIA-Omniverse/Orbit/blob/main/source/extensions/omni.isaac.orbit/omni/isaac/orbit/sensors/camera/camera.py
        height, width = self.camera_height, self.camera_width
        horiz_aperture = 20.955
        focal_length = 24.0
        # calculate the field of view
        fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
        # calculate the focal length in pixels
        focal_px = width * 0.5 / math.tan(fov / 2)
        # create intrinsic matrix for depth linear
        a = focal_px
        b = width * 0.5
        c = focal_px
        d = height * 0.5
        # return the matrix
        return np.array([[a, 0, b], [0, c, d], [0, 0, 1]], dtype=float)

    def set_up_scene(self, scene) -> None:
        self.stage = get_current_stage()
        self.assets_root_path = get_assets_root_path()

        self.get_cloth_bag()
        super().set_up_scene(scene=scene, replicate_physics=False)

        # TODO: Fix prim path.
        self.prim_paths_expr = self.root_prim_path_expr.replace("env_0", ".*")
        self.bag = ClothPrimView(
            prim_paths_expr=self.prim_paths_expr,
            name="cloth_bag"
                        )

        self.sphere1 = RigidPrimView(
                            prim_paths_expr="/World/envs/.*/Cube",
                            name="fancy_sphere1", reset_xform_properties=False
                        )

        self.sphere2 = RigidPrimView(
                            prim_paths_expr="/World/envs/.*/Cube_01",
                            name="fancy_sphere2", reset_xform_properties=False
                        )
        self.cylinder = RigidPrimView(
                            prim_paths_expr="/World/envs/.*/Cylinder",
                            name="cylinder", reset_xform_properties=False
        )

        scene.add(self.bag)
        scene.add(self.sphere1)
        scene.add(self.sphere2)
        scene.add(self.cylinder)
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("cloth"):
            scene.remove_object("cloth", registry_only=True)
        if scene.object_exists("fancy_sphere1"):
            scene.remove_object("fancy_sphere1", registry_only=True)
        if scene.object_exists("fancy_sphere2"):
            scene.remove_object("fancy_sphere2", registry_only=True)
        self.bag = ClothPrimView(
            prim_paths_expr=self.prim_paths_expr,
            name="cloth_bag", reset_xform_properties=False
                        )

        self.sphere1 = RigidPrimView(
                            prim_paths_expr="/World/envs/.*/Cube",
                            name="fancy_sphere1", reset_xform_properties=False
                        )

        self.sphere2 = RigidPrimView(
                            prim_paths_expr="/World/envs/.*/Cube_01",
                            name="fancy_sphere2", reset_xform_properties=False
                        )
        self.cylinder = RigidPrimView(
                            prim_paths_expr="/World/envs/.*/Cylinder",
                            name="cylinder", reset_xform_properties=False
        )
        scene.add(self.bag)
        scene.add(self.sphere2)
        scene.add(self.sphere1)
        scene.add(self.cylinder)

    def get_cloth_bag(self):
        """
        Gets the cloth bag from USD to be added to the stage.
        """
        # _usd_path = '/home/irobotics/thesis/summer-project/sth3.usd'
        # _usd_path = "usd_files/bag_w_attach_on_floor_w_cylinder.usd"
        _usd_path = "usd_files/bag_handle_cubes.usd"
        mesh_path = self.default_zero_env_path
        print(self._usd_path)
        add_reference_to_stage(self._usd_path, mesh_path)

        particle_system_path = "/World/Materials/particleSystem"
        particle_material_path = "/World/Materials/particleMaterial"
        # TODO
        _particle_material = ParticleMaterial(
            prim_path=particle_material_path,
            drag=self.drag,
            lift=self.lift,
            friction=self.friction,
        )

        # TODO: Change to dynamic allocation by using mass/number of vertices.

        _particle_system = ParticleSystem(
            prim_path=particle_system_path,
            simulation_owner="/physicsScene",
            rest_offset=self.restOffset,
            contact_offset=self.contactOffset,
            solid_rest_offset=self.restOffset,
            fluid_rest_offset=0.0,
            particle_contact_offset=self.contactOffset,
            solver_position_iteration_count=56,
            max_neighborhood=96,
            # radius=radius
        )

        print(self._prim_path)
        template_prim = find_first_matching_prim(self._prim_path)
        print(template_prim)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim from expression: '{self.cfg.prim_path}'.")
        
        template_prim_path = template_prim.GetPath().pathString
        print("TEMPLATE PRIM PATH: ", template_prim_path)
        # FIXME
        root_prims = get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(PhysxSchema.PhysxParticleClothAPI)
        )
        
        if len(root_prims) != 1:
            raise RuntimeError(
                f"Failed to find a single rigid body when resolving '{self.cfg.prim_path}'."
                f"Found multiple '{root_prims}' under '{template_prim_path}'."
            )
        
        root_prim_path = root_prims[0].GetPath().pathString
        self.root_prim_path_expr = self._prim_path + root_prim_path[len(template_prim_path):]
        
        ClothPrim(
            prim_path="/World/envs/env_0/baghandle/Cube_0_1_021_Cube_136",
            # prim_path=self.root_prim_path_expr,
            particle_system=_particle_system,
            particle_material=_particle_material,
            stretch_stiffness=self.stretch_stiffness,
            bend_stiffness=self.bend_stiffness,
            shear_stiffness=self.shear_stiffness,
            spring_damping=self.spring_damping,
            particle_mass=self.particle_mass,
            visible=True,
                    )
        

    def add_cylinder(self):
        """
        Gets the Cylinder Floating in the air to be added to the stage.
        """
        DynamicCylinder(
            prim_path=self.default_zero_env_path + "/cylinder",
            radius=0.5,
            height=1.0,
            color=np.array([1.0, 0.0, 0.0]),
            mass=1.0
            )

    def get_observations(self) -> dict:

        # self.obs_buf = torch.zeros(self.num_envs, 80, 3).to(self._device)

        self.bag_area = None
        observations = {

        }

        bag_pos = (self.bag.get_world_positions()[:,
                                                  self.rim_index, :].clone())
        loc1, _ = self.sphere1.get_world_poses()
        loc2, _ = self.sphere2.get_world_poses()
        cyl, _ = self.cylinder.get_world_poses()
        self.obs_buf = torch.concat(
            (bag_pos, loc1.unsqueeze(dim=1), loc2.unsqueeze(dim=1),
             cyl.unsqueeze(dim=1)), dim=1).view(self.num_envs, -1)
        observations = {self.bag.name: {"obs_buf": self.obs_buf}}
        # torch.save(self._bag.get_world_positions(), "bag.pt")
        return observations

    def pre_physics_step(self, actions):
        if not self.world.is_playing():
            exit()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        # actions = torch.randn((2, 6), device=self.device)
        action = actions.clone()  # * 0.1
        self.velocity += action * self.dt
        self.velocity = torch.clamp(self.velocity, self.lower_velocity,
                                    self.upper_velocity)
        action = torch.clamp(action, self.lower_velocity,
                                    self.upper_velocity)
        self.act1[:, :3] = self.velocity[:, :3]
        self.act2[:, :3] = self.velocity[:, 3:]
        # self.act1[:, :3] = self.velocity[:, :3]
        # self.act2[:, :3] = self.velocity[:, 3:]
        self.sphere1.set_velocities(self.act1)
        self.sphere2.set_velocities(self.act2)
        # print(self._cylinder.get_world_poses())
        loc1, _ = self.sphere1.get_world_poses()
        loc2, _ = self.sphere2.get_world_poses()

        # self.sphere1.set_local_poses(translations=loc1 + action[:,:3] )
        # self.sphere2.set_local_poses(translations=loc2 + action[:,3:] )

    def post_reset(self):
        super().post_reset()
        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64,
                               device=self._device)
        self.reset_idx(indices)

    def reset_idx(self, env_ids):
        """Resets the environment"""
        indices = env_ids.to(dtype=torch.int32)
        self.velocity = torch.zeros((self.num_envs, 6), device=self.device)
        # reset robot
        # self.bag.set_world_positions(self.bag_init_pos)
        state = self.bag.get_default_state()
        self.bag.set_world_poses(
            positions=state.positions.clone() + 0.25*torch.randn(
                (self.num_envs, 3),
                device=self.device),
            orientations=state.orientations.clone(), indices=indices)
        state_cube1 = self.sphere1.get_default_state()
        self.sphere1.set_local_poses(
            translations=state_cube1.positions.clone() + 0.25*torch.randn(
                (self.num_envs, 3), device=self.device),
            orientations=state_cube1.orientations.clone())
        state_cube2 = self.sphere2.get_default_state()
        self.sphere2.set_local_poses(
            translations=state_cube2.positions.clone() + 0.25*torch.randn(
                (self.num_envs, 3), device=self.device),
            orientations=state_cube2.orientations.clone())

        cylinder_state = self.cylinder.get_default_state()
        self.cylinder.set_local_poses(
            translations=cylinder_state.positions.clone() + 0.25 * torch.randn(
                (self.num_envs, 3), device=self.device),
            orientations=cylinder_state.orientations.clone())
        self.cylinder.set_velocities(torch.zeros(self.num_envs, 6).to(self._device))
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def calculate_metrics(self) -> None:
        # rewards = 1
        rewards = torch.zeros(self.num_envs, device=self.device)
        # cylinder_pos, _ = self._cylinder.get_world_poses()
        # cube_pos, _ = self._sphere1.get_world_poses()
        # d = torch.norm(cube_pos - cylinder_pos, p=2, dim=-1)
        # dist_reward = 1.0 / (1.0 + d**2)
        # dist_reward *= dist_reward
        # dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)
        rim_pos = self.bag.get_world_positions()[:, self.rim_index, :].clone()
        # cube_pos1, _ = self.sphere1.get_world_poses()
        # cube_pos2, _ = self.sphere2.get_world_poses()
        cylinder_pos, _ = self.cylinder.get_world_poses()
        
        # req_cube_pos1 = cylinder_pos + torch.tensor((0.0, 0.6, 0.5), device=self.device).repeat(self.num_envs, 1)
        # req_cube_pos2 = cylinder_pos + torch.tensor((0.0, -0.6, 0.5), device=self.device).repeat(self.num_envs, 1)
        
        
        mid_point = torch.mean(rim_pos, dim=1)
        # print(cylinder_pos, mid_point)
        
        d = torch.norm(cylinder_pos - mid_point, dim=1)
        # d1 = torch.norm(req_cube_pos1 - cube_pos1, dim=1)
        # d2 = torch.norm(req_cube_pos2 - cube_pos2, dim=1)
        # dist_reward1 = 1 / (
        #     1 + 9*(d1 ** 2))
        # dist_reward2 = 1 / (
        #     1 + 9*(d2 ** 2))
        # print(dist_reward1, dist_reward2, d1, d2, d)
        dist_reward = 1 / (
            1 + 9*(d ** 2))
        # dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)
        
        # print(dist_from_cylinder)
        for i in range(self.num_envs):
            area = Polygon(rim_pos[i, :, 0:2].cpu().numpy()).area
            # top_pts = self.bag.get_world_positions()[i, self.rim_index, :]
            # major, minor, area, centre = self.major_minor_axes(top_pts[:,
            #                                                            [0, 2]]
            #                                                    .cpu().numpy())
            r = (
                ((area/2.5 - 0.6) > 0) * 1.0
                # * ((minor/major - 0.6) > 0) * 1.0
            )
            # print(area)
            rewards[i] = r
        # print(dist_reward)
        self.rew_buf[:] = dist_reward.to(self.device) #* rewards

    def is_done(self) -> None:
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1,
            torch.ones_like(self.reset_buf), self.reset_buf)
        # self.reset_buf = torch.where(
        #     self.dist_from_cylinder > 0.9,
        #     torch.ones_like(self.reset_buf), self.reset_buf)

    def major_minor_axes(
        self, convex_hull_points: np.ndarray) -> tuple[float,
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
