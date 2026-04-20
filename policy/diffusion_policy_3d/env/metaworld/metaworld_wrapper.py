import gym
import numpy as np
import metaworld

from termcolor import cprint
from gym import spaces
from scipy.spatial.transform import Rotation as R

from diffusion_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling


# Bounds in WORLD coordinates (post-transform). Keep a generous workspace that
# includes the table, arm, and objects, but crops the far wall and floor.
#   x: left-right across the table
#   y: near-far (toward the arm base)
#   z: up-down (table at ~0.00-0.05, workspace extends up to ~1.0)
TASK_BOUNDS = {
    'default': [-0.5, -0.5, -0.05, 1.0, 1.5, 1.5],
}


class MetaWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device="cuda:0",
                 use_point_crop=True,
                 num_points=1024,
                 ):
        super(MetaWorldEnv, self).__init__()

        if '-v2' not in task_name:
            task_name = task_name + '-v2-goal-observable'

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        self.env._freeze_rand_vec = False

        # Camera pose. Must be set BEFORE we read extrinsics below.
        # https://arxiv.org/abs/2212.05698
        self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5

        self.device_id = int(device.split(":")[-1])
        self.image_size = 128

        self.pc_generator = PointCloudGenerator(
            sim=self.env.sim, cam_names=['corner2'], img_size=self.image_size
        )
        self.use_point_crop = use_point_crop
        cprint("[MetaWorldEnv] use_point_crop: {}".format(self.use_point_crop), "cyan")
        self.num_points = num_points

        # Build the camera->world transform directly from MuJoCo's camera extrinsics.
        # Raw PCs from the generator are in camera frame; to get world coords:
        #     world = cam_R @ cam_point + cam_pos_world
        # This replaces the hardcoded 61.4°/-7° rotation that did not correspond
        # to the real camera orientation and caused objects to be absent from the
        # cropped point clouds.
        cam_id = self.env.sim.model.camera_name2id('corner2')
        cam_pos_world = self.env.sim.model.cam_pos[cam_id].copy()
        cam_quat_wxyz = self.env.sim.model.cam_quat[cam_id].copy()
        # scipy expects xyzw; MuJoCo stores wxyz
        cam_rotmat = R.from_quat(
            [cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]]
        ).as_matrix()

        # get_point_cloud() applies:  points @ pc_transform.T * pc_scale + pc_offset
        # Since (R @ P.T).T == P @ R.T, setting pc_transform = cam_rotmat gives us
        # the correct rotation, and pc_offset = cam_pos_world adds the translation.
        self.pc_transform = cam_rotmat
        self.pc_scale = np.array([1.0, 1.0, 1.0])
        self.pc_offset = cam_pos_world

        bounds = TASK_BOUNDS.get(task_name, TASK_BOUNDS['default'])
        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]

        self.episode_length = self._max_episode_steps = 200
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32,
            ),
            'depth': spaces.Box(
                low=0, high=255,
                shape=(self.image_size, self.image_size),
                dtype=np.float32,
            ),
            'agent_pos': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32,
            ),
            'point_cloud': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32,
            ),
            'full_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(20,),
                dtype=np.float32,
            ),
        })

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
        img = self.env.sim.render(
            width=self.image_size, height=self.image_size,
            camera_name="corner2", device_id=self.device_id,
        )
        return img

    def render_high_res(self, resolution=1024):
        img = self.env.sim.render(
            width=resolution, height=resolution,
            camera_name="corner2", device_id=self.device_id,
        )
        return img

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(
            device_id=self.device_id
        )  # raw point cloud in CAMERA frame, Nx3 or Nx6

        if not use_rgb:
            point_cloud = point_cloud[..., :3]

        # Apply camera->world transform. Same math as before:
        #   points @ R.T  ==  (R @ points.T).T
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        if self.pc_offset is not None:
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps')
        depth = depth[::-1]
        return point_cloud, depth

    def get_visual_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        return {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
        }

    def step(self, action: np.array):
        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_state,
        }

        done = done or self.cur_step >= self.episode_length
        return obs_dict, reward, done, env_info

    def reset(self):
        self.env.reset()
        self.env.reset_model()
        raw_obs = self.env.reset()
        self.cur_step = 0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        return {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_obs,
        }

    def seed(self, seed=None):
        pass

    def set_seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        return self.get_rgb()

    def close(self):
        pass
