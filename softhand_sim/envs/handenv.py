import numpy as np
from typing import Dict
from os import path
import gymnasium
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
import time

from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 1.5,
    "azimuth": 90.0,
}


class HandEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        reward_type: str = "dense",
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        # reward_near_weight: float = 0.5,
        # reward_dist_weight: float = 1,
        # reward_control_weight: float = 0.1,
        **kwargs,
    ):
        print(path.dirname(path.realpath(__file__)))
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            "assets/finger_model.xml",
        )
        # utils.EzPickle.__init__(
        #     self,
        #     xml_file_path,
        #     frame_skip,
        #     default_camera_config,
        #     reward_near_weight,
        #     reward_dist_weight,
        #     reward_control_weight,
        #     **kwargs
        # )

        # self._reward_near_weight = reward_near_weight
        # self._reward_dist_weight = reward_dist_weight
        # self._reward_control_weight = reward_control_weight

        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(35,),
            dtype=np.float32,
        )

        MujocoEnv.__init__(
            self,
            xml_file_path,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.model)

        # whether to have sparse rewards
        if reward_type.lower() == "dense":
            self.sparse_reward = False
        elif reward_type.lower() == "sparse":
            self.sparse_reward = True
        else:
            raise ValueError(
                f"Unknown reward type, expected `dense` or `sparse` but got {reward_type}"
            )
        
        # Override action_space to -1, 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape
        )

        self._state_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(36,), dtype=np.float64
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(36,), dtype=np.float64
                ),
                "obj_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "target_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
            }
        )

        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )

        self._init_qpos = self.data.qpos.ravel().copy()
        self._init_qvel = self.data.qvel.ravel().copy()
        self._init_touch_data = self.data.sensordata.ravel().copy()
        self._init_obj_pos = self.get_body_com("Object").ravel().copy()
        self._init_hand_pos = self.get_body_com("hand").ravel().copy()
        
        self._init_palm_middle_dif = self.compute_joint_difference("ipalm", "mMCP")
        self.start_time = time.time()
        self.frames = []

        EzPickle.__init__(self, **kwargs)




    def step(self, a):
        # vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        # vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        # reward_near = -np.linalg.norm(vec_1) * self._reward_near_weight
        # reward_dist = -np.linalg.norm(vec_2) * self._reward_dist_weight
        # reward_ctrl = -np.square(action).sum() * self._reward_control_weight

        # self.do_simulation(action, self.frame_skip)

        # observation = self._get_obs()
        # reward = reward_dist + reward_ctrl + reward_near
        # info = {
        #     "reward_dist": reward_dist,
        #     "reward_ctrl": reward_ctrl,
        #     "reward_near": reward_near,
        # }
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mean + a * self.act_rng  # mean center and scale
        self.do_simulation(a, self.frame_skip)
        # Move the hand up by 0.1m
        hand_body_idx = self._model_names.body_name2id['hand']
        #print("hand_idx", hand_body_idx)
        hand_pos = self.data.xpos[hand_body_idx]
        #print("allbody", hand_pos)
        hand_pos[2] += 0.1
       
        # hand_body_idx = self._model_names.body_name2id['hand']
        # print("body_idx:", body_idx)
        # print("hand:", self.data.xpos[body_idx])
        obs = self._get_obs()
        #reward = self._get_reward()
        reward = 0
        end_time = time.time()
        done = self._is_done(end_time, self.start_time)
        if done or (self._init_palm_middle_dif == self.compute_joint_difference("ipalm", "mMCP")):
            reward = 1
        #grasp_success = reward > 0.5
        if self.render_mode == "human":
            self.render()
            self.render.update_scene(data)
            pixels = self.render()
            self.frames.append(pixels)
        truncated = 1
        return obs, reward, done, truncated, {"end_time": end_time}

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco.MjViewer(self.sim)
        self.viewer.render()

    def _get_reward(self):
        # Get the object's position
        obj_pos = self._state_space['obj_pos']
        touch_data = self.data.sensordata.ravel()
        # Check if the object is above the table
        if obj_pos[2] > 0.1 and touch_data[:, 0]>0 and touch_data[:, 3]>0 and touch_data[:, 12] > 0:  # Assuming obj_pos[2] is the height
            return 1
        else:
            return 0
    

    def _get_obs(self):
        # Get observation from the simulation (e.g., state of joints, sensors)
        qp = self.data.qpos.ravel()
        #obj_pos = self.get_body_com("object").ravel()
        hand_pos = self.get_body_com("hand").ravel()
        touch_data = self.data.sensordata.ravel()

        obs = np.concatenate(
            (qp[:-6], hand_pos, touch_data)
        )
        return obs

    def _get_info(self):
        return{
            "touch": self.data.sensordata.ravel(),
            "distance_palm&ball": np.linalg.norm(self._hand_position-self._obj_position, ord = 1),
            "frames":self.frames
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed = seed)
        # If a seed is provided, use it to seed the random number generator
        if seed is not None:
            np.random.seed(seed)
        self._hand_position = self._init_hand_pos
        self._obj_position = self._init_obj_pos
        # Choose the agent's location uniformly at random
        # self._hand_position = self.np_random.integers(0, 2, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._ball_position = self._hand_position
        obs = self._get_obs()
        info = self._get_info()  
       
        return obs, info



    def reset_model(self):
        return self._get_obs()

    def _is_done(self, time_now, start_time):
        # Check if the episode is done
        if time_now - start_time > 10:
            return True
        else:
            return False
    
    def compute_joint_difference(self, joint1, joint2):
      # Get the indices of the joints
      joint1_idx = self._model_names.joint_name2id[joint1]
      joint2_idx = self._model_names.joint_name2id[joint2]

      # Get the positions of the joints
      joint1_pos = self.data.qpos[joint1_idx]
      joint2_pos = self.data.qpos[joint2_idx]

      # Compute the difference
      difference = np.linalg.norm(joint1_pos - joint2_pos)

      return difference


