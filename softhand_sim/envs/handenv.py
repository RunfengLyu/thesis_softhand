import numpy as np
from os import path
import gymnasium as gym
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import mediapy as media
import mujoco


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 5,
}


class HandEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 250,
    }

    def __init__(self, **kwargs):

        observation_space = Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float64)
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            "assets/finger_model.xml",
        )
        EzPickle.__init__(self,  xml_file_path, **kwargs)
        
        MujocoEnv.__init__(
            self,
            xml_file_path,
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 250,
        }

    def step(self, a):
        
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        
        terminated = self.test_grasp_stability()
        if terminated:
            reward= 1
        else:
            reward = 0
        if self.data.time > 2:
            truncated = True
        else:
            truncated = False
        if self.render_mode == "rgb_array":
            frame = self.render()
            media.show_image(frame)

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return ob, reward, terminated, truncated, {}
    
    # def holdup(self):
    #     qpos = self.data.qpos
    #     self.data.qvel[:] = 0
    #     qvel = self.data.qvel
    #     self.set_state(qpos, qvel)
    #     if self.render_mode == "rgb_array":
    #         frame = self.render()
    #         media.show_image(frame)
    #     return

    def reset(self,seed=None,options=None):
        qpos = self.init_qpos 
        qpos[1] = self.init_qpos[1]+ self.np_random.uniform(
            size=1, low=-0.1, high=0.1
        )
        qpos[2] = self.init_qpos[2]+ self.np_random.uniform(
            size=1, low=-0.1, high=0.1
        )
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        # if self.render_mode == "rgb_array":
        #     frame = self.render()
        #     media.show_image(frame)

        return self._get_obs(),{}

    def _get_obs(self):
        print(self.data.sensordata)
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()
    
    # def rotate_hand(self):
    #     self.data.ctrl[-3] = -20

    #     mujoco.mj_step(self.model, self.data,200)
    #     if self.render_mode == "rgb_array":
    #         frame = self.render()
    #         media.show_image(frame)
    #     print(self.data.xpos)
    #     return 
    
    # def grasp(self):
       
    #     self.data.ctrl[0] = -3
        
    #     mujoco.mj_step(self.model, self.data,50)
    #     print("here",self.data.qpos, self.data.qvel)
    #     if self.render_mode == "rgb_array":
    #         frame = self.render()
    #         media.show_image(frame)
    #     #pos, vel = self.save_state()
    #     return 


    # def move_stick(self):
    #     self.data.ctrl[-1] = 20

    #     mujoco.mj_step(self.model, self.data)
    #     if self.render_mode == "rgb_array":
    #         frame = self.render()
    #         media.show_image(frame)
    #     return 
    
    def test_grasp_stability(self):
        self.data.ctrl[-3] = -100
        print("now test stability")
        mujoco.mj_forward(self.model, self.data)
        if(self.data.xpos[-1,2]>0.55):
            print("Grasp is stable")
            return True
        else:
            return False
    
