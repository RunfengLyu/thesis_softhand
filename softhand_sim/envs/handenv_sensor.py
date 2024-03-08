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

        observation_space = Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64)
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
        reward = 0
        terminated = True
        truncated = True
        ob = self._get_obs()
        if self.check_handobject_contact():
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
            # if self.render_mode == "rgb_array":
            #     frame = self.render()
            #     media.show_image(frame)

            # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return ob, reward, terminated, truncated, {}
        else:
            #print("Hand is not in contact with object")
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
        sensor_matrix = self.data.sensordata
        return sensor_matrix
    
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
        self.data.ctrl[-2] = 2

        diff_handball_x = self.data.xpos[-1, 0] - self.data.xpos[2, 0]
        #print(diff_handball_x)
        for i in range(1000):
            mujoco.mj_step(self.model, self.data)
        new_diff_handball_x = self.data.xpos[-1, 0] - self.data.xpos[2, 0]
        # print(new_diff_handball_x)

        if(abs(diff_handball_x - new_diff_handball_x) < 0.02):
            print("Grasp is stable")
            if self.render_mode == "rgb_array":
                frame = self.render()
                media.show_image(frame)
            return True
        else:
            return False
        
    def check_handobject_contact(self):
        sensordata = self._get_obs()
        if np.all(sensordata==0):
            return False
        else:
            return True
            
    
        
    
