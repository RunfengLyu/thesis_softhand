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
        "render_fps": 5,
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
            100,
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
            "render_fps": 5,
        }

    def step(self, a):
        reward = 0
        terminated = True
        truncated = True
        ob = self._get_obs()
        if self.check_handobject_contact():
            self.do_simulation(a, self.frame_skip)
            ob = self._get_obs()
            
            terminated = self.test_grasp_stability_disturbance()
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
    
    def grasp(self):
       
        self.data.ctrl[0:5] = 2
        
        mujoco.mj_step(self.model, self.data,100)
        #print("here",self.data.qpos, self.data.qvel)
        if self.render_mode == "rgb_array":
            frame = self.render()
            media.show_image(frame)
        #pos, vel = self.save_state()
        return 

    
    def test_grasp_stability_handswing(self):
        self.data.ctrl[-3] = 20

        #print("now rotate hand")

        mujoco.mj_step(self.model, self.data, 300)

        if(self.data.xpos[-1, 2]-0.5>0.05 
           and abs(self.data.xpos[-1,0]+0.2)<0.1 
           and abs(self.data.xpos[-1,1]-0.1)<0.1):
            print("Grasp is stable")
            if self.render_mode == "rgb_array":
                frame = self.render()
                media.show_image(frame)

            return True
        else:
            return False
        
    def tet_grasp_stability_handforward(self):
        self.data.ctrl[-3] = 20

        diff_handball_x = self.data.xpos[-1, 0] - self.data.xpos[2, 0]
            
        mujoco.mj_step(self.model, self.data, 300)

        new_diff_handball_x = self.data.xpos[-1, 0] - self.data.xpos[2, 0]

        if(abs(new_diff_handball_x-diff_handball_x)<0.01):
            print("Grasp is stable")
            if self.render_mode == "rgb_array":
                frame = self.render()
                media.show_image(frame)

            return True
        else:
            return False
    
    def test_grasp_stability_disturbance(self):
        self.data.ctrl[-4] = 20
        diff_handball_x = self.data.xpos[-1, 0] - self.data.xpos[2, 0]
            
        mujoco.mj_step(self.model, self.data, 300)

        new_diff_handball_x = self.data.xpos[-1, 0] - self.data.xpos[2, 0]

        if(abs(new_diff_handball_x-diff_handball_x)<0.01):
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
    
    def test_object_move(self):
        self.data.ctrl[-4] = 20
        print("now test object move")
        for i in range(300):
            
            mujoco.mj_step(self.model, self.data)

        if self.render_mode == "rgb_array":
            frame = self.render()
            media.show_image(frame)

    
        
    
