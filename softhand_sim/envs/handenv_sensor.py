import numpy as np
from os import path
import gymnasium as gym
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import mediapy as media
import mujoco


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 6,
    "distance": 6,
}


class HandEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
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
            25,
            observation_space=observation_space,
            camera_name="track_hand",
            **kwargs,
        )
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 20,
        }
        self.sensor_mean_weight = 1e-2
        self.sensor_std_weight = 1e-3

    def step(self, a):
        # print("step")
        reward = 0.5
        terminated = False
        truncated = False
        ob = self._get_obs()

        a[5] = 0
        self.do_simulation(a, self.frame_skip)
        # print("first grasp")
        # if self.render_mode == "rgb_array":
        #     frame = self.render()
        #     media.show_image(frame)
        if self.check_handobject_contact():
            ob = self._get_obs()
            reward= reward*(self.sensor_mean_weight * np.mean(ob) 
                        + self.sensor_std_weight * np.std(ob))
            # self.data.ctrl[0:5] = 0.5
            # print(reward)
            # mujoco.mj_step(self.model, self.data,30)
            # print("grasp more")
            # if self.render_mode == "rgb_array":
            #     frame = self.render()
            #     media.show_image(frame)

            if self.render_mode == "rgb_array":
                frame = self.render()
                media.show_image(frame)

            # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return ob, reward, terminated, truncated, {}
        else:
            reward = -1
            return ob, reward, terminated, truncated, {}

    


    def reset(self,seed=None,options=None):
        qpos = self.init_qpos 
        # qpos[4] = self.init_qpos[1]+ self.np_random.uniform(
        #     size=1, low=-0.3, high=0.3
        # )
        # qpos[5] = self.init_qpos[2]+ self.np_random.uniform(
        #     size=1, low=-0.3, high=0.3
        # )
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
        self.data.ctrl[0:5] = 1.5

        mujoco.mj_step(self.model, self.data,20)
        print("now grasp")
        if self.render_mode == "rgb_array":
            frame = self.render()
            media.show_image(frame)
        #pos, vel = self.save_state()
        return 

        
    def check_handobject_contact(self):
        sensordata = self._get_obs()
        if np.all(sensordata==0):
            return False
        else:
            return True
    
    # def test_object_move(self):
    #     self.data.ctrl[-4] = 20
    #     print("now test object move")
    #     for i in range(300):
            
    #         mujoco.mj_step(self.model, self.data)

    #     if self.render_mode == "rgb_array":
    #         frame = self.render()
    #         media.show_image(frame)
    def test_swing_hang(self):
        #self.data.ctrl[3] =100
        #self.data.ctrl[4] =20
        self.data.ctrl[5] =100

        if self.render_mode == "rgb_array":
            frame = self.render()
            media.show_image(frame)
        mujoco.mj_step(self.model, self.data,100)
        print("hand up")
        if self.render_mode == "rgb_array":
            frame = self.render()
            media.show_image(frame)
            return

    
    def test_grasp_stability_hand_moveup(self):
        self.data.ctrl[5] = 50
        reward = -1
        hand_z_old = self.data.xpos[3, 2]
        ball_z_old = self.data.xpos[-1, 2]
        # if self.render_mode == "rgb_array":
        #     frame = self.render()
        #     media.show_image(frame)

        mujoco.mj_step(self.model, self.data, 400)
        # print("after hand move")
        # if self.render_mode == "rgb_array":
        #     frame = self.render()
        #     media.show_image(frame)

        hand_z_new = self.data.xpos[3, 2]
        ball_z_new = self.data.xpos[-1, 2]
        if abs(hand_z_new - hand_z_old) <0.05:
            return reward

        else:
            reward = (ball_z_new - ball_z_old) / (hand_z_new - hand_z_old)
            mujoco.mj_step(self.model, self.data)
            print("good grasp")
            if self.render_mode == "rgb_array":
                frame = self.render()
                media.show_image(frame)
            return reward 



    

        
