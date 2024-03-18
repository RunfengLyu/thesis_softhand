import numpy as np
from os import path
import gymnasium as gym
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import mediapy as media
import mujoco
from softhand_sim.controller.controller import PIDController


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
        "render_fps": 10,
    }

    def __init__(self, **kwargs):

        observation_space = Box(low=-np.inf, high=np.inf, shape=(51,), dtype=np.float64)
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            "assets/finger_model.xml",
        )
        EzPickle.__init__(self,  xml_file_path, **kwargs)
        
        MujocoEnv.__init__(
            self,
            xml_file_path,
            50,
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
            "render_fps": 10,
        }
        self.sensor_mean_weight = 0.1
        self.sensor_std_weight = 0.01
        self.controller = PIDController(0.1, 0.1, 0.1, 400)

    def step(self, a):
        reward = 0.5
        terminated = False
        truncated = False
        ob = self._get_obs()
        hand_z_old = self.data.xpos[7, 2]
        ball_z_old = self.data.xpos[-1, 2]
        #print("before a step", self.data.ctrl)
        # if self.render_mode == "rgb_array":
        #     frame = self.render()
        #     media.show_image(frame)
        a[-2] = a[-2] -50
        self.do_simulation(a, self.frame_skip)

        if self.check_handobject_contact():

            #grasp more
            # self.data.ctrl[0:5] = 1.5
            # mujoco.mj_step(self.model, self.data,20)

            ob = self._get_obs()
            reward = reward*(self.sensor_mean_weight * np.mean(ob) 
                        + self.sensor_std_weight * np.std(ob))

            #print("after step and in contact")
            # if self.render_mode == "rgb_array":
            #     frame = self.render()
            #     media.show_image(frame)

            hand_z_new = self.data.xpos[7, 2]
            ball_z_new = self.data.xpos[-1, 2]
            

            if hand_z_new - hand_z_old<0.05:
                info = {"reward_info": reward}
                return ob, reward, terminated, truncated, info 
            elif hand_z_new-hand_z_old > 0.05 and ((ball_z_new - ball_z_old) / (hand_z_new - hand_z_old))>0.9:
                reward = reward+((ball_z_new - ball_z_old) / (hand_z_new - hand_z_old))
                terminated = True
                if self.render_mode == "rgb_array":
                    frame = self.render()
                    media.show_image(frame)
                info = {"reward_info": reward}
                return ob, reward, terminated, truncated, info
            else:
                reward = reward+((ball_z_new - ball_z_old) / (hand_z_new - hand_z_old))
                info = {"reward_info": reward}
                return ob, reward, terminated, truncated, info
            # #case1: hand and ball are both stable in z
            # if hand_z_new>hand_z_old and (hand_z_new - hand_z_old) <0.1 and (ball_z_new - ball_z_old) <0.1:
            #     print("case1")
            #     reward = reward - 2
            #     info = {"case": "case1", "reward_info": reward}
            #     # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            #     return ob, reward, terminated, truncated, info
            
            # #case2: hand is up, ball is stable in z
            # elif hand_z_new>hand_z_old and 3 > (hand_z_new - hand_z_old) > 1 and (ball_z_new - ball_z_old) <0.1:
            #     reward = reward + ((ball_z_new - ball_z_old) / (hand_z_new - hand_z_old))
            #     print("case2 reward:", reward)
            #     info = {"case": "case2", "reward_info": reward}
            #     return ob, reward, terminated, truncated, info

            # #case3: hand is up, ball is up
            # elif hand_z_new>hand_z_old and 3 > (hand_z_new - hand_z_old) > 1 and 3 > (ball_z_new - ball_z_old) >0.5:
            #     reward = reward + (ball_z_new - ball_z_old) 
            #     print("hand and ball z position:", hand_z_new, hand_z_old, ball_z_new, ball_z_old)
            #     print("good grasp reward:", reward)
            #     if self.render_mode == "rgb_array":
            #         frame = self.render()
            #         media.show_image(frame)
            #     terminated = True
            #     info = {"case": "case3", "reward_info": reward}
            #     return ob, reward, terminated, truncated, info    
            
            # #case4: other bad behavior
            # else:
            #     print("case 4")
            #     terminated = True
            #     reward = -2
            #     info = {"case": "case4", "reward_info": reward}
            #     return ob, reward, terminated, truncated, info   

        else:
            #print("not in contact after step")
            terminated = True
            reward = -2
            info = {"case": "case0", "reward_info": reward}
            return ob, reward, terminated, truncated, info   


            

    def reset(self,seed=None,options=None):
        self._reset_simulation()
        mujoco.mj_step(self.model, self.data)
        qpos = self.init_qpos 
        # qpos[0] = self.init_qpos[0]+ self.np_random.uniform(
        #     size=1, low=-0.05, high=0
        # )
        qpos[1] = self.init_qpos[1]+ self.np_random.uniform(
            size=1, low=-0.05, high=0.05
        )
        qpos[2] = self.init_qpos[2]+ self.np_random.uniform(
            size=1, low=-0.05, high=0.05
        )
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        # while not self.check_handobject_contact():
        #     self._reset_simulation()
        #     mujoco.mj_step(self.model, self.data)
        #     qpos = self.init_qpos 
        #     qpos[0] = self.init_qpos[0]+ self.np_random.uniform(
        #         size=1, low=-0.05, high=0
        #     )
        #     qpos[1] = self.init_qpos[1]+ self.np_random.uniform(
        #         size=1, low=-0.05, high=0.05
        #     )
        #     qpos[2] = self.init_qpos[2]+ self.np_random.uniform(
        #         size=1, low=-0.05, high=0.05
        #     )
        #     qvel = self.init_qvel
        #     self.set_state(qpos, qvel)
        #     print("try reset")

        #     if self.render_mode == "rgb_array":
        #         frame = self.render()
        #         media.show_image(frame)
        print("reset done")
        # print(self.data.sensordata)
        # if self.render_mode == "rgb_array":
        #     frame = self.render()
        #     media.show_image(frame)
        return self._get_obs(),{}

    def _get_obs(self):
        sensor_matrix = self.data.sensordata
        position_exclude_hand = self.data.qpos[6:-6]
        velocity_exclude_hand = self.data.qvel[6:-6]
        return np.concatenate((sensor_matrix, position_exclude_hand, velocity_exclude_hand))
  
        
    def check_handobject_contact(self):
        sensordata = self._get_obs()
        if not(np.all(sensordata==0)) and self.data.xpos[7, 2] > 0.3:
            # print("contact")
            return True
        else:
            return False
    
    def test_swing_hang(self):
        #self.data.ctrl[-2] = -50
        self.data.ctrl[-1] =50

        if self.render_mode == "rgb_array":
            frame = self.render()
            media.show_image(frame)

        # if self.render_mode == "rgb_array":
        #     frame = self.render()
        #     media.show_image(frame)
        mujoco.mj_step(self.model, self.data,50)
        print("hand up")
        print(self.data.xpos[7, 2])
        print(self.data.xpos[-1, 2])
        if self.render_mode == "rgb_array":
            frame = self.render()
            media.show_image(frame)
        return

    
    def grasp(self):
        self.data.ctrl[0:5] = 1

        mujoco.mj_step(self.model, self.data,30)
        print("now grasp")
        if self.render_mode == "rgb_array":
            frame = self.render()
            media.show_image(frame)
        #pos, vel = self.save_state()
        return 



    

        
