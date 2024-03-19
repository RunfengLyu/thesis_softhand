import numpy as np
from os import path
import gymnasium as gym
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box, Discrete
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
        self.handtable_contact_conditions = {
            "condition0": (self.data.geom('palm2').id,self.data.geom('table_surface').id),
            "condition1": (self.data.geom('palm').id, self.data.geom('table_surface').id),
            "condition2": (self.data.geom('index1g').id, self.data.geom('table_surface').id),
            "condition3": (self.data.geom('index2g').id,self.data.geom('table_surface').id),
            "condition4": (self.data.geom('index3g').id,self.data.geom('table_surface').id),
            "condition5": (self.data.geom('middle1g').id,self.data.geom('table_surface').id),
            "condition6": (self.data.geom('middle2g').id,self.data.geom('table_surface').id),
            "condition7": (self.data.geom('middle3g').id,self.data.geom('table_surface').id),
            "condition8": (self.data.geom('ring1g').id,self.data.geom('table_surface').id),
            "condition9": (self.data.geom('ring2g').id,self.data.geom('table_surface').id),
            "condition10": (self.data.geom('ring3g').id,self.data.geom('table_surface').id),
            "condition11": (self.data.geom('little1g').id,self.data.geom('table_surface').id),
            "condition12": (self.data.geom('little2g').id,self.data.geom('table_surface').id),
            "condition13": (self.data.geom('little3g').id,self.data.geom('table_surface').id),
            "condition14": (self.data.geom('thumb1g').id,self.data.geom('table_surface').id),
            "condition15": (self.data.geom('thumb2g').id,self.data.geom('table_surface').id),
            "condition16": (self.data.geom('thumb3g').id,self.data.geom('table_surface').id),
            "condition17": (self.data.geom('palm2').id,self.data.geom('table_surfacebase').id),
            "condition18": (self.data.geom('palm').id,self.data.geom('table_surfacebase').id),
            "condition19": (self.data.geom('index1g').id,self.data.geom('table_surfacebase').id),
            "condition20": (self.data.geom('index2g').id,self.data.geom('table_surfacebase').id),
            "condition21": (self.data.geom('index3g').id,self.data.geom('table_surfacebase').id),
            "condition22": (self.data.geom('middle1g').id,self.data.geom('table_surfacebase').id),
            "condition23": (self.data.geom('middle2g').id,self.data.geom('table_surfacebase').id),
            "condition24": (self.data.geom('middle3g').id,self.data.geom('table_surfacebase').id),
            "condition25": (self.data.geom('ring1g').id,self.data.geom('table_surfacebase').id),
            "condition26": (self.data.geom('ring2g').id,self.data.geom('table_surfacebase').id),
            "condition27": (self.data.geom('ring3g').id,self.data.geom('table_surfacebase').id),
            "condition28": (self.data.geom('little1g').id,self.data.geom('table_surfacebase').id),
            "condition29": (self.data.geom('little2g').id,self.data.geom('table_surfacebase').id),
            "condition30": (self.data.geom('little3g').id,self.data.geom('table_surfacebase').id),
            "condition31": (self.data.geom('thumb1g').id,self.data.geom('table_surfacebase').id),
            "condition32": (self.data.geom('thumb2g').id,self.data.geom('table_surfacebase').id),
            "condition33": (self.data.geom('thumb3g').id,self.data.geom('table_surfacebase').id)
        }
        self.handobject_contact_conditions = {
            "condition0": (self.data.geom('palm2').id,self.data.geom('object').id),
            "condition1": (self.data.geom('palm').id, self.data.geom('object').id),
            "condition2": (self.data.geom('index1g').id, self.data.geom('object').id),
            "condition3": (self.data.geom('index2g').id,self.data.geom('object').id),
            "condition4": (self.data.geom('index3g').id,self.data.geom('object').id),
            "condition5": (self.data.geom('middle1g').id,self.data.geom('object').id),
            "condition6": (self.data.geom('middle2g').id,self.data.geom('object').id),
            "condition7": (self.data.geom('middle3g').id,self.data.geom('object').id),
            "condition8": (self.data.geom('ring1g').id,self.data.geom('object').id),
            "condition9": (self.data.geom('ring2g').id,self.data.geom('object').id),
            "condition10": (self.data.geom('ring3g').id,self.data.geom('object').id),
            "condition11": (self.data.geom('little1g').id,self.data.geom('object').id),
            "condition12": (self.data.geom('little2g').id,self.data.geom('object').id),
            "condition13": (self.data.geom('little3g').id,self.data.geom('object').id),
            "condition14": (self.data.geom('thumb1g').id,self.data.geom('object').id),
            "condition15": (self.data.geom('thumb2g').id,self.data.geom('object').id),
            "condition16": (self.data.geom('thumb3g').id,self.data.geom('object').id)
        }
        self.action_space = Discrete(2)

    def step(self, a):
        reward = 1
        terminated = False
        truncated = False
        ob = self._get_obs()
        hand_z_old = self.data.xpos[7, 2]
        ball_z_old = self.data.xpos[-1, 2]
        self.data.ctrl[-2] = -50
        #print("before a step touch:")
        # if self.render_mode == "rgb_array":
        #     frame = self.render()
        #     media.show_image(frame)
        # a[-2] = a[-2] -50
        # print("action", a)
        # print(ob)
        #self.do_simulation(a, self.frame_skip)

        #grasping step
        if a == 0:
            self.data.ctrl[0:5] = self.data.ctrl[0:5] + 0.1
            
            mujoco.mj_step(self.model, self.data,50)
            ob = self._get_obs()
            reward = reward*(self.sensor_mean_weight * np.mean(ob[0:20]))
            print("grasping step")
            if self.render_mode == "rgb_array":
                    frame = self.render()
                    media.show_image(frame)
            if self.check_handobject_contact():
                info = {"reward_info": reward}
                return ob, reward, terminated, truncated, info
            else:
                reward = -0.5
                terminated = True
                info = {"reward_info": reward}
                return ob, reward, terminated, truncated, info


        elif a == 1:
            self.data.ctrl[5] = 50
            mujoco.mj_step(self.model, self.data,50)
            print("lifting step")
            if self.render_mode == "rgb_array":
                    frame = self.render()
                    media.show_image(frame)
            ball_z_new = self.data.xpos[-1, 2]
            ob = self._get_obs()
            if self.check_handobject_contact():
                if ball_z_new - ball_z_old > 0.5:
                    reward = (ball_z_new - ball_z_old)*10 +reward*(self.sensor_mean_weight * np.mean(ob[0:20]))
                    terminated=True
                    info = {"reward_info": reward}
                    return ob, reward, terminated, truncated, info
                else:
                    reward = (ball_z_new - ball_z_old) +reward*(self.sensor_mean_weight * np.mean(ob[0:20]))
                    info = {"reward_info": reward}
                    return ob, reward, terminated, truncated, info
            else:
                reward = -0.5
                terminated = True
                info = {"reward_info": reward}
                return ob, reward, terminated, truncated, info

   

    def reset(self,seed=None,options=None):
        self._reset_hand_pose()
        while not self.check_handobject_contact():
            self._reset_hand_pose()
            #print("try reset")

            # if self.render_mode == "rgb_array":
            #     frame = self.render()
            #     media.show_image(frame)
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
        if self._check_handobject_contact() and self.data.xpos[7, 2] > 0.3 and not self._check_handtable_contact() and max(sensordata[0:20])<1000:
            return True
        else:
            return False
    
    def _reset_hand_pose(self):
        self._reset_simulation()
        mujoco.mj_step(self.model, self.data)
        qpos = self.init_qpos 
        qpos[1] = self.init_qpos[1]+ self.np_random.uniform(
            size=1, low=-0.1, high=0.1
        )
        qpos[2] = self.init_qpos[2]+ self.np_random.uniform(
            size=1, low=-0.1, high=0.1
        )
        # qpos[3] = self.init_qpos[3]+ self.np_random.uniform(
        #     size=1, low=-0.3, high=0.3
        # )
        # qpos[5] = self.init_qpos[5]+ self.np_random.uniform(
        #     size=1, low=-0.3, high=0.3
        # )
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

    
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
    
    def _check_handtable_contact(self):
        contact_state = None
        # Access contact information
        for i in range(self.data.ncon):
            # Get the contact object
            contact = self.data.contact[i]
            
            if ((int(contact.geom[0]),int(contact.geom[1])) or (int(contact.geom[1]),int(contact.geom[0]))) in self.handtable_contact_conditions.values():
                contact_state = True
                break
        return contact_state
    
    def _check_handobject_contact(self):
        contact_state = None
        # Access contact information
        for i in range(self.data.ncon):
            # Get the contact object
            contact = self.data.contact[i]
            if ((int(contact.geom1),int(contact.geom2)) or (int(contact.geom2),int(contact.geom1))) in self.handobject_contact_conditions.values():
                contact_state = True
                break
        return contact_state

          

        
