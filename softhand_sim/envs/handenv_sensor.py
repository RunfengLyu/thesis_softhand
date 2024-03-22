import numpy as np
from os import path
import gymnasium as gym
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import mediapy as media
import mujoco
from softhand_sim.controller.controller import PIDController
import math


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

        observation_space = Box(low=-np.inf, high=np.inf, shape=(76,), dtype=np.float64)
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
        self.phase = 1
        self.n_grasping_try = 0
        self.last_contact_qpos = self.data.qpos.copy()
        self.last_contact_qvel = self.data.qvel.copy()
        self.lose_contact_try = 0
        self.after_graspingphase_qpos = self.data.qpos.copy()
        self.after_graspingphase_qvel = self.data.qvel.copy()
        self.after_grasp_phase_try = 0
        self.n_successful_grasp = 0
        self.n_reset = 0
        self.average_successful_grasp = []
        
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

    def step(self, a):
        reward = 1
        terminated = False
        truncated = False

        hand_z_old = self.data.xpos[7, 2]
        ball_z_old = self.data.xpos[-1, 2]
        sensor_old = self.data.sensordata.copy()
        # print("before a step touch:")
        # if self.render_mode == "rgb_array":
        #     frame = self.render()
        #     media.show_image(frame)
        a[-2] = a[-2] -50
        # print("action", a[5])
        # print(ob)
        self.do_simulation(a, self.frame_skip)
        
        sensor_new = self.data.sensordata.copy()
        ob = self._get_obs()
        contact = self.data.contact.geom
        hand_z_new = self.data.xpos[7, 2]
        ball_z_new = self.data.xpos[-1, 2]
        if self.check_hand_contact(ob,contact):
            self.last_contact_qpos = self.data.qpos.copy()
            self.last_contact_qvel = self.data.qvel.copy()

            #grasp more
            # self.data.ctrl[0:5] = 1.5
            # mujoco.mj_step(self.model, self.data,20)

            ob = self._get_obs()
            # print("after step and in contact")
            # print(ob)
            
            if self.phase == 1:
                # print(sensor_old)
                # print(sensor_new)
                if 100>np.mean(sensor_new)>10 and np.mean(sensor_new)>np.mean(sensor_old):
                    # print("grasping phase end")
                    reward = reward*(self.sensor_mean_weight * np.mean(sensor_new))
                    info = {"case": "grasping phase end","reward_info": reward}
                    self.after_graspingphase_qpos = self.data.qpos.copy()
                    self.after_graspingphase_qvel = self.data.qvel.copy()
                    self.phase = 2
                elif np.mean(sensor_new)>np.mean(sensor_old):
                    # print("inter grasping")
                    reward = reward*(self.sensor_mean_weight * np.mean(sensor_new))
                    info = {"case": "inter grasping","reward_info": reward}
                    self.n_grasping_try += 1
                    if(self.n_grasping_try>200):
                        reward = reward*(self.sensor_mean_weight * np.mean(sensor_new))
                        info = {"case": "grasping phase end","reward_info": reward}
                        self.after_graspingphase_qpos = self.data.qpos.copy()
                        self.after_graspingphase_qvel = self.data.qvel.copy()
                        self.phase = 2
                else:
                    # print("bad grasping")
                    reward = -0.1
                    info = {"case": "bad grasping","reward_info": reward}
                    self.n_grasping_try += 1
                    if(self.n_grasping_try>200):
                        reward = reward*(self.sensor_mean_weight * np.mean(sensor_new))
                        info = {"case": "grasping phase end","reward_info": reward}
                        self.after_graspingphase_qpos = self.data.qpos.copy()
                        self.after_graspingphase_qvel = self.data.qvel.copy()
                        self.phase = 2



            elif self.phase==2:
                if hand_z_new - hand_z_old>=0.01:
                    reward = (ball_z_new-0.3) *10
                    if (ball_z_new-0.3) > 0.05 and hand_z_new - 0.6 > 0.2:
                        reward = (hand_z_new - 0.6)*20
                        self.n_successful_grasp += 1
                        terminated = True
                        #print("nice up")
                        # if not self.check_hand_contact():
                        #     terminated = True
                        # print("nice hand up")
                        # if self.render_mode == "rgb_array":
                        #     frame = self.render()
                        #     media.show_image(frame)
                        info = {"case": "nice hand up","reward_info": reward}
                        return ob, reward, terminated, truncated, info 
                    elif hand_z_new-0.6> 0.2 and ball_z_new-0.3<0.05:
                        #print("grasping unstable and needs reset")
                        # if self.render_mode == "rgb_array":
                        #     frame = self.render()
                        #     media.show_image(frame)
                        reward = reward
                        self.set_state(self.after_graspingphase_qpos, self.after_graspingphase_qvel)
                        self.after_grasp_phase_try += 1
                        self.phase=1
                        if self.after_grasp_phase_try >600:
                            terminated = True
                        info = {"case": "grasping unstable and needs reset to grasping end","reward_info": reward}
                        return ob, reward, terminated, truncated, info  
                    info={"case": "already up","reward_info": reward}
                    return ob, reward, terminated, truncated, info     
                elif hand_z_new-hand_z_old < 0.01 :
                    reward = -0.1
                    #terminated = True
                    #print("move up too little")
                    # if self.render_mode == "rgb_array":
                    #     frame = self.render()
                    #     media.show_image(frame)
                    info = {"case": "move up too little","reward_info": reward}
                    return ob, reward, terminated, truncated, info
            return ob, reward, terminated, truncated, info

        else:

            # if self.render_mode == "rgb_array":
            #     frame = self.render()
            #     media.show_image(frame)
            # self.set_state(self.last_contact_qpos, self.last_contact_qvel)
            # self.lose_contact_try += 1
            # if self.lose_contact_try > 300:
            #     terminated = True
            terminated = True
            reward = -0.1
            # print("not in contact after step")
            info = {"case": "not in contact", "reward_info": reward}
            return ob, reward, terminated, truncated, info   


            

    def reset(self,seed=None,options=None):
        while True:
            self.n_grasping_try = 0
            self.lose_contact_try = 0
            self.after_grasp_phase_try = 0
            self.phase = 1
            self.after_graspingphase_qpos = np.zeros(27)
            self.after_graspingphase_qvel = np.zeros(27)
            self.last_contact_qpos = np.zeros(27)
            self.last_contact_qvel = np.zeros(27)
            self._reset_hand_pose()
            ob = self._get_obs()
            contact = self.data.contact.geom
            if self.check_hand_contact(ob,contact):
                break
        self.n_reset += 1
        if self.n_reset%500 == 0:
            #print("number of successful grasp each 500 episode", self.n_successful_grasp/self.n_reset)
            
            self.average_successful_grasp.append(self.n_successful_grasp/self.n_reset)
            self.n_successful_grasp = 0
        return self._get_obs(),{}

    def _get_obs(self):
        sensor_matrix = self.data.sensordata.copy()
        position_exclude_hand = self.data.qpos
        velocity_exclude_hand = self.data.qvel
        
        relative_hand_object = np.array(math.sqrt((self.data.xpos[7, 0] - self.data.xpos[-1, 0])**2 
                                         + (self.data.xpos[7, 1] - self.data.xpos[-1, 1])**2 
                                         + (self.data.xpos[7, 2] - self.data.xpos[-1, 2])**2)).reshape(1,)
        return np.concatenate((sensor_matrix, position_exclude_hand, velocity_exclude_hand, np.array(self.data.qpos[0].reshape(1,))))
  
        
    def check_hand_contact(self,ob,contact):

        # print("1",not(np.all(np.array(self.data.sensordata)==0)))
        # print("2",not(np.all(np.array(sensordata[0:20])==0)))
        # print("3",self._check_handobject_contact())
        # print("4",self.data.xpos[7, 2] > 0.3)
        # print("5",max(self.data.sensordata)<1000)
        if (not(np.all(np.array(ob[0:20])==0))) and (self.data.xpos[7, 2] > 0.3) and (max(ob[0:20])<500) and (not self._check_handtable_contact(contact)) and self._check_handobject_contact(contact):
            return True
        else:
            return False
    
    def _reset_hand_pose(self):
        qpos = np.zeros(27)

        # qpos[0] = self.np_random.uniform(
        #     size=1, low=-0.02, high=0.02
        # )
        qpos[1] = self.np_random.uniform(
            size=1, low=-0.01, high=0.01
        )
        qpos[2] = self.np_random.uniform(
            size=1, low=-0.01, high=0.01
        )
        qpos[3] = self.np_random.uniform(
            size=1, low=-0.01, high=0.01
        )
        qpos[5] = self.np_random.uniform(
            size=1, low=-0.01, high=0.01
        )
        qvel = np.zeros(27)
        self.set_state(qpos, qvel)

    
    def test_swing_hang(self):
        self.data.ctrl[-2] = -50
        self.data.ctrl[5] =50

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
        self.data.ctrl[0:5] = 2

        mujoco.mj_step(self.model, self.data,50)
        print("now grasp")
        if self.render_mode == "rgb_array":
            frame = self.render()
            media.show_image(frame)
        #pos, vel = self.save_state()
        return 
    
    def _check_handtable_contact(self,contact):
        contact_state = False
        # Access contact information
        for i in range(contact.shape[0]):
            # Get the contact object

            
            if ((int(contact[i][0]),int(contact[i][1])) or (int(contact[i][1]),int(contact[i][0]))) in self.handtable_contact_conditions.values():
                contact_state = True
                break
        return contact_state
    
    def _check_handobject_contact(self,contact):
        contact_state = False
        # Access contact information
        for i in range(contact.shape[0]):
            # Get the contact object
            if ((int(contact[i][0]),int(contact[i][1])) or (int(contact[i][1]),int(contact[i][0]))) in self.handobject_contact_conditions.values():

                # print("hand object contact")
                # if np.all(self.data.sensordata==0):
                #     print(self.data.sensordata)
                #print(contact.geom)
                contact_state = True
                break
        return contact_state
    

    def _check_handobject_distance(self):
        distance = np.sqrt((self.data.xpos[7, 0] - self.data.xpos[-1, 0])**2 
                                         + (self.data.xpos[7, 1] - self.data.xpos[-1, 1])**2 
                                         + (self.data.xpos[7, 2] - self.data.xpos[-1, 2])**2)
        if distance < 2 and self.data.xpos[7, 2] > self.data.xpos[-1, 2]:
            return True
        else:
            return False

          
# class CurriculumHandEnv(HandEnv):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.phase = 1

#     def step(self, action, phase=None):
#         phase = self.phase
#         ob, reward, terminated, truncated, info = super().step(action,phase)

#         if phase == 1:
#             # Only reward for grasping
#             if info['case'] == 'grasping phase end':
#                 phase = 2
#             elif info['case'] == 'bad grasping':
#                 reward = -1
#                 return ob, reward, terminated, truncated, info
#             else:
#                 reward = -1 # too early into next stage
#         elif phase == 2:
#             # Only reward for liftingbad grasping
#             if info['case'] == 'nice hand up':
#                 reward = 1
#             elif info['case'] == 'grasping unstable and needs reset':
#                 terminated = True
#                 reward = -0.5
#             elif info['case'] == 'move up too little':
#                 reward = -0.5
#                 return ob, reward, terminated, truncated, info
#             # elif info['case'] in ['grasping phase end', 'bad grasping']:
#             #     reward = -1  # Penalize phase 1 cases during phase 2
#             #     terminated = True  # Optionally terminate the episode

#         return ob, reward, terminated, truncated, info
        
