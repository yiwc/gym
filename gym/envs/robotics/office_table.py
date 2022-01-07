import numpy as np

from gym.envs.robotics import rotations, robot_env, utils
import logging
logging.basicConfig(level=logging.INFO)
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def pos_distance(pos1,pos2):
    assert np.array(pos1).shape == np.array(pos2).shape
    return np.linalg.norm(pos1 - pos2, axis=-1)

import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )

DEFAULT_SIZE = 500
OBS_LOW=-2
OBS_HIGH=2

class RobotEnv_revised(gym.Env):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps, task):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File {} does not exist".format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.seed()
        self._get_predefined_setups(initial_qpos=initial_qpos)
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        self.observation_space = spaces.Box(
                    OBS_LOW, OBS_HIGH, shape=obs.shape, dtype="float32"
        )

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer


TASK_REACH="reach"
TASK_NEAR_PICK="near_pick"
TASK_PLACE="near_place"
TASK_PICK_PLACE="pick_place"
OfficeTableTasks=[TASK_REACH,TASK_NEAR_PICK,TASK_PLACE,TASK_PICK_PLACE]
logger=logging.getLogger("OfficeTable")
class OfficeTable(RobotEnv_revised):
    """Superclass for all Fetch environments."""

    def __init__(
        self,
        reward_type="dense",
        n_substeps=20,
        block_gripper=False,
        gripper_extra_height=0.2,
        target_objects="RGB",
        senstive=0.5, # senstive=1->~1cm trigger max activation. Senstive=2, ~0.5cm trigger max activation. Senstive=3, ~0.33cm trigger max activateiton.
        action_scale=1, # 1 normal speed, 0.1 slow speed of action
        posrel_reciprocal=False,
        reward_scale=1, # 10, scale up the reward
        model_path=os.path.join("fetch", "office_table.xml"),
        task=TASK_PICK_PLACE,
        reach_goal_threshold=0.05, # less than 1cm, we call it reach
        DEBUG=False,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
        """

        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.2, 0.5, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object1:joint": [1.2, 0.7, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object2:joint": [1.2, 0.9, 0.4, 1.0, 0.0, 0.0, 0.0]
        }
        assert task in OfficeTableTasks, "OfficeTableTasks"
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_objects=target_objects
        self.all_objects='RGB'
        self.achieved_goal=""
        self.senstive=senstive
        self.action_scale=action_scale
        self.posrel_reciprocal=posrel_reciprocal
        self.reward_scale=reward_scale
        self.reach_goal_threshold=reach_goal_threshold
        self.task=task
        self.DEBUG=DEBUG

        if self.task==TASK_REACH:
            assert len(self.target_objects)==1,"can only reach one objects, please set target_objects to one color, eg. target_objects=\"R\""
        assert self.all_objects=="RGB"   

        self.obj_color2id={"R":0,"G":1,"B":2 }
        self.obj_id2color={ 0:"R",1:"G",2:"B"}
        self.objs_candidates_positions=[ 
            [1.1, 0.5, 0.425, 1.0, 0.0, 0.0, 0.0], 
            [1.1, 0.7, 0.425, 1.0, 0.0, 0.0, 0.0],
            [1.1, 0.9, 0.425, 1.0, 0.0, 0.0, 0.0]
            ]
        self.targets_positions=[ # the three plates on the table
            [1.4, 0.9, 0.425], # target for id0 obj- red obj
            [1.4, 0.7, 0.425], # target for id1 obj - green obj
            [1.4, 0.5, 0.425]  # target for id2 obj- blue obj
        ] 

        # Configs - Reach
        self.configs_reach_hovering_relative=np.array([0,0,0.1]) # 2cm above object 

        # Configs - Near Pick
        self.configs_nearpick_height=0.05 # raise to 5cm


        super(OfficeTable, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=4,
            initial_qpos=initial_qpos,
            task=self.task
        )

# GoalEnv methods
# ----------------------------
    def compute_reward(self):
        comute_reward_dict={
            TASK_PICK_PLACE:self.compute_reward_pickplace,
            TASK_REACH:self.compute_reward_reach,
            TASK_NEAR_PICK:self.compute_reward_nearpick,
        }
        return comute_reward_dict[self.task]()

    def compute_reward_nearpick(self):

        assert len(self.target_objects)==1

        target_obj_id=self.obj_color2id[self.target_objects]
        obj_pos= self.sim.data.get_site_xpos("object{}".format(target_obj_id))
        raise_height=(obj_pos[2]-0.425)*100


        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dis_grip2obj= pos_distance(grip_pos,obj_pos)

        contact_with_gripper=dis_grip2obj<0.03

        return (raise_height*contact_with_gripper-dis_grip2obj*10+contact_with_gripper)*self.reward_scale


    def compute_reward_reach(self):
        assert len(self.target_objects)==1
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        target_obj_id=self.obj_color2id[self.target_objects]
        obj_pos= self.sim.data.get_site_xpos("object{}".format(target_obj_id))
        # obj_pos[2]+=0.03
        dis_grip2obj= pos_distance(grip_pos,obj_pos+self.configs_reach_hovering_relative)
        return  -dis_grip2obj*self.reward_scale

    def compute_reward_pickplace(self):
        scale=self.reward_scale
        base_reward=1
        
        if self.goal.startswith(self.achieved_goal):
            finished_stage=len(self.achieved_goal)
            # reaching object
            if finished_stage<len(self.goal):
                target_obj_color=self.goal[finished_stage]
                target_obj_id=self.obj_color2id[target_obj_color]
                grip_pos = self.sim.data.get_site_xpos("robot0:grip")
                obj_pos= self.sim.data.get_site_xpos("object{}".format(target_obj_id))
                target_pos= self.targets_positions[target_obj_id].copy()

                dis_grip2obj= pos_distance(grip_pos,obj_pos)
                dis_obj2target= pos_distance(obj_pos,target_pos)
                reward_reaching_obj=-dis_grip2obj
                reward_reaching_target=-dis_obj2target

                reward=finished_stage+reward_reaching_obj+reward_reaching_target
                # return 
            else: # achieved the final goal already, the finished_stage is 1 bigger than as usual
                reward = finished_stage
                # return 
            
        else:
            reward = base_reward
            # return base_reward
        
        # return 0
        return scale*(reward+base_reward)

# RobotEnv methods
# ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", 0.0)
            self.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", 0.0)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05*self.action_scale*0.3  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # print('grep pos',grip_pos)
        # targets_pos = [self.sim.data.get_site_xpos("target{}".format(i)) for i in range(len(self.all_objects))]
        targets_pos = self.targets_positions.copy()

        gripper_state = robot_qpos[-2:]
        gripper_vel = (
            robot_qvel[-2:] * dt 
        )  
        objects_pos=[None for i in range(len(self.all_objects))]
        objects_rot=[None for i in range(len(self.all_objects))]
        objects_velp=[None for i in range(len(self.all_objects))]
        objects_velr=[None for i in range(len(self.all_objects))]
        objects_rel_pos=[None for i in range(len(self.all_objects))]
        objects_rel_pos2target=[None for i in range(len(self.all_objects))]
        achieveds=[False for i in range(len(self.all_objects))]
        
        for _,obj_color in enumerate(self.all_objects):
            id=self.obj_color2id[obj_color]
            objects_pos[id]=self.sim.data.get_site_xpos("object{}".format(id))
            objects_rot[id]=rotations.mat2euler(self.sim.data.get_site_xmat("object{}".format(id)))
            objects_velp[id] = self.sim.data.get_site_xvelp("object{}".format(id)) * dt
            objects_velr[id] = self.sim.data.get_site_xvelr("object{}".format(id)) * dt
            objects_rel_pos[id]=objects_pos[id] - grip_pos
            objects_velp[id]  -= grip_velp
            objects_rel_pos2target[id] = objects_pos[id]-targets_pos[id]

            # Get Achived
            if self.task==TASK_PICK_PLACE:
                achieveds[id]=pos_distance(objects_rel_pos2target[id],np.array([0,0,0]))<0.05
            elif self.task==TASK_REACH:
                object_hover_pos=objects_pos[id].copy()
                object_hover_pos+=self.configs_reach_hovering_relative
                achieveds[id]=pos_distance(object_hover_pos-grip_pos,np.array([0,0,0]))<self.reach_goal_threshold
                logger.info('relate pos={}, dis={}'.format(object_hover_pos-grip_pos,pos_distance(object_hover_pos-grip_pos,np.array([0,0,0]))))
            elif self.task==TASK_NEAR_PICK:
                achieveds[id]=(objects_pos[id][2]-0.425)>self.configs_nearpick_height
            else:
                raise NotImplementedError("Can not get achived goal, please check")
            if self.DEBUG:
                logger.info("objects_pos[id]={}".format(objects_pos[id]))
                logger.info("grip_pos={}".format(grip_pos))
                logger.info("achieveds={}".format(achieveds))
            self.achieved_goal = self.achieved_goal+obj_color if (achieveds[id] and not obj_color in self.achieved_goal) else self.achieved_goal

        # print("achieved_goal= {} , goal={}".format(self.achieved_goal,self.goal))
        # print("objects_rel_pos2target=",objects_rel_pos2target)
        # print('\n')
        grip_velp=np.array(grip_velp)*300*self.senstive
        gripper_state=np.array(gripper_state)*100*self.senstive
        # gripper_vel=np.array(gripper_vel)*100
        objects_rel_pos_reciprocal=np.array(1/(np.array(objects_rel_pos)*10+1e-6))
        objects_rel_pos=np.array(objects_rel_pos)*100*self.senstive
        # objects_rel_pos_reciprocal=np.array(1/(objects_rel_pos+1e-6))
        objects_rel_pos2target=np.array(objects_rel_pos2target)*100*self.senstive
        gripper_hight=np.array([grip_pos[2]-0.418])*100*self.senstive
        _obs=[
                # grip_pos,
                gripper_state,
                grip_velp,
                gripper_hight
                # gripper_vel,
            ] + \
            list(map(lambda x:x.ravel(),objects_rel_pos)) + \
            list(map(lambda x:x.ravel(),objects_rel_pos2target))
        if self.posrel_reciprocal:
            _obs+=list(map(lambda x:x.ravel(),objects_rel_pos_reciprocal))
            # print("objects_rel_pos_reciprocal:",objects_rel_pos_reciprocal)
            # print("objects_rel_pos:",objects_rel_pos)
        obs = np.concatenate(
            _obs
        )
        # print("grip_velp",grip_velp)
        # print("gripper_state",gripper_state)
        # print("gripper_vel",gripper_vel)
        # print("objects_rel_pos",objects_rel_pos)
        # print("objects_rel_pos2target",objects_rel_pos2target)
        obs = np.tanh(obs)*2
        # obs = np.clip(obs,OBS_LOW,OBS_HIGH)
        # print("gripper_hight",gripper_hight)
        return obs.copy()

    def _get_done(self):
        pass
        # finish all goal
        done= (not self.goal.startswith(self.achieved_goal)) or self.goal==self.achieved_goal
        
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")

        # go beyond bounder
        leng=pos_distance(grip_pos[:2],np.array([1,0.75]))
        if grip_pos[0]>1.52 or grip_pos[0]<0.96 or grip_pos[1]<0.15 or grip_pos[1]>1.15 or grip_pos[2]<0.41 or grip_pos[2]>0.8 or leng>0.57:
            done=True
        # collision with table
        if grip_pos[2]<0.421: 
            done=True
        return done

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        done=self._get_done()
        reward = self.compute_reward()
        return obs, reward, done, {}

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

    def _render_callback(self):
        self.sim.forward()
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

        # sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        for id in range(len(self.target_objects)):
            site_id = self.sim.model.site_name2id("target{}".format(id))
            self.sim.model.site_pos[site_id] = self.targets_positions[id]-sites_offset[site_id]
            
    def _reset_sim(self):


        sequence=["R","G","B"]
        np.random.shuffle(sequence)

        if self.task==TASK_NEAR_PICK:
            states={
                0:self.init_states['pos_a'],
                1:self.init_states['pos_b'],
                2:self.init_states['pos_c'],
            }
            indexid=sequence.index(self.target_objects)
            print("index",indexid)
            self.sim.set_state(states[indexid])
        else:
            self.sim.set_state(self.init_states['pos_default'])

        for id,color in enumerate(sequence):
            obj_id=self.obj_color2id[color]
            pos_id=self.objs_candidates_positions[id]
            self.sim.data.set_joint_qpos("object{}:joint".format(obj_id),pos_id)


        # valid_positions=list(self.objs_candidates_positions).copy()
        # for i in range(len(valid_positions)):
        #     choice=valid_positions[np.random.randint(0,len(valid_positions))]
        #     valid_positions.remove(choice)
        #     sequence.append(choice)
        # # randomize objects init position
        # valid_positions=list(self.objs_candidates_positions).copy()
        # for i in range(len(valid_positions)):
        #     choice=valid_positions[np.random.randint(0,len(valid_positions))]
        #     valid_positions.remove(choice)
            

        for i in range(len(self.all_objects)):
            site_id = self.sim.model.site_name2id("target{}".format(i))
            self.sim.model.site_pos[site_id] = self.targets_positions[i]


        self.achieved_goal=""

        self.sim.forward()
        return True

    def _sample_goal(self):
        # Goal is fixed if the target_objects is given.
        goal=self.target_objects # "RGB" or "RG" or "R" or "B" ...
        return goal

    def _get_predefined_setups(self, initial_qpos):

        self.init_states={
            "pos_a":None,
            "pos_b":None,
            "pos_c":None,
            "pos_default":None
        }
        pos_default=[-0.498, 0.005, -0.431 + self.gripper_extra_height]
        pos_a=[-0.75,-0.23, -0.431 + self.gripper_extra_height] # should be on the top of object a
        pos_b=[-0.75,-0.03, -0.431 + self.gripper_extra_height]
        pos_c=[-0.75,0.17, -0.431 + self.gripper_extra_height]

        init_poses={
            "pos_a":pos_a,
            "pos_b":pos_b,
            "pos_c":pos_c,
            "pos_default":pos_default
        }

        for key in init_poses.keys():
            state=self.sim.get_state()
            for name, value in initial_qpos.items():
                self.sim.data.set_joint_qpos(name, value)
            utils.reset_mocap_welds(self.sim)
            self.sim.forward()

            # Move end effector into position.
            pos_default=init_poses[key]
            gripper_target = np.array(
                pos_default
            ) + self.sim.data.get_site_xpos("robot0:grip")
            gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
            self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
            self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
            for _ in range(10):
                self.sim.step()

            self.init_states[key]=copy.deepcopy(self.sim.get_state())
            self.sim.set_state(state)

    def _env_setup(self, initial_qpos):


        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        
        if self.task!=TASK_NEAR_PICK:
            pos_default=[-0.498, 0.005, -0.431 + self.gripper_extra_height]
        else:
            pos_default=[-0.75,-0.03, -0.431 + self.gripper_extra_height]
        gripper_target = np.array(
            pos_default
            # [-0.498, 0.005, -0.431 + self.gripper_extra_height]
            # [-0., 0., -0. + self.gripper_extra_height]
        ) + self.sim.data.get_site_xpos("robot0:grip")

        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()


        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()

    def render(self, mode="human", width=500, height=500):
        return super(OfficeTable, self).render(mode, width, height)
