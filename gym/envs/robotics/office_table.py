import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


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
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps):
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



class OfficeTable(RobotEnv_revised):
    """Superclass for all Fetch environments."""

    def __init__(
        self,
        model_path,
        n_substeps,
        gripper_extra_height,
        block_gripper,
        has_object,
        target_in_the_air,
        target_offset,
        obj_range,
        target_range,
        distance_threshold,
        initial_qpos,
        reward_type,
        target_objects,
        all_objects,
        senstive=1, # senstive=1->~1cm trigger max activation. Senstive=2, ~0.5cm trigger max activation. Senstive=3, ~0.33cm trigger max activateiton.
        action_scale=1, # 1 normal speed, 0.1 slow speed of action
        posrel_reciprocal=False
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.target_objects=target_objects
        self.all_objects=all_objects
        self.achieved_goal=""
        self.senstive=senstive
        self.action_scale=action_scale
        self.posrel_reciprocal=posrel_reciprocal

        print('info created: \n\nsenstive={senstive}, \n\ntarget_objects={target_objects}, \n\nrandom_check={random_check},\n\n action_scale={action_scale}\n\n posrel_reciprocal={posrel_reciprocal}!!!!\n\n'.format(target_objects=target_objects,senstive=senstive,random_check=np.random.rand(),action_scale=action_scale,posrel_reciprocal=posrel_reciprocal))

        assert all_objects=="RGB"

        self.obj_color2id={
            "R":0,"G":1,"B":2
        }
        self.obj_id2color={
            0:"R",1:"G",2:"B"
        }
        self.objs_valid_positions=[
            [1.1, 0.5, 0.425, 1.0, 0.0, 0.0, 0.0], 
            [1.1, 0.7, 0.425, 1.0, 0.0, 0.0, 0.0],
            [1.1, 0.9, 0.425, 1.0, 0.0, 0.0, 0.0]
            ]
        

        self.targets_valid_position=[
            [1.4, 0.9, 0.425], # target for id0 obj- red obj
            [1.4, 0.7, 0.425], # target for id1 obj - green obj
            [1.4, 0.5, 0.425]  # target for id2 obj- blue obj
        ] 

        super(OfficeTable, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=4,
            initial_qpos=initial_qpos,
        )

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self):
        scale=1

        base_reward=1
        # stage
        
        if self.goal.startswith(self.achieved_goal):
            finished_stage=len(self.achieved_goal)
            # reaching object
            if finished_stage<len(self.goal):
                target_obj_color=self.goal[finished_stage]
                target_obj_id=self.obj_color2id[target_obj_color]
                grip_pos = self.sim.data.get_site_xpos("robot0:grip")
                obj_pos= self.sim.data.get_site_xpos("object{}".format(target_obj_id))
                target_pos= self.targets_valid_position[target_obj_id].copy()

                dis_grip2obj= pos_distance(grip_pos,obj_pos)
                dis_obj2target= pos_distance(obj_pos,target_pos)
                reward_reaching_obj=-dis_grip2obj
                reward_reaching_target=-dis_obj2target

                reward=finished_stage*2+2*reward_reaching_obj+2*reward_reaching_target
                # return 
            else: # achieved the final goal already, the finished_stage is 1 bigger than as usual
                reward = finished_stage*2
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

        pos_ctrl *= 0.05*self.action_scale  # limit maximum change in position
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
        targets_pos = self.targets_valid_position.copy()

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
            achieveds[id]=pos_distance(objects_rel_pos2target[id],np.array([0,0,0]))<0.05
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
            print("objects_rel_pos_reciprocal:",objects_rel_pos_reciprocal)
            print("objects_rel_pos:",objects_rel_pos)
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


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()


        # wrong achieved
        done= (not self.goal.startswith(self.achieved_goal)) or self.goal==self.achieved_goal

        # out of range
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        leng=pos_distance(grip_pos[:2],np.array([1,0.75]))
        # print('leng',leng)
        if grip_pos[0]>1.52 or grip_pos[0]<0.96 or grip_pos[1]<0.15 or grip_pos[1]>1.15 or grip_pos[2]<0.41 or grip_pos[2]>0.8 or leng>0.57:
            done=True
        # print(grip_pos[2])
        if grip_pos[2]<0.421: # collision with table
            done=True
        # print("grip_pos",grip_pos)
        
        # print('rob len',leng)
        # if (grip_pos[0]-0.96)**2+()


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
            self.sim.model.site_pos[site_id] = self.targets_valid_position[id]-sites_offset[site_id]
            
    def _reset_sim(self):
        # print("reset")
        self.sim.set_state(self.initial_state)

        # randomize objects init position
        valid_positions=list(self.objs_valid_positions).copy()
        for i in range(len(valid_positions)):
            choice=valid_positions[np.random.randint(0,len(valid_positions))]
            valid_positions.remove(choice)
            self.sim.data.set_joint_qpos("object{}:joint".format(i), choice)

        # no randomize target goal position
        # None
        for i in range(len(self.all_objects)):
            site_id = self.sim.model.site_name2id("target{}".format(i))
            self.sim.model.site_pos[site_id] = self.targets_valid_position[i]

        
        self.achieved_goal=""

        self.sim.forward()
        return True

    def _sample_goal(self):
        # Goal is fixed if the target_objects is given.
        goal=self.target_objects # "RGB" or "RG" or "R" or "B" ...
        return goal

    def _is_success(self, achieved_goal, desired_goal):
        return achieved_goal==desired_goal

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
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
