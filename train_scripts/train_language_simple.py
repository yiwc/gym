import gym

from stable_baselines3 import DQN
from gym.envs.minigrid import wrappers
from stable_baselines3 import PPO
import time
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from collections import OrderedDict
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
import numpy as np
import gym
from inspect import findsource
import math
import operator
from functools import reduce

import numpy as np
import gym
from gym import error, spaces, utils
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env = wrappers.FlatObsWrapper(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

class FlatObsWrapper_LangV2(gym.core.ObservationWrapper):
    """
    Better Language process than FlagObsWrapper
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.colors=['blue','green','yellow','red','purple','grey']
        self.enrich_lanecoding = 10
        self.numCharCodes = len(self.colors)

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.enrich_lanecoding,),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None
    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            # assert len(mission) <= self.maxStrLen, 'mission string too long ({} chars)'.format(len(mission))
            mission = mission.lower()
            strArray = np.zeros(shape=(self.enrich_lanecoding, self.numCharCodes), dtype='float32')

            find_str=False
            for idx,c in enumerate(self.colors):
                if c in mission:
                    strArray[:,idx]=1
                    find_str=True
            assert find_str==True,"not find str"
            
            # for idx, ch in enumerate(mission):
            #     if ch >= 'a' and ch <= 'z':
            #         chNo = ord(ch) - ord('a')
            #     elif ch == ' ':
            #         chNo = ord('z') - ord('a') + 1
            #     assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
            #     strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs

if __name__=="__main__":


    env_id="MiniGrid-Fetch-6x6-N2-v0"
    env = gym.make(env_id)
    env = wrappers.FlatObsWrapper_LangV2(env)
    env.env.env.max_steps=15

    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=150000, log_interval=4)
    # model.save("dqn_cartpole2")

    model = PPO.load("dqn_cartpole2")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        print('reward : {}, action : {}'.format(reward,action))
        time.sleep(0.1)
        if done:
            obs = env.reset()