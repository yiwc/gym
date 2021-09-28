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

class VecExtractDictObsLanguage(VecEnvWrapper):
    """
    A vectorized wrapper for filtering a specific key from dictionary observations.
    Similar to Gym's FilterObservation wrapper:
        https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

    :param venv: The vectorized environment
    :param key: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv):

        # observation_space = gym.spaces.Box(low=-1,high=1,shape=)

        super().__init__(venv=venv, observation_space=venv.observation_space.spaces['image'])


    def _language_infer(self,sentences):
        colors=['yellow','blue','red','purple','grey','green']
        return np.ones([len(sentences),5])

    def _preprocess_obses(self,obses: list):
        batch_obs=OrderedDict([(k, np.stack([o[k] for o in obses])) for k in obses[0].keys()])
        batch_img=batch_obs['image']
        batch_sentences=batch_obs['mission']
        batch_sentences_encoding=self._language_infer(batch_sentences)
        batch_img_reshape=batch_img.reshape(10,-1)
        arrays=[batch_img_reshape,batch_sentences_encoding]
        res=np.hstack(arrays)
        return res
    def reset(self) -> np.ndarray:

        for remote in self.venv.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.venv.remotes]
        obs = self._preprocess_obses(obs)
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in  self.venv.remotes]
        self.venv.waiting = False
        obs, rews, dones, infos = zip(*results)
        obs = self._preprocess_obses(obs)
        return obs, rews, dones, infos

class VecExtractDictObs(VecEnvWrapper):
    """
    A vectorized wrapper for filtering a specific key from dictionary observations.
    Similar to Gym's FilterObservation wrapper:
        https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

    :param venv: The vectorized environment
    :param key: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv, key: str):
        self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs[self.key]

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs[self.key], reward, done, info

if __name__=="__main__":

    # env_id="MiniGrid-GoToDoor-5x5-v0"
    # env = SubprocVecEnv([make_env(env_id, i) for i in range(10)])


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