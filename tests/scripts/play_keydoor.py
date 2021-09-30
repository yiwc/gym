import gym
import rl_helper
from rl_helper import fps
from rl_helper import envhelper
from gym.envs.minigrid.envs.keycorridor import KeyCorridorS3R2
if __name__=="__main__":


    # Asynchronous Environments Test
    # recorder = envhelper()
    env = gym.vector.make('MiniGrid-KeyCorridorS4R3-v0',10,asynchronous=True)
    env.reset()
    # print(env.metadata)
    # fps(env)
    frames=[]
    for episode in range(2): 
        obs = env.reset()
        for step in range(15):
            # print(step)
            action = env.action_space.sample()  # or given a custom model, action = policy(observation)
            nobs, reward, done, info = env.step(action)
            if done.any():
                break
            print(nobs['image'].mean())
            print(nobs['image'].shape)
            # recorder.recording(env)
            # env.render()
    # env.close()
    # recorder.save_gif()